#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HistoryManager - Main orchestrator for ball detection history.

Composes:
- DetectionStorage: Three-tier storage management
- TrajectoryFilter: Outlier detection and blacklist
- TrajectoryInterpolator: Trajectory interpolation

Provides high-level interface for:
- Adding detections
- Retrieving detections with interpolation
- Getting future trajectories
- History processing and cleanup
"""

import time
import math
import logging
from typing import Dict, List, Optional, Any, Tuple

from .detection_storage import DetectionStorage
from .trajectory_filter import TrajectoryFilter
from .trajectory_interpolator import TrajectoryInterpolator

logger = logging.getLogger("panorama-virtualcam")


class HistoryManager:
    """
    Main orchestrator for ball detection history management.

    Delegates to:
    - DetectionStorage: Storage operations
    - TrajectoryFilter: Outlier filtering
    - TrajectoryInterpolator: Trajectory interpolation
    """

    def __init__(self, history_duration=10.0, cleanup_interval=1000):
        """
        Initialize the history manager with all components.

        Args:
            history_duration: Duration to keep history (seconds)
            cleanup_interval: Interval for cleanup operations
        """
        # Initialize components
        self.storage = DetectionStorage(
            history_duration=history_duration,
            cleanup_interval=cleanup_interval,
            max_confirmed_points=200
        )
        self.filter = TrajectoryFilter(
            outlier_ban_threshold=6,
            ban_radius=30
        )
        self.interpolator = TrajectoryInterpolator(
            fps=30,
            max_gap=10.0
        )

        # Processing state
        self.last_full_process_time = 0

        # Expose storage properties for backward compatibility
        self.history_lock = self.storage.history_lock
        self.confirmed_history = self.storage.confirmed_history
        self.raw_future_history = self.storage.raw_future_history
        self.processed_future_history = self.storage.processed_future_history
        self.history = self.storage.history
        self.interpolated_history = self.storage.interpolated_history
        self.current_display_timestamp = self.storage.current_display_timestamp
        self.frame_counter = self.storage.frame_counter
        self.last_detection_return = self.storage.last_detection_return

        # Expose filter properties
        self.permanent_blacklist = self.filter.permanent_blacklist
        self.outlier_removal_count = self.filter.outlier_removal_count

    def add_detection(self, detection, timestamp, frame_num):
        """
        Add detection and trigger processing.

        Args:
            detection: Detection data
            timestamp: Detection timestamp
            frame_num: Frame number
        """
        self.storage.add_detection(detection, timestamp, frame_num)

        # Update local references
        self.frame_counter = self.storage.frame_counter

        # Trigger processing
        self._process_future_history()

    def update_display_timestamp(self, timestamp):
        """
        Update the current display timestamp.

        Args:
            timestamp: Current display timestamp
        """
        self.storage.update_display_timestamp(timestamp)
        self.current_display_timestamp = self.storage.current_display_timestamp

    def get_detection_for_timestamp(self, timestamp, max_delta=0.12):
        """
        Find detection for a given timestamp with on-the-fly interpolation.

        Args:
            timestamp: Target timestamp
            max_delta: Maximum time difference for nearest match

        Returns:
            Detection data or None
        """
        with self.storage.history_lock:
            # IMPORTANT: Don't update current_display_timestamp here!
            # It's done via update_display_timestamp() separately
            # Otherwise playback (which LAGS by 7 sec) will delete
            # fresh detections from analysis!

            # Don't call _process_future_history() here!
            # It's called in add_detection() when new detections arrive

            # First look for exact match
            exact = self.storage.get_detection_exact(timestamp)
            if exact:
                return exact

            # Now interpolate between neighboring points
            all_history = self.storage.get_all_history_combined()

            if not all_history:
                # Log if history is empty
                if self.storage.frame_counter % 30 == 0:
                    logger.warning(f"📭 HISTORY EMPTY: ts={timestamp:.2f}, no detections in history")
                return self.storage.last_detection_return.copy() if self.storage.last_detection_return else None

            times = sorted(all_history.keys())
            before_ts = None
            after_ts = None

            for t in times:
                if t <= timestamp:
                    before_ts = t
                elif t > timestamp and after_ts is None:
                    after_ts = t
                    break

            # Interpolation between points
            if before_ts and after_ts:
                gap = after_ts - before_ts

                # Check: if gap > 3 seconds - DON'T interpolate (clear ball loss)
                if gap > 3.0:
                    if self.storage.frame_counter % 30 == 0:
                        logger.warning(f"⚠️ GAP TOO LARGE for interpolation: {gap:.2f}s "
                                      f"between {before_ts:.2f} and {after_ts:.2f}, "
                                      f"requested ts={timestamp:.2f} → switching to players fallback")

                    # Try using nearest point if close enough
                    if abs(before_ts - timestamp) < abs(after_ts - timestamp):
                        if abs(before_ts - timestamp) < max_delta:
                            det = all_history[before_ts].copy()
                            self.storage.last_detection_return = det.copy()
                            return det
                    else:
                        if abs(after_ts - timestamp) < max_delta:
                            det = all_history[after_ts].copy()
                            self.storage.last_detection_return = det.copy()
                            return det

                    # Gap too large - return None to switch to players
                    return None

                # Normal interpolation for gaps <= 3 seconds
                det = self.interpolator.interpolate_between_points(
                    all_history[before_ts],
                    all_history[after_ts],
                    before_ts,
                    after_ts,
                    timestamp
                )
                self.storage.last_detection_return = det.copy()
                return det

            # If only one point before or after
            if before_ts and abs(before_ts - timestamp) < max_delta:
                det = all_history[before_ts].copy()
                self.storage.last_detection_return = det.copy()
                return det

            if after_ts and abs(after_ts - timestamp) < max_delta:
                det = all_history[after_ts].copy()
                self.storage.last_detection_return = det.copy()
                return det

            # Didn't find suitable detection
            if self.storage.frame_counter % 30 == 0:
                logger.warning(f"⚠️ NO MATCH: ts={timestamp:.2f}, before={before_ts:.2f if before_ts else None}, "
                              f"after={after_ts:.2f if after_ts else None}, hist_size={len(all_history)}")

            return self.storage.last_detection_return.copy() if self.storage.last_detection_return else None

    def get_future_trajectory(self, current_timestamp, look_ahead_seconds=1.0, max_points=10):
        """
        Get future ball trajectory relative to current time.

        Args:
            current_timestamp: Current display time
            look_ahead_seconds: How many seconds ahead to look
            max_points: Maximum number of points

        Returns:
            List[dict]: List of points with fields 'time', 'x', 'y', 'width'
        """
        with self.storage.history_lock:
            future_points = []

            # Define time range
            start_time = float(current_timestamp)
            end_time = start_time + float(look_ahead_seconds)

            # Collect points from processed future history
            for ts, det in self.storage.processed_future_history.items():
                if start_time <= float(ts) <= end_time and det:
                    future_points.append({
                        'time': float(ts),
                        'x': float(det[0]),
                        'y': float(det[1]),
                        'width': float(det[2]) if len(det) > 2 else 0
                    })

            # If few points in processed, add from confirmed
            if len(future_points) < 3:
                for ts, det in self.storage.confirmed_history.items():
                    if float(ts) > start_time and float(ts) <= end_time and det:
                        future_points.append({
                            'time': float(ts),
                            'x': float(det[0]),
                            'y': float(det[1]),
                            'width': float(det[2]) if len(det) > 2 else 0
                        })

            # Sort by time and limit count
            future_points.sort(key=lambda p: p['time'])
            return future_points[:max_points]

    def _process_future_history(self):
        """
        Full history processing with aggressive interpolation.

        Steps:
        1. Transfer displayed detections to confirmed
        2. Cleanup old confirmed detections
        3. Get context from confirmed history
        4. Detect and remove outliers (with rate limiting)
        5. Clean detection history
        6. Extract future-only portion
        7. Interpolate gaps
        """
        # Check if called too frequently
        current_time = time.time()

        # Limit heavy processing frequency
        time_since_last = current_time - self.last_full_process_time
        need_heavy_processing = (
            time_since_last >= 0.5 or  # At least 0.5 sec passed
            len(self.storage.raw_future_history) >= 10  # Or lots of data accumulated
        )

        # Always do light operations
        self.storage.transfer_displayed_to_confirmed()
        self.storage.cleanup_confirmed_history()

        # Process even if few points
        if len(self.storage.raw_future_history) >= 2:
            # Get context from confirmed history
            context_points = self.storage.get_context_from_confirmed(num_points=30)

            # Combine context with raw history
            combined_history = {}
            combined_history.update(context_points)
            combined_history.update(self.storage.raw_future_history)

            # Heavy cleanup only if needed
            if need_heavy_processing:
                # Outlier removal
                cleaned_combined = self.filter.detect_and_remove_false_trajectories(combined_history)

                # Additional cleanup
                refined_combined = self.clean_detection_history(
                    cleaned_combined,
                    preserve_recent_seconds=0.3,
                    outlier_threshold=2.5,
                    window_size=3
                )
                self.last_full_process_time = current_time
            else:
                # Without cleanup, use as is
                refined_combined = combined_history

            # Extract only future part
            lookback_buffer = 1.0
            cutoff_time = self.storage.current_display_timestamp - lookback_buffer

            future_only = {
                ts: det for ts, det in refined_combined.items()
                if ts > cutoff_time
            }

            # ALWAYS interpolate (it's fast)
            interpolated = self.interpolator.interpolate_history_gaps(
                future_only,
                fps=30,
                max_gap=10.0  # Support long flights
            )

            self.storage.set_processed_history(interpolated)

            # Update references
            self.processed_future_history = self.storage.processed_future_history
            self.interpolated_history = self.storage.interpolated_history

    def clean_detection_history(self, history, preserve_recent_seconds=0.5,
                               outlier_threshold=2.5, window_size=3):
        """
        Simplified additional cleanup.

        Args:
            history: History to clean
            preserve_recent_seconds: Preserve recent detections
            outlier_threshold: Outlier detection threshold
            window_size: Window size for analysis

        Returns:
            Cleaned history
        """
        if len(history) < 5:
            return history
        return history

    def get_last_confirmed_detection(self):
        """
        Get the last confirmed detection.

        Returns:
            Dictionary with timestamp, x, y or None
        """
        return self.storage.get_last_confirmed_detection()

    def insert_backward_interpolation(self, start_ts, end_ts, start_pos, end_pos):
        """
        Insert synthetic detections for smooth camera movement.

        Args:
            start_ts: Start time of interpolation (usually end_ts - 1.0)
            end_ts: Time of ball rediscovery
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position
        """
        with self.storage.history_lock:
            self.interpolator.insert_backward_interpolation(
                self.storage.processed_future_history,
                start_ts,
                end_ts,
                start_pos,
                end_pos
            )

            # Update references
            self.processed_future_history = self.storage.processed_future_history
            self.interpolated_history = self.storage.interpolated_history

    def is_point_banned(self, x, y):
        """
        Check if a point is in a banned zone.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if banned, False otherwise
        """
        return self.filter.is_point_banned(x, y)

    def get_stats(self):
        """
        Get statistics about the history manager.

        Returns:
            Dictionary with various statistics
        """
        with self.storage.history_lock:
            return {
                'raw_size': len(self.storage.raw_future_history),
                'processed_size': len(self.storage.processed_future_history),
                'confirmed_size': len(self.storage.confirmed_history),
                'current_display_timestamp': self.storage.current_display_timestamp,
                'frame_counter': self.storage.frame_counter,
                'blacklist_size': len(self.filter.permanent_blacklist),
                'outlier_counts': len(self.filter.outlier_removal_count)
            }
