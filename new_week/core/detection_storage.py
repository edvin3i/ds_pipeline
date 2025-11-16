#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DetectionStorage - Three-tier storage management for ball detections.

Manages:
- raw_future_history: Incoming detections
- processed_future_history: Processed and interpolated detections
- confirmed_history: Detections that have been displayed
"""

import threading
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger("panorama-virtualcam")


class DetectionStorage:
    """
    Three-tier storage system for ball detections with thread-safe access.

    Storage tiers:
    1. raw_future_history: Raw incoming detections
    2. processed_future_history: Processed/interpolated detections
    3. confirmed_history: Displayed detections (limited to max_confirmed_points)
    """

    def __init__(self, history_duration=10.0, cleanup_interval=1000, max_confirmed_points=200):
        """
        Initialize the three-tier storage system.

        Args:
            history_duration: Duration to keep history (seconds)
            cleanup_interval: Interval for cleanup operations
            max_confirmed_points: Maximum number of confirmed detections to keep
        """
        # Three-tier storage
        self.confirmed_history = {}
        self.raw_future_history = {}
        self.processed_future_history = {}

        # Aliases for backward compatibility
        self.history = self.raw_future_history
        self.interpolated_history = self.processed_future_history

        # Thread safety
        self.history_lock = threading.RLock()

        # Tracking
        self.current_display_timestamp = 0.0
        self.max_confirmed_points = max_confirmed_points
        self.frame_index = {}

        # Last detection tracking (for duplicate filtering)
        self.last_detection = None
        self.last_detection_time = 0
        self.last_detection_return = None

        # Configuration
        self.history_duration = float(history_duration)
        self.cleanup_interval = int(cleanup_interval)
        self.frame_counter = 0

    def add_detection(self, detection, timestamp, frame_num):
        """
        Add a detection to raw_future_history with duplicate filtering.

        Args:
            detection: Detection data (list/array with bbox coordinates)
            timestamp: Detection timestamp
            frame_num: Frame number
        """
        # Debug logging
        if detection is not None:
            logger.info(f"🔵 CALL add_detection: ts={timestamp:.2f}, pos=({detection[6]:.0f},{detection[7]:.0f})")
        else:
            logger.info(f"🔵 CALL add_detection: ts={timestamp:.2f}, detection=None")

        if detection is None:
            return

        # Filter duplicates by global coordinate proximity
        if self.last_detection is not None:
            prev_x, prev_y = self.last_detection[6], self.last_detection[7]
            curr_x, curr_y = detection[6], detection[7]
            dx = abs(curr_x - prev_x)
            dy = abs(curr_y - prev_y)
            if dx <= 2 and dy <= 2:
                # Log duplicates (limited frequency)
                if not hasattr(self, '_dup_log_counter'):
                    self._dup_log_counter = 0
                self._dup_log_counter += 1
                if self._dup_log_counter % 5 == 0:  # Every 5th duplicate
                    logger.info(f"⛔ DUPLICATE #{self._dup_log_counter}: pos=({curr_x:.0f},{curr_y:.0f}), "
                               f"prev=({prev_x:.0f},{prev_y:.0f}), diff=({dx},{dy}), "
                               f"time_since_last={timestamp - self.last_detection_time:.2f}s")
                return

        self.last_detection = list(detection)
        self.last_detection_time = float(timestamp)

        with self.history_lock:
            self.raw_future_history[float(timestamp)] = detection
            self.frame_index[int(frame_num)] = float(timestamp)

            # Log addition every 30 frames
            if self.frame_counter % 30 == 0:
                logger.info(f"📥 ADD: ts={timestamp:.2f}, pos=({detection[6]:.0f},{detection[7]:.0f}), "
                           f"hist_size: raw={len(self.raw_future_history)}, "
                           f"proc={len(self.processed_future_history)}, conf={len(self.confirmed_history)}")

            self.frame_counter += 1

    def update_display_timestamp(self, timestamp):
        """
        Update the current display timestamp.

        Args:
            timestamp: Current display timestamp
        """
        with self.history_lock:
            self.current_display_timestamp = float(timestamp)

    def get_raw_history(self):
        """Get a copy of raw_future_history."""
        with self.history_lock:
            return dict(self.raw_future_history)

    def get_processed_history(self):
        """Get a copy of processed_future_history."""
        with self.history_lock:
            return dict(self.processed_future_history)

    def get_confirmed_history(self):
        """Get a copy of confirmed_history."""
        with self.history_lock:
            return dict(self.confirmed_history)

    def set_processed_history(self, history):
        """
        Set processed_future_history.

        Args:
            history: Dictionary of timestamp -> detection mappings
        """
        with self.history_lock:
            self.processed_future_history = dict(history)
            self.interpolated_history = self.processed_future_history

    def get_detection_exact(self, timestamp):
        """
        Get exact detection for a timestamp (no interpolation).

        Args:
            timestamp: Timestamp to query

        Returns:
            Detection if found, None otherwise
        """
        with self.history_lock:
            # Check processed history first
            exact = self.processed_future_history.get(timestamp)
            if exact:
                return exact.copy()

            # Check confirmed history
            exact = self.confirmed_history.get(timestamp)
            if exact:
                return exact.copy()

            return None

    def get_all_history_combined(self):
        """
        Get combined history (confirmed + processed).

        Returns:
            Dictionary combining confirmed_history and processed_future_history
        """
        with self.history_lock:
            all_history = {}
            all_history.update(self.confirmed_history)
            all_history.update(self.processed_future_history)
            return all_history

    def transfer_displayed_to_confirmed(self):
        """
        Transfer displayed detections from processed to confirmed storage.
        Removes detections that have been displayed from both raw and processed history.
        """
        moved = 0
        with self.history_lock:
            for ts in list(self.processed_future_history.keys()):
                if float(ts) <= float(self.current_display_timestamp):
                    self.confirmed_history[ts] = self.processed_future_history[ts]
                    del self.processed_future_history[ts]
                    moved += 1
            for ts in list(self.raw_future_history.keys()):
                if float(ts) <= float(self.current_display_timestamp):
                    del self.raw_future_history[ts]
        return moved

    def cleanup_confirmed_history(self):
        """
        Remove old confirmed detections to maintain max_confirmed_points limit.
        """
        with self.history_lock:
            if len(self.confirmed_history) <= self.max_confirmed_points:
                return
            sorted_ts = sorted(self.confirmed_history.keys())
            to_remove = len(self.confirmed_history) - self.max_confirmed_points
            for i in range(to_remove):
                del self.confirmed_history[sorted_ts[i]]

    def get_context_from_confirmed(self, num_points=30):
        """
        Get the most recent N points from confirmed history for context.

        Args:
            num_points: Number of recent points to retrieve

        Returns:
            Dictionary of timestamp -> detection mappings
        """
        with self.history_lock:
            if not self.confirmed_history:
                return {}

            sorted_ts = sorted(self.confirmed_history.keys())
            start_idx = max(0, len(sorted_ts) - num_points)

            context = {}
            for ts in sorted_ts[start_idx:]:
                context[ts] = self.confirmed_history[ts]

            return context

    def get_last_confirmed_detection(self):
        """
        Get the last confirmed detection.

        Returns:
            Dictionary with timestamp, x, y or None
        """
        with self.history_lock:
            if not self.confirmed_history:
                return None

            last_ts = max(self.confirmed_history.keys())
            det = self.confirmed_history[last_ts]
            if det and len(det) >= 2:
                return {
                    'timestamp': float(last_ts),
                    'x': float(det[6] if len(det) > 6 else det[0]),
                    'y': float(det[7] if len(det) > 7 else det[1])
                }
            return None

    def clear_all(self):
        """Clear all storage tiers."""
        with self.history_lock:
            self.raw_future_history.clear()
            self.processed_future_history.clear()
            self.confirmed_history.clear()
            self.frame_index.clear()
