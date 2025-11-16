#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrajectoryInterpolator - Linear and parabolic trajectory interpolation.

Handles:
- Linear interpolation for short movements
- Parabolic flight trajectories for long gaps
- Gap filling in detection history
- Backward interpolation for smooth camera movement
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("panorama-virtualcam")


class TrajectoryInterpolator:
    """
    Interpolation engine for ball trajectories.

    Features:
    - Linear interpolation for nearby points
    - Parabolic trajectories for ball flight (gaps > 1s)
    - Configurable gap limits (max_gap)
    - Backward interpolation for camera tracking
    """

    def __init__(self, fps=30, max_gap=10.0):
        """
        Initialize the interpolator.

        Args:
            fps: Frames per second for interpolation (default: 30)
            max_gap: Maximum gap to interpolate in seconds (default: 10.0)
        """
        self.fps = fps
        self.max_gap = max_gap

    def interpolate_between_points(self, det1, det2, ts1, ts2, target_ts):
        """
        Interpolate between two detections for a specific timestamp.

        Uses:
        - Linear interpolation for short gaps (<= 1s)
        - Parabolic trajectory for long gaps (> 1s, likely ball flight)

        Args:
            det1: First detection
            det2: Second detection
            ts1: Timestamp of first detection
            ts2: Timestamp of second detection
            target_ts: Target timestamp for interpolation

        Returns:
            Interpolated detection
        """
        if not det1 or not det2:
            return det1 or det2

        gap = ts2 - ts1
        t = (target_ts - ts1) / gap if gap > 0 else 0.5
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

        result = list(det1)

        # Interpolate positions and sizes
        # Local positions (indices 0-3)
        for i in [0, 1, 2, 3]:
            if i < len(det1) and i < len(det2):
                result[i] = det1[i] + (det2[i] - det1[i]) * t

        # Global positions (indices 6-9)
        for i in [6, 7, 8, 9]:
            if i < len(det1) and i < len(det2):
                result[i] = det1[i] + (det2[i] - det1[i]) * t

        # For large gaps (ball flight) add parabolic trajectory
        if gap > 1.0:  # More than 1 second - likely flight
            # Parabola height depends on distance
            if len(det1) > 6 and len(det2) > 6:
                dx = abs(det2[6] - det1[6])
                dy = abs(det2[7] - det1[7])
                distance = math.sqrt(dx*dx + dy*dy)

                # Flight height proportional to distance
                max_height = min(150, distance * 0.1)

                # Parabolic correction for Y (4t(1-t) gives parabola with max at t=0.5)
                parabola_factor = 4 * t * (1 - t)
                y_offset = max_height * parabola_factor

                # Apply correction (up is negative Y)
                result[1] -= y_offset  # Local Y
                result[7] -= y_offset  # Global Y

                # Ball size also changes during flight
                size_factor = 1.0 + (y_offset / 200)  # Increases when higher
                result[2] *= size_factor
                result[3] *= size_factor
                result[8] *= size_factor
                result[9] *= size_factor

        # Confidence: interpolate or take minimum
        if len(det1) > 4 and len(det2) > 4:
            result[4] = min(det1[4], det2[4]) * 0.8  # Reduce confidence for interpolated

        return result

    def interpolate_history_gaps(self, history, fps=None, max_gap=None):
        """
        Interpolate gaps in detection history with support for long ball flights.

        Args:
            history: Dictionary of timestamp -> detection mappings
            fps: Frames per second (uses instance fps if None)
            max_gap: Maximum gap to interpolate in seconds (uses instance max_gap if None)

        Returns:
            Dictionary with interpolated detections added
        """
        if fps is None:
            fps = self.fps
        if max_gap is None:
            max_gap = self.max_gap

        if len(history) < 2:
            return history

        frame_dt = 1.0 / float(fps)
        interpolated = dict(history)
        timestamps = sorted(history.keys())
        added_count = 0

        for i in range(len(timestamps) - 1):
            ts1, ts2 = timestamps[i], timestamps[i + 1]
            det1, det2 = history[ts1], history[ts2]

            gap = ts2 - ts1

            # Skip very small gaps
            if gap <= frame_dt * 1.5:
                continue

            # Interpolate ANY gaps up to max_gap
            if gap > max_gap:
                logger.warning(f"🕳️ GAP TOO LARGE: {gap:.2f}s between ts1={ts1:.2f} and ts2={ts2:.2f}, "
                              f"pos1=({det1[6]:.0f},{det1[7]:.0f}), pos2=({det2[6]:.0f},{det2[7]:.0f})")
                continue

            # Log large gaps (possible out)
            if gap > 1.0:
                dx = det2[6] - det1[6]
                dy = det2[7] - det1[7]
                distance = math.sqrt(dx*dx + dy*dy)
                logger.info(f"🔄 INTERPOLATING GAP: {gap:.2f}s, dist={distance:.0f}px, "
                           f"from ({det1[6]:.0f},{det1[7]:.0f}) to ({det2[6]:.0f},{det2[7]:.0f})")

            if not det1 or not det2 or len(det1) < 10 or len(det2) < 10:
                continue

            num_frames = int(gap * fps) - 1
            if num_frames <= 0:
                continue

            # Determine movement type by distance and time
            dx = det2[6] - det1[6] if len(det1) > 6 else 0
            dy = det2[7] - det1[7] if len(det1) > 7 else 0
            distance = math.sqrt(dx*dx + dy*dy)

            # If large distance in short time - it's flight
            is_flight = (gap > 0.5 and distance > 500) or gap > 1.5

            for j in range(1, num_frames + 1):
                w_ratio = j / (num_frames + 1)
                new_ts = ts1 + j * frame_dt

                if is_flight:
                    # Use helper function for flight
                    new_det = self.interpolate_between_points(
                        det1, det2, ts1, ts2, new_ts
                    )
                else:
                    # Simple linear interpolation for short movements
                    new_det = list(det1)

                    # Interpolate all coordinates linearly
                    new_det[0] = det1[0] + (det2[0] - det1[0]) * w_ratio
                    new_det[1] = det1[1] + (det2[1] - det1[1]) * w_ratio
                    new_det[2] = det1[2] + (det2[2] - det1[2]) * w_ratio
                    new_det[3] = det1[3] + (det2[3] - det1[3]) * w_ratio
                    new_det[6] = det1[6] + (det2[6] - det1[6]) * w_ratio
                    new_det[7] = det1[7] + (det2[7] - det1[7]) * w_ratio
                    new_det[8] = det1[8] + (det2[8] - det1[8]) * w_ratio
                    new_det[9] = det1[9] + (det2[9] - det1[9]) * w_ratio

                    # Reduce confidence for interpolated points
                    new_det[4] = min(det1[4], det2[4]) * 0.7

                interpolated[new_ts] = new_det
                added_count += 1

        if added_count > 0:
            logger.debug(f"Added {added_count} interpolated points")

        return interpolated

    def insert_backward_interpolation(self, processed_history, start_ts, end_ts, start_pos, end_pos):
        """
        Insert synthetic detections for smooth camera movement (backward in time).

        Used when ball is rediscovered after being lost - creates smooth
        transition by inserting points backward in time.

        Args:
            processed_history: Dictionary to insert into (modified in place)
            start_ts: Start time of interpolation (usually end_ts - 1.0)
            end_ts: Time of ball rediscovery
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position

        Returns:
            Number of points inserted
        """
        duration = end_ts - start_ts
        if duration <= 0:
            return 0

        # Generate 30 points per second
        num_points = int(duration * 30)
        if num_points < 1:
            return 0

        inserted = 0
        for i in range(num_points):
            t = start_ts + (i / num_points) * duration
            alpha = i / num_points  # 0.0 → 1.0

            # Linear interpolation
            x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
            y = start_pos[1] + alpha * (end_pos[1] - start_pos[1])

            # Create synthetic detection (same format as real detection)
            synthetic_det = [
                int(x), int(y),     # cx, cy (local)
                15.0, 15.0,         # width, height (fixed size)
                0.5, 0,             # confidence, class_id
                int(x), int(y),     # cx_global, cy_global
                15.0, 15.0          # width_global, height_global
            ]

            # Insert into processed_future_history (so it appears in playback)
            processed_history[float(t)] = synthetic_det
            inserted += 1

        logger.info(f"🔄 BACKWARD INTERP: inserted {inserted} points from "
                   f"({start_pos[0]:.0f},{start_pos[1]:.0f}) to ({end_pos[0]:.0f},{end_pos[1]:.0f}) "
                   f"over {duration:.1f}s (ts {start_ts:.2f}→{end_ts:.2f})")

        return inserted

    def find_closest_points(self, history, timestamp):
        """
        Find the closest points before and after a timestamp.

        Args:
            history: Dictionary of timestamp -> detection mappings
            timestamp: Target timestamp

        Returns:
            Tuple of (before_ts, after_ts) or (None, None) if not found
        """
        if not history:
            return None, None

        times = sorted(history.keys())
        before_ts = None
        after_ts = None

        for t in times:
            if t <= timestamp:
                before_ts = t
            elif t > timestamp and after_ts is None:
                after_ts = t
                break

        return before_ts, after_ts

    def get_interpolation_info(self):
        """
        Get interpolator configuration.

        Returns:
            Dictionary with fps and max_gap settings
        """
        return {
            'fps': self.fps,
            'max_gap': self.max_gap
        }
