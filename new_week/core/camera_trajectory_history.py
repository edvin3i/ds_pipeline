#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CameraTrajectoryHistory - Smooth camera movement tracking.

Builds camera trajectory from:
- Interpolated ball detection history
- Player center-of-mass (fallback for ball loss > 3 sec)
- Applies smoothing to remove outliers
- Interpolates between points for smooth camera motion
"""

import math
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger("panorama-virtualcam")


class CameraTrajectoryHistory:
    """
    Manages smooth camera trajectory based on ball and player positions.

    Features:
    - Loads data from interpolated ball history
    - Falls back to player positions when ball is lost (gap > 3 sec)
    - Smooths trajectory to remove outliers (e.g., 1000px jumps)
    - Interpolates between points for fluid camera movement
    """

    def __init__(self, history_duration=10.0, max_gap=3.0, outlier_threshold=300):
        """
        Initialize camera trajectory history.

        Args:
            history_duration: Duration to keep history (seconds)
            max_gap: Maximum gap before switching to players (seconds)
            outlier_threshold: Distance threshold for outlier detection (pixels)
        """
        self.camera_trajectory = {}  # {timestamp → point_dict}
        self.history_duration = float(history_duration)
        self.max_gap = float(max_gap)
        self.outlier_threshold = float(outlier_threshold)

    def populate_from_ball_and_players(self, ball_history_dict, players_history):
        """
        Fill camera trajectory from ball history and player positions.

        Args:
            ball_history_dict: Dictionary of timestamp → detection (from processed_future_history)
            players_history: PlayersHistory instance for fallback to center-of-mass

        Algorithm:
        1. Load all ball detections (original YOLO or interpolated)
        2. For gaps > max_gap: substitute with player center-of-mass
        3. Interpolate transitions from ball to player positions
        """
        if not ball_history_dict:
            logger.warning("🚨 CAMERA_TRAJ: Empty ball history")
            return

        self.camera_trajectory.clear()

        # Single-pass algorithm: iterate through ball history and fill gaps immediately
        sorted_timestamps = sorted(ball_history_dict.keys())
        if not sorted_timestamps:
            return

        for i, ts in enumerate(sorted_timestamps):
            detection = ball_history_dict[ts]
            if not detection or len(detection) < 8:
                continue

            is_interpolated = detection[10] if len(detection) > 10 else False
            source_type = 'interpolated_ball' if is_interpolated else 'ball'

            # Add current ball point
            self.camera_trajectory[float(ts)] = {
                'x': float(detection[6]),
                'y': float(detection[7]),
                'timestamp': float(ts),
                'source_type': source_type,
                'confidence': float(detection[4]) if len(detection) > 4 else 0.5
            }

            # Check gap to next ball point
            if i + 1 < len(sorted_timestamps):
                ts_next = sorted_timestamps[i + 1]
                gap = ts_next - ts

                # If gap > max_gap, fill with player COM positions
                if gap > self.max_gap:
                    logger.info(f"🔄 CAMERA_TRAJ: Gap {gap:.2f}s > {self.max_gap}s at ts={ts:.2f}→{ts_next:.2f}, "
                               f"filling with player positions")

                    next_detection = ball_history_dict[ts_next]
                    next_x = float(next_detection[6])
                    next_y = float(next_detection[7])

                    # Fill gap with player COM every 2 seconds
                    # Continue until within 2 seconds of next ball position
                    current_ts = ts
                    fill_step = 2.0

                    while current_ts + fill_step < ts_next - 2.0:
                        current_ts += fill_step

                        # Get player center of mass
                        player_com = players_history.calculate_center_of_mass(current_ts)

                        if player_com:
                            self.camera_trajectory[float(current_ts)] = {
                                'x': float(player_com[0]),
                                'y': float(player_com[1]),
                                'timestamp': float(current_ts),
                                'source_type': 'player',
                                'confidence': 0.35
                            }
                            logger.info(f"  ➕ Player COM at ts={current_ts:.2f}: ({player_com[0]:.0f}, {player_com[1]:.0f})")

                    # Add smooth transition blend point (50% player, 50% next ball)
                    if current_ts < ts_next - 0.1:
                        transition_ts = current_ts + (ts_next - current_ts) * 0.5
                        player_com = players_history.calculate_center_of_mass(transition_ts)

                        if player_com:
                            alpha = 0.5
                            blend_x = (1 - alpha) * player_com[0] + alpha * next_x
                            blend_y = (1 - alpha) * player_com[1] + alpha * next_y

                            self.camera_trajectory[float(transition_ts)] = {
                                'x': blend_x,
                                'y': blend_y,
                                'timestamp': float(transition_ts),
                                'source_type': 'player',
                                'confidence': 0.3
                            }
                            logger.info(f"  ➕ Blend at ts={transition_ts:.2f}: ({blend_x:.0f}, {blend_y:.0f})")

        logger.info(f"📍 CAMERA_TRAJ: Loaded {len(self.camera_trajectory)} points (ball + player fills)")

    def smooth_trajectory(self, window_size=5, threshold_px=None):
        """
        Smooth trajectory to remove outliers (e.g., detection jumps).

        Args:
            window_size: Window size for median filtering
            threshold_px: Distance threshold for outlier detection (uses outlier_threshold if None)

        Algorithm:
        - For each point: if distance to neighbors > threshold AND detour > 1.5x
        - Replace with median of neighbors
        """
        if threshold_px is None:
            threshold_px = self.outlier_threshold

        if len(self.camera_trajectory) < 3:
            return

        times = sorted(self.camera_trajectory.keys())
        outliers_removed = 0

        for i in range(1, len(times) - 1):
            prev_t, curr_t, next_t = times[i - 1], times[i], times[i + 1]

            prev_point = self.camera_trajectory[prev_t]
            curr_point = self.camera_trajectory[curr_t]
            next_point = self.camera_trajectory[next_t]

            # Calculate distances
            dist_to_prev = math.sqrt((curr_point['x'] - prev_point['x']) ** 2 +
                                     (curr_point['y'] - prev_point['y']) ** 2)
            dist_to_next = math.sqrt((curr_point['x'] - next_point['x']) ** 2 +
                                     (curr_point['y'] - next_point['y']) ** 2)
            dist_prev_next = math.sqrt((next_point['x'] - prev_point['x']) ** 2 +
                                       (next_point['y'] - prev_point['y']) ** 2)

            # Check for outlier: large detour (jump and return)
            if dist_to_prev > threshold_px and dist_to_next > threshold_px:
                if dist_prev_next < max(dist_to_prev, dist_to_next) * 0.7:
                    # This is an outlier - replace with median
                    smoothed_x = (prev_point['x'] + next_point['x']) / 2
                    smoothed_y = (prev_point['y'] + next_point['y']) / 2

                    curr_point['x'] = smoothed_x
                    curr_point['y'] = smoothed_y

                    logger.warning(f"🔧 SMOOTHED outlier at {curr_t:.2f}: "
                                 f"jump={dist_to_prev:.0f}px→{dist_to_next:.0f}px, "
                                 f"direct={dist_prev_next:.0f}px")
                    outliers_removed += 1

        if outliers_removed > 0:
            logger.info(f"🔧 CAMERA_TRAJ: Removed {outliers_removed} outliers via smoothing")

    def interpolate_gaps(self, fps=30):
        """
        Interpolate between trajectory points for smooth camera motion.

        Args:
            fps: Frames per second for interpolation density

        Adds linear interpolated points between existing keyframes.
        """
        if len(self.camera_trajectory) < 2:
            return

        times = sorted(self.camera_trajectory.keys())
        interpolated = {}
        added_count = 0

        for i in range(len(times) - 1):
            ts1, ts2 = times[i], times[i + 1]

            # Add current point
            interpolated[ts1] = self.camera_trajectory[ts1]

            # Interpolate between ts1 and ts2
            gap = ts2 - ts1
            num_frames = max(1, int(gap * fps))

            p1 = self.camera_trajectory[ts1]
            p2 = self.camera_trajectory[ts2]

            for j in range(1, num_frames + 1):
                t_interp = ts1 + (j / (num_frames + 1)) * gap
                alpha = j / (num_frames + 1)

                # Linear interpolation
                x = (1 - alpha) * p1['x'] + alpha * p2['x']
                y = (1 - alpha) * p1['y'] + alpha * p2['y']

                interpolated[float(t_interp)] = {
                    'x': x,
                    'y': y,
                    'timestamp': float(t_interp),
                    'source_type': 'interpolated',
                    'confidence': 0.5
                }
                added_count += 1

        # Add last point
        interpolated[times[-1]] = self.camera_trajectory[times[-1]]

        self.camera_trajectory = interpolated

        logger.info(f"📍 CAMERA_TRAJ: Interpolated {added_count} points across gaps")

    def get_point_for_timestamp(self, timestamp, max_delta=0.1):
        """
        Get camera position for given timestamp.

        Args:
            timestamp: Target timestamp
            max_delta: Maximum time difference for nearest match

        Returns:
            Point dict or None
        """
        if not self.camera_trajectory:
            return None

        # Exact match
        if timestamp in self.camera_trajectory:
            return self.camera_trajectory[timestamp].copy()

        # Find nearest point
        times = sorted(self.camera_trajectory.keys())
        closest_t = min(times, key=lambda t: abs(t - timestamp))

        if abs(closest_t - timestamp) <= max_delta:
            return self.camera_trajectory[closest_t].copy()

        return None

    def get_trajectory_segment(self, start_ts, end_ts):
        """
        Get all trajectory points in time range.

        Args:
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            List of points sorted by timestamp
        """
        segment = []
        for ts in sorted(self.camera_trajectory.keys()):
            if start_ts <= ts <= end_ts:
                segment.append(self.camera_trajectory[ts].copy())

        return segment

    def get_stats(self):
        """Get statistics about the trajectory."""
        return {
            'total_points': len(self.camera_trajectory),
            'time_span': self._get_time_span(),
            'sources': self._count_sources()
        }

    def _get_time_span(self):
        """Get min-max time span."""
        if not self.camera_trajectory:
            return None

        times = self.camera_trajectory.keys()
        return (min(times), max(times))

    def _count_sources(self):
        """Count points by source type."""
        sources = {}
        for point in self.camera_trajectory.values():
            source = point.get('source_type', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        return sources

    def clear(self):
        """Clear all trajectory data."""
        self.camera_trajectory.clear()
