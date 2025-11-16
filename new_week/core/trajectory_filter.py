#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrajectoryFilter - Outlier detection and permanent blacklist management.

Detects false trajectories and maintains a permanent blacklist of
problematic coordinate clusters that repeatedly cause outliers.
"""

import math
import logging
from collections import defaultdict
from typing import Dict, Set, Tuple, Any

logger = logging.getLogger("panorama-virtualcam")


class TrajectoryFilter:
    """
    Outlier detection and blacklist management for trajectory filtering.

    Features:
    - Detects outliers using distance-based heuristics
    - Maintains permanent blacklist of coordinate clusters
    - Tracks outlier removal frequency per cluster
    - Automatic banning of persistent outliers
    """

    def __init__(self, outlier_ban_threshold=6, ban_radius=30):
        """
        Initialize the trajectory filter.

        Args:
            outlier_ban_threshold: Number of removals before permanent ban (default: 6)
            ban_radius: Radius in pixels for ban zone (default: 30)
        """
        # Outlier tracking
        self.outlier_removal_count = defaultdict(int)  # Removal counter per cluster
        self.permanent_blacklist = set()  # Permanent ban set (automatic)

        # Configuration
        self.outlier_ban_threshold = outlier_ban_threshold
        self.ban_radius = ban_radius

    def detect_and_remove_false_trajectories(self, history):
        """
        Detect and remove outliers with permanent ban for persistent ones.

        Uses multiple heuristics:
        1. Frequency-based detection (positions appearing 3+ times)
        2. Distance-based detection (large jumps from neighbors)
        3. Window-based averaging (5-point window analysis)

        Args:
            history: Dictionary of timestamp -> detection mappings

        Returns:
            Cleaned history dictionary
        """
        if len(history) < 5:
            return history

        clean_history = dict(history)
        coords = []

        # Collect all points for analysis
        for ts in sorted(history.keys()):
            det = history[ts]
            if det and len(det) >= 8:
                # Check permanent blacklist first
                coord_key = (int(det[6]), int(det[7]))
                if coord_key in self.permanent_blacklist:
                    del clean_history[ts]
                    continue

                coords.append({
                    'ts': ts,
                    'x': det[6],
                    'y': det[7],
                    'det': det
                })

        if len(coords) < 5:
            return clean_history

        # NEW: Count position frequency
        position_frequency = defaultdict(list)
        for i, point in enumerate(coords):
            key = (round(point['x']/30)*30, round(point['y']/30)*30)
            position_frequency[key].append(i)

        # NEW: Find suspiciously frequent positions
        suspicious_positions = set()
        for pos_key, indices in position_frequency.items():
            if len(indices) >= 3:  # Appears 3+ times
                suspicious_positions.add(pos_key)
                logger.debug(f"Suspicious frequent position {pos_key}: {len(indices)} times")

        outliers_to_remove = []

        # Check each point with extended context
        for i in range(len(coords)):
            curr = coords[i]
            curr_key = (round(curr['x']/30)*30, round(curr['y']/30)*30)

            # NEW: Check if point is from frequent positions
            if curr_key in suspicious_positions:
                # Verify if it's actually an outlier
                is_outlier = False

                # Check 1: Large distance to neighbors
                if i > 0 and i < len(coords) - 1:
                    prev = coords[i-1]
                    next = coords[i+1]
                    dist_to_prev = math.sqrt((curr['x'] - prev['x'])**2 +
                                            (curr['y'] - prev['y'])**2)
                    dist_to_next = math.sqrt((curr['x'] - next['x'])**2 +
                                            (curr['y'] - next['y'])**2)

                    if dist_to_prev > 500 and dist_to_next > 500:
                        is_outlier = True

                # Check 2: 5-point window analysis (if possible)
                if not is_outlier and i >= 2 and i < len(coords) - 2:
                    # Take window of 5 points
                    window = coords[i-2:i+3]

                    # Calculate average distance to other points in window
                    total_dist = 0
                    count = 0
                    for j, other in enumerate(window):
                        if j != 2:  # Not the point itself (i in window is index 2)
                            dist = math.sqrt((curr['x'] - other['x'])**2 +
                                        (curr['y'] - other['y'])**2)
                            total_dist += dist
                            count += 1

                    avg_dist = total_dist / count if count > 0 else 0

                    # If far from all on average - outlier
                    if avg_dist > 600:
                        is_outlier = True

                if is_outlier:
                    outliers_to_remove.append(curr)
                    # Increase counter for ban (+2 for frequent)
                    self.outlier_removal_count[curr_key] += 2

            # Old check for normal outliers (not frequent)
            elif i > 0 and i < len(coords) - 1:
                prev = coords[i-1]
                next = coords[i+1]

                # Distances between points
                dist_to_prev = math.sqrt((curr['x'] - prev['x'])**2 +
                                        (curr['y'] - prev['y'])**2)
                dist_to_next = math.sqrt((curr['x'] - next['x'])**2 +
                                        (curr['y'] - next['y'])**2)
                dist_prev_next = math.sqrt((next['x'] - prev['x'])**2 +
                                          (next['y'] - prev['y'])**2)

                # Old outlier check
                if dist_to_prev + dist_to_next > dist_prev_next * 2.5:
                    outliers_to_remove.append(curr)
                elif dist_to_prev > 1000 or dist_to_next > 1000:
                    if dist_prev_next < max(dist_to_prev, dist_to_next) * 0.7:
                        outliers_to_remove.append(curr)

        # Process found outliers
        banned_count = 0
        for outlier in outliers_to_remove:
            # Round to 30px cluster for grouping nearby outliers
            cluster_key = (round(outlier['x'] / 30) * 30, round(outlier['y'] / 30) * 30)

            # Increase removal counter for this cluster
            self.outlier_removal_count[cluster_key] += 1

            # Check threshold for ban
            if self.outlier_removal_count[cluster_key] >= self.outlier_ban_threshold:
                # Check if zone is already banned
                already_banned = False
                for (bx, by) in self.permanent_blacklist:
                    if abs(cluster_key[0] - bx) < self.ban_radius and abs(cluster_key[1] - by) < self.ban_radius:
                        already_banned = True
                        break

                if not already_banned:
                    self.permanent_blacklist.add(cluster_key)
                    banned_count += 1
                    logger.warning(f"⛔ PERMANENT BAN: cluster {cluster_key} "
                                f"(removed {self.outlier_removal_count[cluster_key]} times)")

            # Remove from history
            if outlier['ts'] in clean_history:
                del clean_history[outlier['ts']]
                logger.debug(f"Removed outlier at ({outlier['x']:.0f},{outlier['y']:.0f}), "
                        f"cluster={cluster_key}, count={self.outlier_removal_count[cluster_key]}")

        # Periodic cleanup of old entries in counter
        if len(self.outlier_removal_count) > 50:
            filtered = {
                k: v for k, v in self.outlier_removal_count.items()
                if v >= self.outlier_ban_threshold - 1
            }
            self.outlier_removal_count = defaultdict(int, filtered)

        if banned_count > 0:
            logger.info(f"🚫 Banned {banned_count} persistent outliers. "
                    f"Total banned: {len(self.permanent_blacklist)}")

        return clean_history

    def is_point_banned(self, x, y):
        """
        Check if a point is in a banned zone.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is banned, False otherwise
        """
        for (bx, by) in self.permanent_blacklist:
            distance = math.sqrt((x - bx)**2 + (y - by)**2)
            if distance < self.ban_radius:
                logger.warning(f"⛔ BANNED: ({x:.1f}, {y:.1f}) near banned zone ({bx}, {by}), dist={distance:.0f}px")
                return True
        return False

    def quick_outlier_check(self, point, window, point_idx):
        """
        Fast outlier check for a point in a window.

        Args:
            point: Point to check (dict with 'x', 'y')
            window: List of surrounding points
            point_idx: Index of point in window

        Returns:
            True if outlier, False otherwise
        """
        if point_idx < 1 or point_idx >= len(window) - 1:
            return False

        prev_point = window[point_idx - 1]
        next_point = window[point_idx + 1]

        dist_to_prev = math.sqrt((point['x'] - prev_point['x'])**2 +
                                 (point['y'] - prev_point['y'])**2)
        dist_to_next = math.sqrt((point['x'] - next_point['x'])**2 +
                                 (point['y'] - next_point['y'])**2)
        dist_prev_next = math.sqrt((next_point['x'] - prev_point['x'])**2 +
                                   (next_point['y'] - prev_point['y'])**2)

        detour_factor = (dist_to_prev + dist_to_next) / (dist_prev_next + 0.1)

        if detour_factor > 1.5:
            if point_idx < len(window) - 2:
                after_next = window[point_idx + 2]
                dist_prev_after = math.sqrt((after_next['x'] - prev_point['x'])**2 +
                                           (after_next['y'] - prev_point['y'])**2)

                if dist_prev_after < dist_to_prev + dist_to_next:
                    return True

        return False

    def validate_outlier_series(self, outliers, coords):
        """
        Validate a series of outliers to prevent removing valid trajectories.

        If 3+ consecutive points are marked as outliers, they're likely
        a valid trajectory, not outliers.

        Args:
            outliers: Set of timestamps marked as outliers
            coords: List of all coordinates

        Returns:
            Validated set of outliers (empty if consecutive series found)
        """
        if len(outliers) < 3:
            return outliers

        outlier_ts = sorted(outliers)
        consecutive = 1
        for i in range(1, len(outlier_ts)):
            prev_idx = next((j for j, c in enumerate(coords) if c['ts'] == outlier_ts[i-1]), -1)
            curr_idx = next((j for j, c in enumerate(coords) if c['ts'] == outlier_ts[i]), -1)

            if curr_idx - prev_idx == 1:
                consecutive += 1
            else:
                consecutive = 1

            if consecutive >= 3:
                return set()

        return outliers

    def get_blacklist_info(self):
        """
        Get information about the current blacklist.

        Returns:
            Dictionary with blacklist statistics
        """
        return {
            'total_banned': len(self.permanent_blacklist),
            'blacklist': list(self.permanent_blacklist),
            'outlier_counts': dict(self.outlier_removal_count)
        }

    def clear_blacklist(self):
        """Clear the permanent blacklist and outlier counts."""
        self.permanent_blacklist.clear()
        self.outlier_removal_count.clear()
