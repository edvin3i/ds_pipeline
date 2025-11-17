#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Display Probe Handler for Panorama Mode

Handles nvdsosd rendering for the panorama display mode including:
- Priority-based object rendering (ball → players)
- 16-object limit handling for nvdsosd on Jetson
- Color coding (RED for ball, GREEN for players)
- Border widths (3 for ball, 2 for players)
- Center of mass visualization
- Text overlays with frame numbers and timestamps
- All complex pyds metadata handling
"""

import time
import logging
from gi.repository import Gst
import pyds

logger = logging.getLogger("panorama-virtualcam")


class DisplayProbeHandler:
    """
    Handles display probe updates for panorama mode rendering.

    This class manages:
    - Ball and player detection rendering via nvdsosd
    - Priority-based rendering with 16-object limit
    - Center of mass computation for player tracking
    - Smooth interpolation for visual quality
    - Text overlays with statistics
    """

    def __init__(self, ball_history, players_history, all_detections_history,
                 display_mode='panorama'):
        """
        Initialize the DisplayProbeHandler.

        Args:
            ball_history: BallDetectionHistory instance for ball tracking
            players_history: PlayersHistory instance for player tracking
            all_detections_history: Dict of all detections by timestamp for multi-class tracking
            display_mode: Display mode ('panorama', 'virtualcam', etc.)
        """
        # Dependencies
        self.history = ball_history
        self.players_history = players_history
        self.all_detections_history = all_detections_history
        self.display_mode = display_mode

        # State variables for tracking and rendering
        self.display_frame_count = 0
        self.playback_log_count = 0

        # Statistics tracking (these will be updated externally)
        self.start_time = None
        self.frames_sent = 0
        self.current_fps = 0.0
        self.display_buffer_duration = 0.0

        # Center of mass computation state
        self.players_center_mass_smoothed = None  # (x, y) - smoothed position
        self._sorted_history_keys_cache = []
        self._sorted_history_keys_cache_time = 0

    def set_stats(self, start_time, frames_sent, current_fps, display_buffer_duration):
        """Update statistics from parent pipeline."""
        self.start_time = start_time
        self.frames_sent = frames_sent
        self.current_fps = current_fps
        self.display_buffer_duration = display_buffer_duration

    def get_all_detections_for_timestamp(self, ts, max_delta=0.12):
        """
        Get all detections (all classes) for timestamp.

        Args:
            ts: Target timestamp
            max_delta: Maximum time delta to consider a match

        Returns:
            Dict with detections by class or None if no match found
        """
        if not self.all_detections_history:
            return None

        # Find nearest timestamp
        timestamps = list(self.all_detections_history.keys())
        if not timestamps:
            return None

        closest_ts = min(timestamps, key=lambda t: abs(t - ts))
        if abs(closest_ts - ts) > max_delta:
            return None

        return self.all_detections_history[closest_ts]

    def _compute_smoothed_center_of_mass(self, current_ts):
        """
        Compute smoothed center of mass using all available history (7 seconds).

        Args:
            current_ts: Current timestamp for history lookup
        """
        if not self.all_detections_history:
            return

        # OPTIMIZATION: Cache computations (no frame skipping as needed for rendering)
        # Compute every frame but cache sorting

        lookback = 7.0  # Use entire buffer
        start_ts = current_ts - lookback

        # OPTIMIZATION: Cache sorted keys
        if not hasattr(self, '_sorted_history_keys_cache'):
            self._sorted_history_keys_cache = []
            self._sorted_history_keys_cache_time = 0

        # Update cache only if history changed
        if len(self.all_detections_history) != len(self._sorted_history_keys_cache):
            self._sorted_history_keys_cache = sorted(self.all_detections_history.keys())

        # Collect all centers of mass for last 7 seconds
        centers_history = []
        for ts in self._sorted_history_keys_cache:
            if start_ts <= ts <= current_ts:
                detections = self.all_detections_history[ts]
                players = detections.get('player', [])
                if players and len(players) > 0:
                    center_x = sum(p['x'] for p in players) / len(players)
                    center_y = sum(p['y'] for p in players) / len(players)
                    centers_history.append((ts, center_x, center_y))

        if len(centers_history) < 3:
            # Not enough data
            if self.display_frame_count % 30 == 0:
                logger.warning(f"⚠️ COM: Not enough data - only {len(centers_history)} points in history")
            return

        # Apply median filter to remove outliers
        # Take last 30 points (approximately 1 second at 30 FPS)
        recent_centers = centers_history[-30:] if len(centers_history) >= 30 else centers_history

        # Compute median X and Y
        x_values = [c[1] for c in recent_centers]
        y_values = [c[2] for c in recent_centers]

        # Sort and get median
        x_values_sorted = sorted(x_values)
        y_values_sorted = sorted(y_values)

        n = len(x_values_sorted)
        if n % 2 == 0:
            median_x = (x_values_sorted[n//2-1] + x_values_sorted[n//2]) / 2
            median_y = (y_values_sorted[n//2-1] + y_values_sorted[n//2]) / 2
        else:
            median_x = x_values_sorted[n//2]
            median_y = y_values_sorted[n//2]

        # Filter outliers: keep only points close to median
        filtered_centers = []
        for ts, x, y in recent_centers:
            dist_to_median = ((x - median_x)**2 + (y - median_y)**2)**0.5
            # Cut off points farther than 200px from median
            if dist_to_median < 200:
                filtered_centers.append((ts, x, y))

        if not filtered_centers:
            filtered_centers = recent_centers  # Fallback

        # Weighted average: fresher points are more important
        # Use exponential weights: more recent points get higher weight
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for i, (ts, x, y) in enumerate(filtered_centers):
            # Weight grows exponentially for fresher points
            # Last point gets maximum weight
            weight = (i + 1) ** 1.5  # Power 1.5 gives good balance
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        if total_weight > 0:
            final_x = weighted_x / total_weight
            final_y = weighted_y / total_weight

            # Update smoothed position directly (no EMA, already smoothed)
            self.players_center_mass_smoothed = (final_x, final_y)

            # DEBUG: Log update (every 30 frames)
            if self.display_frame_count % 30 == 0:
                logger.info(f"🎯 Center of Mass updated: ({final_x:.0f}, {final_y:.0f}), "
                           f"from {len(filtered_centers)} points")

    def handle_playback_draw_probe(self, pad, info, u_data):
        """
        Render detections in playback pipeline (nvdsosd sink pad probe).

        This is the main rendering callback that:
        1. Gets current frame metadata
        2. Retrieves all detections for current timestamp
        3. Computes smoothed center of mass for players
        4. Renders bounding boxes with priority-based selection
        5. Adds text overlays with statistics

        Args:
            pad: GStreamer pad (nvdsosd sink pad)
            info: GstPadProbeInfo containing buffer
            u_data: User data (unused)

        Returns:
            Gst.PadProbeReturn.OK - Always continue processing.
            We never drop buffers as all frames need rendering for display.

        Reference:
            DeepStream SDK 7.1 - /ds_doc/7.1/text/DS_Zero_Coding_DS_Components.html
        """
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            pts_sec = float(gst_buffer.pts) / float(Gst.SECOND)

            # IMPORTANT: Update display timestamp for correct history cleanup
            self.history.update_display_timestamp(pts_sec)

            # Get ball detection for current moment (for text output)
            det = self.history.get_detection_for_timestamp(pts_sec, max_delta=0.12)

            # Get ALL detections of all classes for rendering
            all_detections = self.get_all_detections_for_timestamp(pts_sec, max_delta=0.12)

            # ========== PANORAMA MODE: add interpolated ball points ==========
            # In panorama mode show ALL points including interpolation
            # In camera mode - only real detections (resource saving)
            if self.display_mode == 'panorama' and det:
                # If found ball detection from history, add it to all_detections
                if all_detections is None:
                    all_detections = {'ball': [], 'player': [], 'staff': [], 'referee': []}

                # Form detection in format for rendering
                ball_det = {
                    'x': det[6] if len(det) > 6 else det[0],  # global X
                    'y': det[7] if len(det) > 7 else det[1],  # global Y
                    'width': det[8] if len(det) > 8 else det[2],
                    'height': det[9] if len(det) > 9 else det[3],
                    'confidence': det[4] if len(det) > 4 else 0.5,
                    'is_interpolated': True  # Marker for visual distinction
                }

                # Check if we don't already have same detection (to avoid duplicates)
                existing_balls = all_detections.get('ball', [])
                is_duplicate = False
                for existing in existing_balls:
                    if abs(existing['x'] - ball_det['x']) < 5 and abs(existing['y'] - ball_det['y']) < 5:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    all_detections['ball'].append(ball_det)

            # Compute smoothed center of mass using entire history (7 seconds)
            self._compute_smoothed_center_of_mass(pts_sec)

            # DEBUG: show status (first 5 times) - FIXED: added counter increment
            if not hasattr(self, 'playback_log_count'):
                self.playback_log_count = 0
            if self.playback_log_count < 5:
                history_size = len(self.history.raw_future_history) + len(self.history.processed_future_history) + len(self.history.confirmed_history)
                det_summary = ""
                if all_detections:
                    det_summary = f"ball={len(all_detections.get('ball', []))}, player={len(all_detections.get('player', []))}, staff={len(all_detections.get('staff', []))}, ref={len(all_detections.get('referee', []))}"
                logger.info(f"🎨 playback_draw_probe: frame={self.playback_log_count}, pts={pts_sec:.2f}, "
                           f"display_ts={self.history.current_display_timestamp:.2f}, "
                           f"history_size={history_size}, det={'found' if det else 'None'}, all_det=[{det_summary}]")
                self.playback_log_count += 1

            # Draw on nvdsosd
            # CRITICAL FIX: Proper metadata iteration with StopIteration handling
            # Reference: DeepStream SDK 7.1 documentation
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                if not fm:
                    try:
                        l_frame = l_frame.next
                    except StopIteration:
                        break
                    continue

                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                if not display_meta:
                    try:
                        l_frame = l_frame.next
                    except StopIteration:
                        break
                    continue

                # Prepare text
                elapsed = max(1e-6, time.time() - self.start_time) if self.start_time else 0
                self.current_fps = float(self.frames_sent) / elapsed if elapsed > 0 else 0
                text = f"FPS:{self.current_fps:.1f} | Buf:{self.display_buffer_duration:.2f}s"

                # Tile rendering DISABLED (free up space for detections)
                # TILE_POSITIONS contains (x, y, width, height) for each tile
                num_tiles = 0  # Disabled tiles

                # Count how many boxes we need to draw
                # ONLY BALL AND PLAYERS (staff and referees disabled)
                num_detection_rects = 0
                if all_detections:
                    num_detection_rects = (len(all_detections.get('ball', [])) +
                                          len(all_detections.get('player', [])))

                # Limit total number of rectangles
                # IMPORTANT: On this platform limit rect_params = 16
                # 0 tiles + maximum 16 detections
                max_available_rects = 16
                total_rects_needed = num_detection_rects

                if total_rects_needed > max_available_rects:
                    num_detection_rects = max_available_rects

                display_meta.num_rects = num_detection_rects

                # Tiles disabled - go straight to detections

                # Add only main text (no tile labels)
                display_meta.num_labels = 1  # only main text

                # Draw detections ONLY for ball and players
                rect_idx = 0  # Start from 0, as no tiles
                total_drawn = 0

                if all_detections:
                    # Color scheme: only ball and players
                    # ball: red (1.0, 0.0, 0.0, 1.0)
                    # player: green (0.0, 1.0, 0.0, 1.0)

                    class_colors = {
                        'ball': (1.0, 0.0, 0.0, 1.0),      # Red
                        'player': (0.0, 1.0, 0.0, 1.0)      # Green
                    }

                    class_widths = {
                        'ball': 3,      # Thicker for ball
                        'player': 2     # Thicker for visibility
                    }

                    # Limit number of boxes to draw
                    max_boxes = num_detection_rects

                    for class_name, color in class_colors.items():
                        detections_list = all_detections.get(class_name, [])

                        # For players draw center of mass instead of individual boxes
                        if class_name == 'player' and self.players_center_mass_smoothed:
                            # Use precomputed smoothed position from _compute_smoothed_center_of_mass()
                            # This function uses entire 7-second history for maximum smooth movement
                            center_x, center_y = self.players_center_mass_smoothed

                            # Center of mass box size (can be configured)
                            cm_box_size = 100  # 100x100 pixels

                            # Check limit
                            if rect_idx < max_boxes and rect_idx < display_meta.num_rects:
                                try:
                                    left = int(center_x - cm_box_size / 2)
                                    top = int(center_y - cm_box_size / 2)

                                    rect = display_meta.rect_params[rect_idx]
                                    rect.left = max(0, left)
                                    rect.top = max(0, top)
                                    rect.width = cm_box_size
                                    rect.height = cm_box_size
                                    rect.border_width = 4  # Thicker for center of mass
                                    rect.border_color.set(*color)  # Green color
                                    rect.has_bg_color = 0

                                    rect_idx += 1
                                    total_drawn += 1
                                except (IndexError, Exception) as e:
                                    logger.warning(f"Reached rect_params limit at index {rect_idx}: {e}")
                                    break

                        # For ball draw as usual
                        elif class_name == 'ball':
                            for d in detections_list:
                                # Check limit
                                if rect_idx >= max_boxes or rect_idx >= display_meta.num_rects:
                                    break

                                try:
                                    cx = d['x']
                                    cy = d['y']
                                    w = d['width']
                                    h = d['height']
                                    is_interp = d.get('is_interpolated', False)

                                    left = int(cx - w / 2)
                                    top = int(cy - h / 2)

                                    rect = display_meta.rect_params[rect_idx]
                                    rect.left = max(0, left)
                                    rect.top = max(0, top)
                                    rect.width = int(max(2, w))
                                    rect.height = int(max(2, h))

                                    # Interpolated points - yellow with same thickness
                                    if is_interp:
                                        rect.border_width = class_widths[class_name]  # Same thickness
                                        rect.border_color.set(1.0, 1.0, 0.0, 1.0)  # Bright yellow
                                    else:
                                        rect.border_width = class_widths[class_name]
                                        rect.border_color.set(*color)  # Red for real

                                    rect.has_bg_color = 0

                                    rect_idx += 1
                                    total_drawn += 1
                                except (IndexError, Exception) as e:
                                    # Reached rect_params limit, stop
                                    logger.warning(f"Reached rect_params limit at index {rect_idx}: {e}")
                                    break

                        # If limit reached, break outer loop
                        if rect_idx >= max_boxes or rect_idx >= display_meta.num_rects:
                            break

                # Update text
                if det is not None:
                    cx, cy, w, h, conf = det[0:5]
                    text += f" | Ball:({int(cx)},{int(cy)}) conf={conf:.2f}"

                if all_detections:
                    # Show only ball and players
                    text += f" | Drawn: Ball={len(all_detections.get('ball', []))}, Players={len(all_detections.get('player', []))}"

                # Add main text (FPS, buffer, ball)
                lbl = display_meta.text_params[0]  # Index 0, as no tiles
                lbl.display_text = text
                lbl.x_offset = 10
                lbl.y_offset = 10
                lbl.font_params.font_name = "Serif"
                lbl.font_params.font_size = 20
                lbl.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                lbl.set_bg_clr = 1
                lbl.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

                pyds.nvds_add_display_meta_to_frame(fm, display_meta)
                break  # Only process first frame in batch

        except Exception as e:
            logger.error(f"playback_draw_probe error: {e}")
            import traceback
            traceback.print_exc()

        return Gst.PadProbeReturn.OK
