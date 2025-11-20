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

    def populate_camera_trajectory_from_ball_history(self, ball_history_dict, players_history, fps=30):
        """
        Одна монолитная функция для построения полной траектории камеры.

        Делает всё в одном проходе:
        1. Заполняет из истории мяча
        2. Обнаруживает разрывы > max_gap
        3. Заполняет разрывы player COM
        4. Сглаживает outliers (> outlier_threshold px)
        5. Интерполирует для smooth 30fps движения

        Args:
            ball_history_dict: Очищенная история мяча {timestamp → detection}
            players_history: PlayersHistory для fallback player COM
            fps: Частота кадров для финальной интерполяции (по умолчанию 30)

        Returns:
            None (обновляет self.camera_trajectory)
        """
        if not ball_history_dict:
            logger.warning("🚨 CAMERA_TRAJ: Empty ball history")
            return

        self.camera_trajectory.clear()

        # ===== ЭТАП 1: Заполнение из мяча + обнаружение разрывов =====
        sorted_timestamps = sorted(ball_history_dict.keys())
        if not sorted_timestamps:
            return

        for i, ts in enumerate(sorted_timestamps):
            detection = ball_history_dict[ts]
            if not detection or len(detection) < 8:
                continue

            is_interpolated = detection[10] if len(detection) > 10 else False
            source_type = 'interpolated_ball' if is_interpolated else 'ball'

            # Добавляем точку мяча
            self.camera_trajectory[float(ts)] = {
                'x': float(detection[6]),
                'y': float(detection[7]),
                'timestamp': float(ts),
                'source_type': source_type,
                'confidence': float(detection[4]) if len(detection) > 4 else 0.5
            }

            # Проверяем разрыв до следующей точки
            if i + 1 < len(sorted_timestamps):
                ts_next = sorted_timestamps[i + 1]
                gap = ts_next - ts

                # Если разрыв > max_gap → заполняем player COM
                if gap > self.max_gap:
                    logger.info(f"🔄 CAMERA_TRAJ: Gap {gap:.2f}s > {self.max_gap}s at ts={ts:.2f}→{ts_next:.2f}, "
                               f"filling with player positions")

                    next_detection = ball_history_dict[ts_next]
                    next_x = float(next_detection[6])
                    next_y = float(next_detection[7])

                    # ===== Заполняем разрыв player COM с шагом для 30fps =====
                    # Для 4с разрыва: 4 * 30 = 120 интерполяций, но показываем каждый 15-й кадр = 8 точек
                    frame_step = 15  # Показываем каждый 15-й кадр (0.5с при 30fps)
                    num_frames = int(gap * 30)  # Всего кадров в разрыве
                    points_added = 0

                    for frame_idx in range(frame_step, num_frames, frame_step):
                        current_ts = ts + (frame_idx / 30.0)

                        # Не добавляем точку слишком близко к концу (оставляем место для переходной точки)
                        if current_ts >= ts_next - 0.2:
                            break

                        player_com = players_history.get_player_com_for_timestamp(current_ts)

                        if player_com:
                            self.camera_trajectory[float(current_ts)] = {
                                'x': float(player_com[0]),
                                'y': float(player_com[1]),
                                'timestamp': float(current_ts),
                                'source_type': 'player',
                                'confidence': 0.35
                            }
                            points_added += 1
                            logger.info(f"  ➕ Player COM[{points_added}] at ts={current_ts:.2f}: ({player_com[0]:.0f}, {player_com[1]:.0f})")

                    # ===== Добавляем плавный переход (blend) перед восстановлением мяча =====
                    transition_ts = ts + gap * 0.85  # 85% пути в разрыв
                    player_com = players_history.get_player_com_for_timestamp(transition_ts)

                    if player_com:
                        alpha = 0.5  # 50% игрок, 50% мяч
                        blend_x = (1 - alpha) * player_com[0] + alpha * next_x
                        blend_y = (1 - alpha) * player_com[1] + alpha * next_y

                        self.camera_trajectory[float(transition_ts)] = {
                            'x': blend_x,
                            'y': blend_y,
                            'timestamp': float(transition_ts),
                            'source_type': 'blend',
                            'confidence': 0.4
                        }
                        logger.info(f"  ➕ Blend[transition] at ts={transition_ts:.2f}: ({blend_x:.0f}, {blend_y:.0f})")

                    logger.info(f"  📊 Filled gap with {points_added} player COM points + 1 blend point")

        logger.info(f"📍 CAMERA_TRAJ: Loaded {len(self.camera_trajectory)} points (ball + player fills)")

        # ===== ЭТАП 2: Фильтрование временных движений (разворотов) =====
        self._filter_temporary_movements()

        # ===== ЭТАП 3: Финальная интерполяция для 30fps =====
        self._interpolate_gaps_internal(fps)

    def _filter_temporary_movements(self):
        """
        Фильтрует временные движения (развороты) в траектории.

        Удаляет MIDDLE points в последовательности, где движения REVERSAL друг друга.
        Например: мяч движется 300px вправо, потом 300px влево - удаляем middle point.

        Алгоритм:
        1. Для каждой тройки consecutive ball-points (prev, curr, next)
        2. Вычисляем movement vectors: prev→curr и curr→next
        3. Если vectors OPPOSITE (angle > 120°) - это reversal
        4. Удаляем MIDDLE point (curr)
        """
        if len(self.camera_trajectory) < 3:
            return

        times = sorted(self.camera_trajectory.keys())

        points_to_remove = set()

        # Проходим по последовательностям ONLY из ball-source точек
        # (interpolated точки добавятся позже, не нужно их анализировать сейчас)
        ball_times = [t for t in times
                      if self.camera_trajectory[t].get('source_type') == 'ball']

        if len(ball_times) < 3:
            return  # Need at least 3 points to detect reversals

        for i in range(1, len(ball_times) - 1):
            prev_point = self.camera_trajectory[ball_times[i - 1]]
            curr_point = self.camera_trajectory[ball_times[i]]
            next_point = self.camera_trajectory[ball_times[i + 1]]

            # Вычисляем vectors движений
            vec1_x = curr_point['x'] - prev_point['x']
            vec1_y = curr_point['y'] - prev_point['y']

            vec2_x = next_point['x'] - curr_point['x']
            vec2_y = next_point['y'] - curr_point['y']

            # Вычисляем dot product и lengths
            dot_product = vec1_x * vec2_x + vec1_y * vec2_y
            len1 = math.sqrt(vec1_x ** 2 + vec1_y ** 2)
            len2 = math.sqrt(vec2_x ** 2 + vec2_y ** 2)

            # Если один из vectors очень мал, пропускаем
            if len1 < 10 or len2 < 10:
                continue

            # Вычисляем косинус угла между vectors
            cos_angle = dot_product / (len1 * len2) if len1 > 0 and len2 > 0 else 1.0

            # Если cos_angle < -0.5 (angle > 120°) - это reversal
            # cos(120°) = -0.5, так что отрицательное значение означает vectors противоположные
            if cos_angle < -0.5:
                logger.info(f"🔄 FILTERED reversal at {ball_times[i]:.2f}: "
                           f"angle={math.degrees(math.acos(max(-1, min(1, cos_angle)))):.0f}°, "
                           f"prev→curr: ({vec1_x:.0f}, {vec1_y:.0f}), "
                           f"curr→next: ({vec2_x:.0f}, {vec2_y:.0f})")

                # Удаляем MIDDLE point
                points_to_remove.add(ball_times[i])

        # Удаляем помеченные точки
        for ts in points_to_remove:
            del self.camera_trajectory[ts]

        if points_to_remove:
            logger.info(f"🔧 CAMERA_TRAJ: Filtered {len(points_to_remove)} reversal points from trajectory")

    def _interpolate_gaps_internal(self, fps=30):
        """
        Внутренняя функция: интерполирует разрывы для smooth 30fps движения.

        Добавляет синтетические точки между ключевыми кадрами.

        Args:
            fps: Частота кадров для интерполяции
        """
        if len(self.camera_trajectory) < 2:
            return

        times = sorted(self.camera_trajectory.keys())
        interpolated = {}
        added_count = 0

        for i in range(len(times) - 1):
            ts1, ts2 = times[i], times[i + 1]

            # Добавляем текущую точку
            interpolated[ts1] = self.camera_trajectory[ts1]

            # Интерполируем между ts1 и ts2
            gap = ts2 - ts1
            num_frames = max(1, int(gap * fps))

            p1 = self.camera_trajectory[ts1]
            p2 = self.camera_trajectory[ts2]

            for j in range(1, num_frames + 1):
                t_interp = ts1 + (j / (num_frames + 1)) * gap
                alpha = j / (num_frames + 1)

                # Линейная интерполяция
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

        # Добавляем последнюю точку
        interpolated[times[-1]] = self.camera_trajectory[times[-1]]

        self.camera_trajectory = interpolated

        logger.info(f"📍 CAMERA_TRAJ: Interpolated {added_count} points across gaps")

    def populate_from_ball_and_players(self, ball_history_dict, players_history):
        """
        [DEPRECATED] Используй populate_camera_trajectory_from_ball_history() вместо этого.

        Это старая функция, оставлена для обратной совместимости.
        """
        self.populate_camera_trajectory_from_ball_history(ball_history_dict, players_history)

    def smooth_trajectory(self, window_size=5, threshold_px=None):
        """
        [DEPRECATED] Используй populate_camera_trajectory_from_ball_history() вместо этого.

        Эта функция теперь вызывается внутри populate_camera_trajectory_from_ball_history().
        """
        logger.warning("⚠️ smooth_trajectory() is deprecated. Use populate_camera_trajectory_from_ball_history()")

    def interpolate_gaps(self, fps=30):
        """
        [DEPRECATED] Используй populate_camera_trajectory_from_ball_history() вместо этого.

        Эта функция теперь вызывается внутри populate_camera_trajectory_from_ball_history().
        """
        logger.warning("⚠️ interpolate_gaps() is deprecated. Use populate_camera_trajectory_from_ball_history()")

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
