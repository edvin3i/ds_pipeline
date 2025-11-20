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
        1. Заполняет из истории мяча (если есть)
        2. Если нет мяча → использует центр масс игроков
        3. Обнаруживает разрывы > max_gap
        4. Заполняет разрывы player COM
        5. Сглаживает outliers (> outlier_threshold px)
        6. Интерполирует для smooth 30fps движения

        Args:
            ball_history_dict: Очищенная история мяча {timestamp → detection}
            players_history: PlayersHistory для fallback player COM
            fps: Частота кадров для финальной интерполяции (по умолчанию 30)

        Returns:
            None (обновляет self.camera_trajectory)
        """
        self.camera_trajectory.clear()

        if not ball_history_dict:
            # Мяч потерян на 7+ сек (история очищена) или при старте
            logger.warning("🚨 CAMERA_TRAJ: Empty ball history - using PLAYER CENTER-OF-MASS fallback")

            # Заполняем траекторию исключительно player COM
            if not players_history:
                logger.warning("  ⚠️ No players_history available - cannot fill trajectory")
                return

            # Заполняем последние 3 сек центром масс игроков
            # (это будет видно в отображении 7 сек назад)
            import time
            current_time = time.time()
            lookback_seconds = 3.0

            # Используем последние известные timestamps из players_history
            player_times = sorted(players_history.detections.keys()) if hasattr(players_history, 'detections') else []

            if player_times:
                # Используем временной диапазон из доступных данных игроков
                start_ts = player_times[0]
                end_ts = player_times[-1]

                # Заполняем в диапазоне доступных данных
                frame_step = 15  # 0.5s интервал
                current_ts = start_ts
                points_added = 0

                while current_ts <= end_ts:
                    try:
                        player_com = players_history.get_player_com_for_timestamp(current_ts)
                        if player_com:
                            self.camera_trajectory[float(current_ts)] = {
                                'x': float(player_com[0]),
                                'y': float(player_com[1]),
                                'timestamp': float(current_ts),
                                'source_type': 'player_only',  # Нет мяча, только игроки
                                'confidence': 0.25  # Низкая уверенность
                            }
                            points_added += 1
                    except (ValueError, RuntimeError, IndexError) as e:
                        logger.debug(f"  ⚠️ Could not get player COM at ts={current_ts:.2f}: {e}")

                    current_ts += (frame_step / fps)

                if points_added > 0:
                    logger.info(f"  ✅ Filled trajectory with {points_added} player COM points (no ball detected)")

            return

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

                # Если разрыв >= max_gap → заполняем player COM
                if gap >= self.max_gap:
                    logger.warning(f"🔴 БОЛЬШОЙ РАЗРЫВ: {gap:.2f}s > {self.max_gap}s at ts={ts:.2f}→{ts_next:.2f}")
                    logger.info(f"🔄 CAMERA_TRAJ: Gap {gap:.2f}s > {self.max_gap}s at ts={ts:.2f}→{ts_next:.2f}, "
                               f"filling with player positions")
                    logger.info(f"  📌 players_history type: {type(players_history)}")
                    logger.info(f"  📌 players_history has detections: {hasattr(players_history, 'detections')}")

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

                        try:
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
                            else:
                                logger.debug(f"  ⚠️ No player COM available at ts={current_ts:.2f}")
                        except Exception as e:
                            logger.warning(f"  ❌ Error getting player COM at ts={current_ts:.2f}: {e}")

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

        # ===== ВЫВОД ПЕРЕД ИНТЕРПОЛЯЦИЕЙ =====
        self._dump_trajectory_before_interpolation()

        # ===== ЭТАП 2: Фильтрование временных движений (разворотов) =====
        # self._filter_temporary_movements()  # DISABLED FOR NOW

        # ===== ЭТАП 3: Финальная интерполяция для 30fps =====
        self._interpolate_gaps_internal(fps)

        # ===== ЭТАП 4: Масштабирование мяча по скорости =====
        self._apply_speed_scaling()

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

    def fill_gaps_in_trajectory(self, players_history, current_display_ts=None):
        """
        Заполняет пропуски В СУЩЕСТВУЮЩЕЙ траектории player COM.

        Эта функция вызывается ПОСЛЕ populate_camera_trajectory_from_ball_history(),
        чтобы заполнить пропуски МЕЖДУ последовательными вызовами populate().

        Обрабатывает три случая:
        1. Пропуски МЕЖДУ соседними точками траектории (gap > max_gap)
        2. ПУСТАЯ траектория (начало работы) → заполняет от 0 до текущего времени
        3. Долгая потеря мяча (история очищена) → заполняет от последней известной точки

        Args:
            players_history: PlayersHistory object для получения COM позиций
            current_display_ts: Текущее время отображения (для случая пустой истории)
        """
        if not players_history:
            return

        times = sorted(self.camera_trajectory.keys()) if self.camera_trajectory else []
        gaps_found = 0

        # ===== СЛУЧАЙ 1: ПУСТАЯ траектория (начало или полная очистка) =====
        # Траектория пуста = мяч потерян на 7+ секунд и история очищена
        # Заполняем исключительно центром масс игроков (последние 3 секунды)
        if not times and current_display_ts is not None:
            logger.info(f"🎯 EMPTY TRAJECTORY at ts={current_display_ts:.2f} - filling with PLAYER CENTER-OF-MASS")

            # Заполняем последние 3 секунды перед текущим временем
            lookback_seconds = 3.0
            start_ts = current_display_ts - lookback_seconds
            frame_step = 15  # Каждый 15-й кадр = 0.5s интервал
            num_frames = int(lookback_seconds * 30)  # 3s * 30fps = 90 кадров
            points_added = 0

            for frame_idx in range(0, num_frames, frame_step):
                fill_ts = start_ts + (frame_idx / 30.0)

                try:
                    player_com = players_history.get_player_com_for_timestamp(fill_ts)

                    if player_com:
                        self.camera_trajectory[float(fill_ts)] = {
                            'x': float(player_com[0]),
                            'y': float(player_com[1]),
                            'timestamp': float(fill_ts),
                            'source_type': 'player_init',  # Инициализация пустой траектории
                            'confidence': 0.30  # Низкая уверенность (нет мяча)
                        }
                        points_added += 1
                except (ValueError, RuntimeError, IndexError) as e:
                    logger.debug(f"  ⚠️ Could not get player COM at ts={fill_ts:.2f}: {e}")
                    continue

            if points_added > 0:
                logger.info(f"  ✅ Filled empty trajectory with {points_added} player COM points (fallback mode)")
                gaps_found += 1
            else:
                logger.warning(f"  ⚠️ Could not fill empty trajectory - no player COM data available")

        # ===== СЛУЧАЙ 2: Пропуски МЕЖДУ соседними точками =====
        elif len(times) >= 2:
            for i in range(len(times) - 1):
                ts = times[i]
                ts_next = times[i + 1]
                gap = ts_next - ts

                # Если gap > max_gap → заполняем player COM
                if gap > self.max_gap:
                    gaps_found += 1
                    logger.info(f"🔴 FILL GAP: {gap:.2f}s > {self.max_gap}s at ts={ts:.2f}→{ts_next:.2f}")

                    # Получаем позиции для заполнения
                    current_point = self.camera_trajectory[ts]
                    next_point = self.camera_trajectory[ts_next]
                    next_x = next_point['x']
                    next_y = next_point['y']

                    # Заполняем разрыв player COM с шагом 0.5s (15 кадров)
                    frame_step = 15
                    num_frames = int(gap * 30)
                    points_added = 0

                    for frame_idx in range(frame_step, num_frames, frame_step):
                        current_ts = ts + (frame_idx / 30.0)

                        # Не добавляем точку слишком близко к концу
                        if current_ts >= ts_next - 0.2:
                            break

                        try:
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
                        except (ValueError, RuntimeError) as e:
                            # Если players_history не имеет данных или ошибка
                            logger.debug(f"  ⚠️ Could not get player COM at ts={current_ts:.2f}: {e}")
                            continue

                    # Добавляем blend точку перед восстановлением мяча
                    transition_ts = ts + gap * 0.85
                    try:
                        player_com = players_history.get_player_com_for_timestamp(transition_ts)

                        if player_com:
                            alpha = 0.5
                            blend_x = (1 - alpha) * player_com[0] + alpha * next_x
                            blend_y = (1 - alpha) * player_com[1] + alpha * next_y

                            self.camera_trajectory[float(transition_ts)] = {
                                'x': blend_x,
                                'y': blend_y,
                                'timestamp': float(transition_ts),
                                'source_type': 'blend',
                                'confidence': 0.4
                            }
                            if points_added > 0:
                                logger.info(f"  ✅ Added {points_added} player COM + 1 blend point to fill gap")
                            else:
                                logger.info(f"  ℹ️ Added 1 blend point only (no player COM available for gap)")
                    except (ValueError, RuntimeError) as e:
                        logger.debug(f"  ⚠️ Could not get player COM at transition ts={transition_ts:.2f}: {e}")
                        if points_added > 0:
                            logger.info(f"  ℹ️ Added {points_added} player COM points (no blend point)")

        # Логирование итогов
        if gaps_found > 0:
            logger.info(f"📊 fill_gaps_in_trajectory: Found and filled {gaps_found} gaps, "
                       f"total trajectory points: {len(self.camera_trajectory)}")
        else:
            logger.info(f"✓ fill_gaps_in_trajectory: No gaps > {self.max_gap}s found")

    def _dump_trajectory_before_interpolation(self):
        """
        Выводит всю траектории ДО интерполяции.

        Показывает:
        - Какие точки исходные от мяча (source_type='ball')
        - Какие точки добавлены для заполнения gaps (source_type='player')
        - Какие точки переходные (source_type='blend')

        Это помогает понять структуру заполнения пропусков.
        """
        if not self.camera_trajectory:
            print("❌ TRAJECTORY EMPTY BEFORE INTERPOLATION")
            return

        times = sorted(self.camera_trajectory.keys())
        print(f"\n{'='*100}")
        print(f"📊 TRAJECTORY BEFORE INTERPOLATION: {len(self.camera_trajectory)} points")
        print(f"{'='*100}")

        # Группируем по source_type для статистики
        source_counts = {}
        for ts in times:
            source = self.camera_trajectory[ts].get('source_type', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"Source breakdown: {source_counts}")
        print(f"{'='*100}")
        print(f"{'Время':<10} {'X':<8} {'Y':<8} {'Тип':<20} {'Confidence':<12} {'Расстояние':<12}")
        print(f"{'-'*100}")

        prev_x, prev_y = None, None

        for ts in times:
            point = self.camera_trajectory[ts]
            x = point.get('x', 0)
            y = point.get('y', 0)
            source = point.get('source_type', 'unknown')
            conf = point.get('confidence', 0)

            # Вычисляем расстояние от предыдущей точки
            if prev_x is not None and prev_y is not None:
                distance = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                dist_str = f"{distance:6.1f}px"
            else:
                dist_str = "-"

            # Форматируем вывод
            source_name = {
                'ball': '🔴 BALL',
                'player': '🔵 PLAYER_COM',
                'blend': '🟡 BLEND',
                'interpolated': '⚪ INTERP',
                'interpolated_ball': '⚪ INTERP_BALL'
            }.get(source, f"? {source}")

            print(f"{ts:7.2f}s  {x:7.0f} {y:7.0f} {source_name:<20} {conf:6.2f}    {dist_str}")

            prev_x, prev_y = x, y

        print(f"{'='*100}\n")

    def _get_ball_scale(self, distance_px):
        """
        Линейное увеличение размера мяча в зависимости от расстояния между кадрами.

        Args:
            distance_px: расстояние между двумя соседними точками (в пиксelях)

        Returns:
            float: масштаб мяча (от 1.0 до 2.5)

        Логика:
        - distance_px < 50: scale = 1.0 (не реагируем на медленное движение)
        - 50 <= distance_px <= 500: scale плавно растёт от 1.0 к 2.5
        - distance_px > 500: scale = 2.5 (максимум, clamped)
        """
        min_distance = 50
        max_distance = 500
        min_scale = 1.0
        max_scale = 2.5

        if distance_px < min_distance:
            return 1.0

        if distance_px >= max_distance:
            return 2.5

        # Линейная интерполяция в диапазоне [50, 500]
        t = (distance_px - min_distance) / (max_distance - min_distance)  # t ∈ [0, 1]
        scale = min_scale + t * (max_scale - min_scale)

        return scale

    def _apply_speed_scaling(self):
        """
        После интерполяции добавляем ball_scale к каждой точке.

        ball_scale определяет размер мяча на основе скорости движения.
        Сразу применяем коэффициент к 'radius' или 'width' мяча в point dict.
        """
        times = sorted(self.camera_trajectory.keys())

        if len(times) < 2:
            # Если мало точек, просто установим scale = 1.0 для всех
            for point in self.camera_trajectory.values():
                point['ball_scale'] = 1.0
            return

        # Проходим по каждой точке, начиная со второй
        distances = []
        for i in range(1, len(times)):
            curr_time = times[i]
            prev_time = times[i - 1]

            curr_point = self.camera_trajectory[curr_time]
            prev_point = self.camera_trajectory[prev_time]

            # Считаем расстояние между соседними точками (в пиксельях)
            dx = curr_point['x'] - prev_point['x']
            dy = curr_point['y'] - prev_point['y']
            distance = math.sqrt(dx ** 2 + dy ** 2)
            distances.append(distance)

            # Получаем scale на основе расстояния
            scale = self._get_ball_scale(distance)

            # Добавляем в точку
            curr_point['ball_scale'] = scale

        # Для первой точки используем значение от второй
        if len(times) > 1:
            self.camera_trajectory[times[0]]['ball_scale'] = \
                self.camera_trajectory[times[1]]['ball_scale']
        else:
            self.camera_trajectory[times[0]]['ball_scale'] = 1.0

        # Логирование статистики
        scales = [p.get('ball_scale', 1.0) for p in self.camera_trajectory.values()]
        if scales:
            min_scale = min(scales)
            max_scale = max(scales)
            avg_scale = sum(scales) / len(scales)
            msg = f"📊 SPEED_SCALING: scale range [{min_scale:.2f}, {max_scale:.2f}], average={avg_scale:.2f}, points={len(self.camera_trajectory)}"
            if distances:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                msg += f" | distance range [{min_dist:.2f}, {max_dist:.2f}], avg={avg_dist:.2f}"
            logger.info(msg)

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

    def print_full_trajectory(self, label="TRAJECTORY", max_points=50):
        """
        Выводит полную траекторию для анализа (первые N и последние N точек).

        Args:
            label: префикс для вывода
            max_points: максимум точек для вывода (первые N и последние N)
        """
        if not self.camera_trajectory:
            logger.info(f"📭 {label}: Empty trajectory")
            return

        times = sorted(self.camera_trajectory.keys())
        sources_count = {}

        # Count sources
        for ts in times:
            source = self.camera_trajectory[ts].get('source_type', 'unknown')
            sources_count[source] = sources_count.get(source, 0) + 1

        # First, log summary
        summary = f"\n{'='*100}\n📊 {label}: {len(self.camera_trajectory)} points total, time span [{times[0]:.2f}, {times[-1]:.2f}]s, sources={sources_count}\n"
        logger.info(summary)
        print(summary)

        # Then log first N and last N points
        display_times = list(times[:max_points//2]) + list(times[-(max_points//2):])

        header = f"📍 First {max_points//2} + Last {max_points//2} points:\n"
        logger.info(header)
        print(header)

        for i, ts in enumerate(display_times):
            if i == max_points//2 and len(times) > max_points:
                omitted_msg = f"  ... ({len(times) - max_points} points omitted) ...\n"
                logger.info(omitted_msg)
                print(omitted_msg)

            point = self.camera_trajectory[ts]
            source = point.get('source_type', 'unknown')
            scale = point.get('ball_scale', 'N/A')
            scale_str = f"{scale:.2f}" if isinstance(scale, float) else str(scale)

            line = f"  t={ts:7.2f}: ({point['x']:7.0f}, {point['y']:7.0f}) [{source:15s}] scale={scale_str:5s} conf={point.get('confidence', 0.0):.2f}\n"
            logger.info(line)
            print(line, end='')

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
