#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Camera Probe Handler

Handles virtual camera parameter updates including:
- Ball tracking with speed-based zoom
- Player fallback when ball is lost
- Smooth radius interpolation
- Speed calculation from trajectory
"""

import time
import math
import logging
from gi.repository import Gst

logger = logging.getLogger("panorama-virtualcam")


class VirtualCameraProbeHandler:
    """
    Handles virtual camera probe updates for ball tracking.

    This class manages:
    - Ball position tracking with smooth interpolation
    - Speed-based dynamic zoom
    - Player center of mass fallback when ball is lost
    - Smooth transitions between tracking modes
    """

    def __init__(self, ball_history, players_history, all_detections_history,
                 vcam=None, radius_smooth_factor=0.3,
                 speed_low_threshold=300.0, speed_high_threshold=1200.0,
                 speed_zoom_max_factor=3.0):
        """
        Initialize the VirtualCameraProbeHandler.

        Args:
            ball_history: BallDetectionHistory instance for ball tracking
            players_history: PlayersHistory instance for player tracking
            all_detections_history: Dict of all detections by timestamp for multi-class tracking
            vcam: Virtual camera element (can be set later via set_vcam)
            radius_smooth_factor: Smoothing factor for radius interpolation (0-1)
            speed_low_threshold: Speed threshold to start zoom reaction (px/s)
            speed_high_threshold: Speed threshold for maximum zoom (px/s)
            speed_zoom_max_factor: Maximum zoom factor at high speed
        """
        # Dependencies
        self.history = ball_history
        self.players_history = players_history
        self.all_detections_history = all_detections_history
        self.vcam = vcam

        # Configuration parameters
        self.radius_smooth_factor = radius_smooth_factor
        self.speed_low_threshold = speed_low_threshold
        self.speed_high_threshold = speed_high_threshold
        self.speed_zoom_max_factor = speed_zoom_max_factor

        # State variables for tracking
        self.display_frame_count = 0
        self.current_display_timestamp = 0.0

        # Radius smoothing state
        self.smooth_ball_radius = 20.0  # Initial value

        # Speed calculation state
        self.last_speed_calc_pos = None
        self.last_speed_calc_time = 0
        self.current_smooth_speed = 0.0
        self.speed_zoom_factor = 1.6

        # Center of mass computation cache
        self._sorted_history_keys_cache = []
        self._sorted_history_keys_cache_time = 0
        self.players_center_mass_smoothed = None

    def set_vcam(self, vcam):
        """Set the virtual camera element."""
        self.vcam = vcam

    def handle_vcam_update_probe(self, pad, info, u_data):
        """Обновление параметров виртуальной камеры."""
        try:
            if not self.vcam:
                # Дебаг: vcam не создан!
                if self.display_frame_count == 0:
                    logger.warning("⚠️ vcam is None! Ball drawing disabled")
                return Gst.PadProbeReturn.OK

            # Получаем текущий timestamp
            buffer = info.get_buffer()
            if not buffer:
                return Gst.PadProbeReturn.OK

            ts = buffer.pts / 1e9 if buffer.pts != Gst.CLOCK_TIME_NONE else time.time()

            # Обновляем timestamp в истории
            self.history.update_display_timestamp(ts)

            # Обновляем текущий timestamp для backward interpolation
            self.current_display_timestamp = ts

            # Получаем детекцию для текущего времени
            det = self.history.get_detection_for_timestamp(ts)

            # Дебаг: показываем статус детекций (первые 5 раз)
            if self.display_frame_count < 5:
                history_size = len(self.history.raw_future_history) + len(self.history.processed_future_history) + len(self.history.confirmed_history)
                logger.info(f"🎨 vcam_update_probe: frame={self.display_frame_count}, ts={ts:.2f}, "
                           f"history_size={history_size}, det={'found' if det else 'None'}")

            if det is None:
                # ========== МЯЧ ПОТЕРЯН - ПРОБУЕМ ЦЕНТРИРОВАТЬ ПО ИГРОКАМ ==========
                # Вычисляем центр масс игроков как fallback
                players_center = self.players_history.calculate_center_of_mass(ts)

                if players_center:
                    # Центрируем камеру по центру масс игроков с МАКСИМАЛЬНЫМ зумом
                    self.vcam.set_property("ball-x", float(players_center[0]))
                    self.vcam.set_property("ball-y", float(players_center[1]))
                    self.vcam.set_property("ball-radius", 50.0)  # Максимальный зум для обзора всего поля

                    if self.display_frame_count % 30 == 0:
                        logger.info(f"⚽→👥 Ball lost! Centering on players center: ({players_center[0]:.0f}, {players_center[1]:.0f}) with max zoom (radius=50)")
                else:
                    # Нет ни мяча, ни игроков - максимальное отдаление
                    self.vcam.set_property("ball-radius", 50.0)

                    if self.display_frame_count % 30 == 0:
                        logger.warning(f"⚠️ Ball and players lost! Max zoom out (radius=50px → FOV=68°)")

                self.display_frame_count += 1
                return Gst.PadProbeReturn.OK

            #save_detection_to_csv(det, ts, self.display_frame_count, file_path='ball_display_used.csv')
            # Извлекаем координаты и размер
            # det = [cx, cy, w, h, conf, 0, cx_global, cy_global, w_global, h_global]
            cx_g = det[6] if len(det) > 6 else det[0]  # Используем глобальные координаты если есть
            cy_g = det[7] if len(det) > 7 else det[1]
            ball_width = det[8] if len(det) > 8 else det[2]
            ball_radius_raw = ball_width / 2.0  # Сырой радиус из детекции

            # ========== ИНТЕРПОЛЯЦИЯ РАЗМЕРА МЯЧА ==========
            # Плавное изменение радиуса для плавного зума (аналогично координатам)
            # smooth_radius = smooth_radius * (1 - alpha) + new_radius * alpha
            self.smooth_ball_radius = (self.smooth_ball_radius * (1.0 - self.radius_smooth_factor) +
                                      ball_radius_raw * self.radius_smooth_factor)

            # Используем сглаженный радиус (пока без учёта скорости)
            ball_radius_base = self.smooth_ball_radius

            # ========== РАСЧЕТ СКОРОСТИ ==========
            current_time = time.time()
            if self.last_speed_calc_pos and (current_time - self.last_speed_calc_time) > 0.1:
                dx = cx_g - self.last_speed_calc_pos[0]
                dy = cy_g - self.last_speed_calc_pos[1]
                dt = current_time - self.last_speed_calc_time

                if dt > 0:
                    speed = math.sqrt(dx*dx + dy*dy) / dt
                    # Сглаживание скорости
                    self.current_smooth_speed = (self.current_smooth_speed * 0.7 +
                                                speed * 0.3)

                    # Расчет коэффициента зума на основе скорости
                    if self.current_smooth_speed > self.speed_low_threshold:
                        speed_normalized = min(
                            (self.current_smooth_speed - self.speed_low_threshold) /
                            (self.speed_high_threshold - self.speed_low_threshold),
                            1.0
                        )
                        self.speed_zoom_factor = 1.0 + (self.speed_zoom_max_factor - 1.0) * speed_normalized
                    else:
                        self.speed_zoom_factor = max(1.0, self.speed_zoom_factor * 0.95)

                    self.last_speed_calc_pos = (cx_g, cy_g)
                    self.last_speed_calc_time = current_time

            if self.last_speed_calc_pos is None:
                self.last_speed_calc_pos = (cx_g, cy_g)
                self.last_speed_calc_time = current_time

            # ========== ПРИМЕНЕНИЕ КОЭФФИЦИЕНТА СКОРОСТИ К РАДИУСУ ==========
            # При быстром движении мяча увеличиваем его эффективный размер,
            # чтобы виртуальная камера автоматически отдалялась (увеличивала FOV)
            # Это даёт больше контекста и показывает куда движется мяч
            ball_radius_unclamped = ball_radius_base * self.speed_zoom_factor

            # Ограничиваем радиус: минимум 5px, максимум 50px (ограничения плагина)
            # radius=5 → FOV=40° (приближение), radius=50 → FOV=68° (отдаление)
            ball_radius = min(max(ball_radius_unclamped, 5.0), 50.0)

            # ========== ПЕРЕДАЧА ДАННЫХ В ПЛАГИН ==========
            # Желаемый размер мяча на экране с учетом скорости
            target_ball_size = 0.055 * self.speed_zoom_factor

            # Ограничиваем значения
            target_ball_size = min(max(target_ball_size, 0.05), 0.15)

            # Передаем в плагин только данные о мяче
            self.vcam.set_property("ball-x", float(cx_g))
            self.vcam.set_property("ball-y", float(cy_g))
            self.vcam.set_property("ball-radius", float(ball_radius))
            self.vcam.set_property("target-ball-size", float(target_ball_size))

            # Логирование каждые 30 кадров
            if self.display_frame_count % 30 == 0:
                clamped_suffix = "" if ball_radius == ball_radius_unclamped else f"→{ball_radius:.1f}px(clamped)"
                logger.info(f"Ball tracking: pos=({cx_g:.0f},{cy_g:.0f}), "
                        f"radius={ball_radius_raw:.1f}px→{ball_radius_base:.1f}px(smooth)→{ball_radius_unclamped:.1f}px(speed×{self.speed_zoom_factor:.2f}){clamped_suffix}, "
                        f"speed={self.current_smooth_speed:.0f}px/s, "
                        f"target_size={target_ball_size:.3f}")

            self.display_frame_count += 1
            return Gst.PadProbeReturn.OK

        except Exception as e:
            logger.error(f"vcam_update_probe error: {e}")
            return Gst.PadProbeReturn.OK

    def get_all_detections_for_timestamp(self, ts, max_delta=0.12):
        """Получить все детекции (всех классов) для timestamp."""
        if not self.all_detections_history:
            return None

        # Находим ближайший timestamp
        timestamps = list(self.all_detections_history.keys())
        if not timestamps:
            return None

        closest_ts = min(timestamps, key=lambda t: abs(t - ts))
        if abs(closest_ts - ts) > max_delta:
            return None

        return self.all_detections_history[closest_ts]

    def _compute_smoothed_center_of_mass(self, current_ts):
        """Вычисляет сглаженный центр масс используя всю доступную историю (7 секунд)."""
        if not self.all_detections_history:
            return

        # ОПТИМИЗАЦИЯ: Кеширование вычислений (без пропуска кадров, т.к. нужно для отрисовки)
        # Вычисляем каждый кадр, но кешируем сортировку

        lookback = 7.0  # Используем весь буфер
        start_ts = current_ts - lookback

        # ОПТИМИЗАЦИЯ: Кешируем отсортированные ключи
        if not hasattr(self, '_sorted_history_keys_cache'):
            self._sorted_history_keys_cache = []
            self._sorted_history_keys_cache_time = 0

        # Обновляем кеш только если история изменилась
        if len(self.all_detections_history) != len(self._sorted_history_keys_cache):
            self._sorted_history_keys_cache = sorted(self.all_detections_history.keys())

        # Собираем все центры масс за последние 7 секунд
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
            # Недостаточно данных
            if self.display_frame_count % 30 == 0:
                logger.warning(f"⚠️ COM: Not enough data - only {len(centers_history)} points in history")
            return

        # Применяем медианный фильтр для удаления выбросов
        # Берём последние 30 точек (примерно 1 секунда при 30 FPS)
        recent_centers = centers_history[-30:] if len(centers_history) >= 30 else centers_history

        # Вычисляем медиану X и Y
        x_values = [c[1] for c in recent_centers]
        y_values = [c[2] for c in recent_centers]

        # Сортируем и берём медиану
        x_values_sorted = sorted(x_values)
        y_values_sorted = sorted(y_values)

        n = len(x_values_sorted)
        if n % 2 == 0:
            median_x = (x_values_sorted[n//2-1] + x_values_sorted[n//2]) / 2
            median_y = (y_values_sorted[n//2-1] + y_values_sorted[n//2]) / 2
        else:
            median_x = x_values_sorted[n//2]
            median_y = y_values_sorted[n//2]

        # Фильтруем выбросы: оставляем только точки близкие к медиане
        filtered_centers = []
        for ts, x, y in recent_centers:
            dist_to_median = ((x - median_x)**2 + (y - median_y)**2)**0.5
            # Отсекаем точки дальше 200px от медианы
            if dist_to_median < 200:
                filtered_centers.append((ts, x, y))

        if not filtered_centers:
            filtered_centers = recent_centers  # Fallback

        # Взвешенное среднее: свежие точки важнее
        # Используем экспоненциальные веса: более свежие точки получают больший вес
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for i, (ts, x, y) in enumerate(filtered_centers):
            # Вес растёт экспоненциально для более свежих точек
            # Последняя точка получает максимальный вес
            weight = (i + 1) ** 1.5  # Степень 1.5 даёт хороший баланс
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        if total_weight > 0:
            final_x = weighted_x / total_weight
            final_y = weighted_y / total_weight

            # Обновляем smoothed позицию напрямую (без EMA, так как уже сгладили)
            self.players_center_mass_smoothed = (final_x, final_y)

            # ДЕБАГ: Логируем обновление (каждые 30 кадров)
            if self.display_frame_count % 30 == 0:
                logger.info(f"🎯 Center of Mass updated: ({final_x:.0f}, {final_y:.0f}), "
                           f"from {len(filtered_centers)} points")
