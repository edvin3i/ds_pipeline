#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЭКСПЕРИМЕНТАЛЬНАЯ ВЕРСИЯ С МУЛЬТИКЛАССОВОЙ ДЕТЕКЦИЕЙ

Панорама с двумя режимами отображения и записью:
- panorama: полная панорама с отрисовкой bbox через nvdsosd
- virtualcam: виртуальная камера, следящая за мячом (с возможностью записи)
- stream: стриминг на stream
Поддержка источников: файлы или камеры MIPI CSI

=== МУЛЬТИКЛАССОВЫЕ ВОЗМОЖНОСТИ ===
1. Детекция 5 классов: ball, player, staff, side_referee, main_referee
2. Хранение истории игроков для расчёта центра масс
3. Fallback: при потере мяча камера центрируется на игроках
4. Визуализация в panorama режиме:
   - Мяч: КРАСНЫЙ цвет (border=3)
   - Игроки: ЗЕЛЁНЫЙ цвет (border=2)
   - Лимит отрисовки: 16 объектов (ограничение nvdsosd на Jetson)
   - Тайлы ОТКЛЮЧЕНЫ для экономии слотов

ПРИОРИТЕТ ОТРИСОВКИ: мяч → игроки (персонал и судьи отключены)
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
import ctypes
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
import logging
import time
import math
import threading
import csv
import cv2 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("panorama-virtualcam")

# ============================================================
# Инициализация GStreamer
# ============================================================
# ВАЖНО: Устанавливаем путь к кастомным плагинам ПЕРЕД Gst.init()
tilebatcher_path = "/home/nvidia/deep_cv_football/my_tile_batcher/src"
virtualcam_path = "/home/nvidia/deep_cv_football/my_virt_cam/src"
ringbuffer_path = "/home/nvidia/deep_cv_football/my_ring_buffer"
os.environ['GST_PLUGIN_PATH'] = f"{tilebatcher_path}:{virtualcam_path}:{ringbuffer_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

Gst.init(None)


# ============================================================
# КОНСТАНТЫ КОНФИГУРАЦИИ ПАНОРАМЫ
# ============================================================
# Размеры панорамы (обновлено с 1632 на 1800 для поддержки FOV до 75°)
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

# Параметры тайлов для nvtilebatcher
TILE_WIDTH = 1024
TILE_HEIGHT = 1024
TILES_COUNT = 6

# Вертикальный отступ тайлов: рассчитан АВТОМАТИЧЕСКИ на основе маски поля
# Логика: находим центр поля (field_top + field_bottom)/2, вычитаем половину тайла (512)
# Field bounds: top=438, bottom=1454 → center=946 → offset = 946 - 512 = 434
TILE_OFFSET_Y = 434  # Рассчитано из field_mask.png (было: 304 симметричное)
TILE_OFFSET_X = 192  # Горизонтальный margin для 6 тайлов (6×1024=6144, margin=(6528-6144)/2)

# Координаты тайлов (автоматически вычисляются)
TILE_POSITIONS = [
    (TILE_OFFSET_X,                   TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 0
    (TILE_OFFSET_X + TILE_WIDTH,      TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 1
    (TILE_OFFSET_X + TILE_WIDTH * 2,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 2
    (TILE_OFFSET_X + TILE_WIDTH * 3,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 3
    (TILE_OFFSET_X + TILE_WIDTH * 4,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 4
    (TILE_OFFSET_X + TILE_WIDTH * 5,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 5
]


# =========================
# УТИЛИТЫ ДЛЯ CSV
# =========================

def save_detection_to_csv(detection, timestamp, frame_num, file_path=None):
    """Запись детекции в TSV файл."""
    import os
    file_path = file_path or "ball_events.tsv"
    
    ts_round = round(float(timestamp), 6)
    
    if detection is None:
        cx = cy = w = h = conf = 0
        cx_gl = cy_gl = w_gl = h_gl = 0
    else:
        cx, cy, w, h, conf = detection[0:5]
        if len(detection) >= 10:
            cx_gl, cy_gl, w_gl, h_gl = detection[6:10]
        else:
            cx_gl = cy_gl = w_gl = h_gl = 0
    
    row = {
        'system_time': time.time(),
        'frame_timestamp': ts_round,
        'frame_num': int(frame_num),
        'cx': cx, 'cy': cy, 'width': w, 'height': h, 'confidence': conf,
        'cx_global': cx_gl, 'cy_global': cy_gl, 'width_global': w_gl, 'height_global': h_gl
    }
    
    fieldnames = ['system_time', 'frame_timestamp', 'frame_num',
                  'cx', 'cy', 'width', 'height', 'confidence',
                  'cx_global', 'cy_global', 'width_global', 'height_global']
    
    new_file = not os.path.exists(file_path)
    try:
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.warning(f"CSV append error: {e}")




class FieldMaskBinary:
    """Бинарная маска поля для фильтрации детекций."""
    
    def __init__(self, mask_path='field_mask.png', panorama_width=PANORAMA_WIDTH, panorama_height=PANORAMA_HEIGHT):
        self.width = panorama_width
        self.height = panorama_height
        
        if mask_path and os.path.exists(mask_path):
            # Загружаем маску
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img.shape != (self.height, self.width):
                mask_img = cv2.resize(mask_img, (self.width, self.height))
            self.mask = (mask_img > 127).astype(np.uint8)
            logger.info(f"✓ Маска поля загружена: {mask_path}")
        else:
            # Без маски - всё разрешено
            self.mask = np.ones((self.height, self.width), dtype=np.uint8)
            logger.warning(f"Маска не найдена: {mask_path}")
    
    def is_inside_field(self, x, y):
        """Проверка точки - O(1)."""
        x, y = int(x), int(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.mask[y, x] > 0
        return False

# =========================
# ПОЛНАЯ ИСТОРИЯ ДЕТЕКЦИЙ
# =========================

class BallDetectionHistory:
    """Полная версия истории детекций из big_x.py."""
    
    def __init__(self, history_duration=10.0, cleanup_interval=1000):
        self.confirmed_history = {}
        self.raw_future_history = {}
        self.processed_future_history = {}
        
        self.history = self.raw_future_history
        self.interpolated_history = self.processed_future_history
        
        self.history_lock = threading.RLock()
        self.current_display_timestamp = 0.0
        
        self.max_confirmed_points = 200
        self.frame_index = {}
        self.last_detection = None
        self.last_detection_time = 0
        self.last_detection_return = None
        
        self.history_duration = float(history_duration)
        self.cleanup_interval = int(cleanup_interval)
        self.frame_counter = 0

        self.last_process_time = 0

        self.outlier_removal_count = defaultdict(int)  # Счётчик удалений
        self.permanent_blacklist = set()               # Пожизненный бан (автоматический)

        self.outlier_ban_threshold = 6                 # 6 удалений = бан (уменьшено с 12)
        self.ban_radius = 30                           # Радиус бана в пикселях (увеличено с 5)
        
    def add_detection(self, detection, timestamp, frame_num):
        """Добавляем детекцию в историю с фильтрацией дублей."""
        # ДЕБАГ: Логируем каждый вызов
        if detection is not None:
            logger.info(f"🔵 CALL add_detection: ts={timestamp:.2f}, pos=({detection[6]:.0f},{detection[7]:.0f})")
        else:
            logger.info(f"🔵 CALL add_detection: ts={timestamp:.2f}, detection=None")

        if detection is None:
            return

        # Фильтруем дубли по близости глобальных координат
        if self.last_detection is not None:
            prev_x, prev_y = self.last_detection[6], self.last_detection[7]
            curr_x, curr_y = detection[6], detection[7]
            dx = abs(curr_x - prev_x)
            dy = abs(curr_y - prev_y)
            if dx <= 2 and dy <= 2:
                # ДЕБАГ: дубликат отфильтрован (логируем ВСЕГДА, но ограничиваем частоту)
                if not hasattr(self, '_dup_log_counter'):
                    self._dup_log_counter = 0
                self._dup_log_counter += 1
                if self._dup_log_counter % 5 == 0:  # каждый 5-й дубликат
                    logger.info(f"⛔ DUPLICATE #{self._dup_log_counter}: pos=({curr_x:.0f},{curr_y:.0f}), prev=({prev_x:.0f},{prev_y:.0f}), diff=({dx},{dy}), time_since_last={timestamp - self.last_detection_time:.2f}s")
                return

        self.last_detection = list(detection)
        self.last_detection_time = float(timestamp)

        with self.history_lock:
            size_before = len(self.raw_future_history) + len(self.processed_future_history) + len(self.confirmed_history)

            self.raw_future_history[float(timestamp)] = detection
            self.frame_index[int(frame_num)] = float(timestamp)

            # Логируем добавление детекции каждые 30 кадров
            if self.frame_counter % 30 == 0:
                logger.info(f"📥 ADD: ts={timestamp:.2f}, pos=({detection[6]:.0f},{detection[7]:.0f}), "
                           f"hist_size: raw={len(self.raw_future_history)}, proc={len(self.processed_future_history)}, conf={len(self.confirmed_history)}")

            #save_detection_to_csv(detection, timestamp, frame_num, file_path='ball_raw_future.csv')

            # ДЕБАГ: ДО обработки
            # if self.frame_counter < 10:
            #     logger.info(f"📝 BEFORE process: ts={timestamp:.2f}, display_ts={self.current_display_timestamp:.2f}, "
            #                f"raw={len(self.raw_future_history)}, processed={len(self.processed_future_history)}, confirmed={len(self.confirmed_history)}")

            self.frame_counter += 1
            self._process_future_history()

            size_after = len(self.raw_future_history) + len(self.processed_future_history) + len(self.confirmed_history)

            # ДЕБАГ: ПОСЛЕ обработки
            # if self.frame_counter <= 10:
            #     logger.info(f"📝 AFTER process #{self.frame_counter}: ts={timestamp:.2f}, "
            #                f"pos=({detection[6]:.0f},{detection[7]:.0f}), "
            #                f"history: {size_before}→{size_after} (raw={len(self.raw_future_history)}, "
            #                f"processed={len(self.processed_future_history)}, confirmed={len(self.confirmed_history)})")
            
    def update_display_timestamp(self, timestamp):
        """Обновляем точку показанного времени."""
        with self.history_lock:
            self.current_display_timestamp = float(timestamp)
            
    def get_detection_for_timestamp(self, timestamp, max_delta=0.12):
        """Найти детекцию для заданного timestamp с интерполяцией на лету."""
        with self.history_lock:
            # ВАЖНО: НЕ обновляем current_display_timestamp здесь!
            # Это делается через update_display_timestamp() отдельно
            # Иначе playback (который ОТСТАЕТ на 7 сек) будет удалять
            # свежие детекции из analysis!

            # НЕ вызываем _process_future_history() здесь!
            # Она вызывается в add_detection() при новых детекциях

            # Сначала ищем точное совпадение
            exact = self.processed_future_history.get(timestamp)
            if exact:
                return exact.copy()

            # Ищем в confirmed_history
            exact = self.confirmed_history.get(timestamp)
            if exact:
                return exact.copy()

            # Теперь интерполируем между соседними точками
            all_history = {}
            all_history.update(self.confirmed_history)
            all_history.update(self.processed_future_history)

            if not all_history:
                # Логируем если история пуста
                if self.frame_counter % 30 == 0:
                    logger.warning(f"📭 HISTORY EMPTY: ts={timestamp:.2f}, no detections in history")
                return self.last_detection_return.copy() if self.last_detection_return else None
            
            times = sorted(all_history.keys())
            before_ts = None
            after_ts = None
            
            for t in times:
                if t <= timestamp:
                    before_ts = t
                elif t > timestamp and after_ts is None:
                    after_ts = t
                    break
            
            # Интерполяция между точками
            if before_ts and after_ts:
                gap = after_ts - before_ts

                # Проверка: если разрыв больше 3 секунд - НЕ интерполируем (явная потеря мяча)
                if gap > 3.0:
                    if self.frame_counter % 30 == 0:
                        logger.warning(f"⚠️ GAP TOO LARGE for interpolation: {gap:.2f}s "
                                      f"between {before_ts:.2f} and {after_ts:.2f}, "
                                      f"requested ts={timestamp:.2f} → switching to players fallback")

                    # Пробуем использовать ближайшую точку, если она достаточно близко
                    if abs(before_ts - timestamp) < abs(after_ts - timestamp):
                        if abs(before_ts - timestamp) < max_delta:
                            det = all_history[before_ts].copy()
                            self.last_detection_return = det.copy()
                            return det
                    else:
                        if abs(after_ts - timestamp) < max_delta:
                            det = all_history[after_ts].copy()
                            self.last_detection_return = det.copy()
                            return det

                    # Разрыв слишком большой - возвращаем None для переключения на игроков
                    return None

                # Нормальная интерполяция для разрывов <= 3 секунд
                det = self._interpolate_between_points(
                    all_history[before_ts],
                    all_history[after_ts],
                    before_ts,
                    after_ts,
                    timestamp
                )
                self.last_detection_return = det.copy()
                return det
            
            # Если только одна точка до или после
            if before_ts and abs(before_ts - timestamp) < max_delta:
                det = all_history[before_ts].copy()
                self.last_detection_return = det.copy()
                return det

            if after_ts and abs(after_ts - timestamp) < max_delta:
                det = all_history[after_ts].copy()
                self.last_detection_return = det.copy()
                return det

            # Не нашли подходящую детекцию
            if self.frame_counter % 30 == 0:
                logger.warning(f"⚠️ NO MATCH: ts={timestamp:.2f}, before={before_ts:.2f if before_ts else None}, "
                              f"after={after_ts:.2f if after_ts else None}, hist_size={len(all_history)}")

            return self.last_detection_return.copy() if self.last_detection_return else None

    def _interpolate_between_points(self, det1, det2, ts1, ts2, target_ts):
        """Интерполяция между двумя детекциями для конкретного timestamp."""
        if not det1 or not det2:
            return det1 or det2
        
        gap = ts2 - ts1
        t = (target_ts - ts1) / gap if gap > 0 else 0.5
        t = max(0.0, min(1.0, t))  # Ограничиваем [0, 1]
        
        result = list(det1)
        
        # Интерполируем позиции и размеры
        # Позиции локальные (индексы 0-3)
        for i in [0, 1, 2, 3]:
            if i < len(det1) and i < len(det2):
                result[i] = det1[i] + (det2[i] - det1[i]) * t
        
        # Позиции глобальные (индексы 6-9)
        for i in [6, 7, 8, 9]:
            if i < len(det1) and i < len(det2):
                result[i] = det1[i] + (det2[i] - det1[i]) * t
        
        # Для больших разрывов (полет мяча) добавляем параболическую траекторию
        if gap > 1.0:  # Больше 1 секунды - вероятно полет
            # Высота параболы зависит от расстояния
            if len(det1) > 6 and len(det2) > 6:
                dx = abs(det2[6] - det1[6])
                dy = abs(det2[7] - det1[7])
                distance = math.sqrt(dx*dx + dy*dy)

                # Высота полета пропорциональна расстоянию
                max_height = min(150, distance * 0.1)

                # Параболическая поправка для Y (4t(1-t) дает параболу с максимумом в t=0.5)
                parabola_factor = 4 * t * (1 - t)
                y_offset = max_height * parabola_factor

                # Логируем применение параболической траектории (только для дебага)
                # if abs(t - 0.1) < 0.05:  # Логируем только для первой интерполированной точки
                #     logger.info(f"    🛸 Applying parabolic trajectory: gap={gap:.2f}s, dist={distance:.0f}px, "
                #                f"max_height={max_height:.0f}px, y_offset_at_t={t:.2f} is {y_offset:.0f}px")

                # Применяем поправку (вверх это минус по Y)
                result[1] -= y_offset  # Локальная Y
                result[7] -= y_offset  # Глобальная Y

                # Размер мяча тоже меняется при полете
                size_factor = 1.0 + (y_offset / 200)  # Увеличивается когда выше
                result[2] *= size_factor
                result[3] *= size_factor
                result[8] *= size_factor
                result[9] *= size_factor
        
        # Уверенность интерполируем или берем минимум
        if len(det1) > 4 and len(det2) > 4:
            result[4] = min(det1[4], det2[4]) * 0.8  # Снижаем уверенность для интерполированных
        
        return result


    def get_future_trajectory(self, current_timestamp, look_ahead_seconds=1.0, max_points=10):
        """
        Получает будущую траекторию мяча относительно текущего времени.
        
        Args:
            current_timestamp: Текущее время отображения
            look_ahead_seconds: На сколько секунд вперёд смотреть
            max_points: Максимальное количество точек
            
        Returns:
            List[dict]: Список точек с полями 'time', 'x', 'y', 'width'
        """
        with self.history_lock:
            future_points = []
            
            # Определяем временной диапазон
            start_time = float(current_timestamp)
            end_time = start_time + float(look_ahead_seconds)
            
            # Собираем точки из обработанной истории будущего
            for ts, det in self.processed_future_history.items():
                if start_time <= float(ts) <= end_time and det:
                    future_points.append({
                        'time': float(ts),
                        'x': float(det[0]),
                        'y': float(det[1]),
                        'width': float(det[2]) if len(det) > 2 else 0
                    })
            
            # Если мало точек в processed, добавляем из confirmed
            if len(future_points) < 3:
                for ts, det in self.confirmed_history.items():
                    if float(ts) > start_time and float(ts) <= end_time and det:
                        future_points.append({
                            'time': float(ts),
                            'x': float(det[0]),
                            'y': float(det[1]),
                            'width': float(det[2]) if len(det) > 2 else 0
                        })
            
            # Сортируем по времени и ограничиваем количество
            future_points.sort(key=lambda p: p['time'])
            return future_points[:max_points]
            
            
    def _find_detection_in_history(self, history, timestamp, max_delta=0.12):
        """Внутренний поиск ближайшей точки."""
        if not history:
            return None
            
        t = float(timestamp)
        closest_ts = None
        min_diff = 1e9
        
        for ts in sorted(history.keys()):
            diff = abs(float(ts) - t)
            if diff < min_diff:
                min_diff = diff
                closest_ts = ts
                
        if closest_ts is None or min_diff > float(max_delta):
            return None
            
        return history[closest_ts]
        
    def _process_future_history(self):
        """Полная обработка истории с агрессивной интерполяцией."""
        # Проверяем, не слишком ли часто вызывается
        current_time = time.time()
        if not hasattr(self, 'last_full_process_time'):
            self.last_full_process_time = 0

        # Ограничиваем частоту тяжелой обработки
        time_since_last = current_time - self.last_full_process_time
        need_heavy_processing = (
            time_since_last >= 0.5 or  # Прошло минимум 0.5 сек
            len(self.raw_future_history) >= 10  # Или накопилось много данных
        )

        # Всегда делаем легкие операции
        self._transfer_displayed_to_confirmed()
        self._cleanup_confirmed_history()

        # Обрабатываем даже если мало точек
        if len(self.raw_future_history) >= 2:
            # Получаем контекст из подтверждённой истории
            context_points = self._get_context_from_confirmed(num_points=30)

            # Объединяем контекст с сырой историей
            combined_history = {}
            combined_history.update(context_points)
            combined_history.update(self.raw_future_history)

            # Тяжелую очистку делаем только если нужно
            if need_heavy_processing:
                # Очистка от выбросов
                cleaned_combined = self.detect_and_remove_false_trajectories(combined_history)

                # Дополнительная очистка
                refined_combined = self.clean_detection_history(
                    cleaned_combined,
                    preserve_recent_seconds=0.3,
                    outlier_threshold=2.5,
                    window_size=3
                )
                self.last_full_process_time = current_time
            else:
                # Без очистки, просто используем как есть
                refined_combined = combined_history

            # Извлекаем только будущую часть
            lookback_buffer = 1.0
            cutoff_time = self.current_display_timestamp - lookback_buffer

            future_only = {
                ts: det for ts, det in refined_combined.items()
                if ts > cutoff_time
            }

            # ДЕБАГ: первые 3 раза
            # if self.frame_counter < 3:
            #     logger.info(f"🔍 _process: cutoff_time={cutoff_time:.2f}, refined={len(refined_combined)}, future_only={len(future_only)}")

            # ВСЕГДА интерполируем (это быстро)
            self.processed_future_history = self.interpolate_history_gaps(
                future_only,
                fps=30,
                max_gap=10.0  # Поддержка длинных полетов
            )
            self.interpolated_history = self.processed_future_history
            
    def detect_and_remove_false_trajectories(self, history):
        """Обнаружение и удаление выбросов с пожизненным баном для упорных."""
        if len(history) < 5:
            return history
            
        clean_history = dict(history)
        coords = []
        
        # Собираем все точки для анализа
        for ts in sorted(history.keys()):
            det = history[ts]
            if det and len(det) >= 8:
                # Сначала проверяем пожизненный бан
                coord_key = (int(det[6]), int(det[7]))
                if coord_key in self.permanent_blacklist:
                    del clean_history[ts]
                    #logger.debug(f"⛔ Blocked banned point: ({d['x']:.0f}, {d['y']:.0f})")
                    continue
                    
                coords.append({
                    'ts': ts,
                    'x': det[6], 
                    'y': det[7],
                    'det': det
                })
        
        if len(coords) < 5:  # Изменено с 3 на 5
            return clean_history
        
        # НОВОЕ: Сначала считаем частоту позиций
        position_frequency = defaultdict(list)
        for i, point in enumerate(coords):
            key = (round(point['x']/30)*30, round(point['y']/30)*30)
            position_frequency[key].append(i)
        
        # НОВОЕ: Находим подозрительно частые позиции
        suspicious_positions = set()
        for pos_key, indices in position_frequency.items():
            if len(indices) >= 3:  # Появляется 3+ раза
                suspicious_positions.add(pos_key)
                logger.debug(f"Suspicious frequent position {pos_key}: {len(indices)} times")
        
        outliers_to_remove = []
        
        # Проверяем каждую точку с расширенным контекстом
        for i in range(len(coords)):
            curr = coords[i]
            curr_key = (round(curr['x']/30)*30, round(curr['y']/30)*30)
            
            # НОВОЕ: Если точка из частых позиций
            if curr_key in suspicious_positions:
                # Проверяем, действительно ли это выброс
                is_outlier = False
                
                # Проверка 1: Большое расстояние до соседей
                if i > 0 and i < len(coords) - 1:
                    prev = coords[i-1]
                    next = coords[i+1]
                    dist_to_prev = math.sqrt((curr['x'] - prev['x'])**2 + 
                                            (curr['y'] - prev['y'])**2)
                    dist_to_next = math.sqrt((curr['x'] - next['x'])**2 + 
                                            (curr['y'] - next['y'])**2)
                    
                    if dist_to_prev > 500 and dist_to_next > 500:
                        is_outlier = True
                
                # Проверка 2: Для окна из 5 точек (если возможно)
                if not is_outlier and i >= 2 and i < len(coords) - 2:
                    # Берем окно из 5 точек
                    window = coords[i-2:i+3]
                    
                    # Считаем среднее расстояние до других точек в окне
                    total_dist = 0
                    count = 0
                    for j, other in enumerate(window):
                        if j != 2:  # Не сама точка (i в окне это индекс 2)
                            dist = math.sqrt((curr['x'] - other['x'])**2 + 
                                        (curr['y'] - other['y'])**2)
                            total_dist += dist
                            count += 1
                    
                    avg_dist = total_dist / count if count > 0 else 0
                    
                    # Если в среднем далеко от всех - выброс
                    if avg_dist > 600:
                        is_outlier = True
                
                if is_outlier:
                    outliers_to_remove.append(curr)
                    # Увеличиваем счетчик для бана
                    self.outlier_removal_count[curr_key] += 2  # +2 для частых
            
            # Старая проверка для обычных выбросов (не частых)
            elif i > 0 and i < len(coords) - 1:
                prev = coords[i-1]
                next = coords[i+1]
                
                # Расстояния между точками
                dist_to_prev = math.sqrt((curr['x'] - prev['x'])**2 + 
                                        (curr['y'] - prev['y'])**2)
                dist_to_next = math.sqrt((curr['x'] - next['x'])**2 + 
                                        (curr['y'] - next['y'])**2)
                dist_prev_next = math.sqrt((next['x'] - prev['x'])**2 + 
                                        (next['y'] - prev['y'])**2)
                
                # Старая проверка на выброс
                if dist_to_prev + dist_to_next > dist_prev_next * 2.5:
                    outliers_to_remove.append(curr)
                elif dist_to_prev > 1000 or dist_to_next > 1000:
                    if dist_prev_next < max(dist_to_prev, dist_to_next) * 0.7:
                        outliers_to_remove.append(curr)
        
        # Обрабатываем найденные выбросы
        banned_count = 0
        for outlier in outliers_to_remove:
            # Округляем до кластера 30px для группировки близких выбросов
            cluster_key = (round(outlier['x'] / 30) * 30, round(outlier['y'] / 30) * 30)

            # Увеличиваем счётчик удалений для этого кластера
            self.outlier_removal_count[cluster_key] += 1

            # Проверяем порог для бана
            if self.outlier_removal_count[cluster_key] >= self.outlier_ban_threshold:
                # Проверяем, не забанена ли уже эта зона
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

            # Удаляем из истории
            if outlier['ts'] in clean_history:
                del clean_history[outlier['ts']]
                logger.debug(f"Removed outlier at ({outlier['x']:.0f},{outlier['y']:.0f}), "
                        f"cluster={cluster_key}, count={self.outlier_removal_count[cluster_key]}")
        
        # Периодическая очистка старых записей в счётчике
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
        """Проверяет, находится ли точка в забаненной зоне."""
        for (bx, by) in self.permanent_blacklist:
            distance = math.sqrt((x - bx)**2 + (y - by)**2)
            if distance < self.ban_radius:
                logger.warning(f"⛔ BANNED: ({x:.1f}, {y:.1f}) near banned zone ({bx}, {by}), dist={distance:.0f}px")
                return True
        return False
        
    def _quick_outlier_check(self, point, window, point_idx):
        """Быстрая проверка точки на выброс."""
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
        
    def _validate_outlier_series(self, outliers, coords):
        """Проверка серии выбросов."""
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
        
    def _transfer_displayed_to_confirmed(self):
        """Переносит показанные кадры в подтверждённую историю."""
        moved = 0
        for ts in list(self.processed_future_history.keys()):
            if float(ts) <= float(self.current_display_timestamp):
                self.confirmed_history[ts] = self.processed_future_history[ts]
                del self.processed_future_history[ts]
                moved += 1
        for ts in list(self.raw_future_history.keys()):
            if float(ts) <= float(self.current_display_timestamp):
                del self.raw_future_history[ts]
                
    def _cleanup_confirmed_history(self):
        """Очистка старых подтверждённых точек."""
        if len(self.confirmed_history) <= self.max_confirmed_points:
            return
        sorted_ts = sorted(self.confirmed_history.keys())
        to_remove = len(self.confirmed_history) - self.max_confirmed_points
        for i in range(to_remove):
            del self.confirmed_history[sorted_ts[i]]
            
    def _get_context_from_confirmed(self, num_points=30):
        """Получаем последние N точек из подтверждённой истории."""
        if not self.confirmed_history:
            return {}
            
        sorted_ts = sorted(self.confirmed_history.keys())
        start_idx = max(0, len(sorted_ts) - num_points)
        
        context = {}
        for ts in sorted_ts[start_idx:]:
            context[ts] = self.confirmed_history[ts]
            
        return context
        
    def interpolate_history_gaps(self, history, fps=30, max_gap=10.0):
        """Интерполяция с поддержкой длинных полетов мяча."""
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
            
            # Пропускаем слишком маленькие промежутки
            if gap <= frame_dt * 1.5:
                continue
            
            # Интерполируем ЛЮБЫЕ разрывы до max_gap
            if gap > max_gap:
                logger.warning(f"🕳️ GAP TOO LARGE: {gap:.2f}s between ts1={ts1:.2f} and ts2={ts2:.2f}, "
                              f"pos1=({det1[6]:.0f},{det1[7]:.0f}), pos2=({det2[6]:.0f},{det2[7]:.0f})")
                continue

            # Логируем большие разрывы (возможный аут)
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
            
            # Определяем тип движения по расстоянию и времени
            dx = det2[6] - det1[6] if len(det1) > 6 else 0
            dy = det2[7] - det1[7] if len(det1) > 7 else 0
            distance = math.sqrt(dx*dx + dy*dy)

            # Если большое расстояние за короткое время - это полет
            is_flight = (gap > 0.5 and distance > 500) or gap > 1.5

            # Логируем решение о типе интерполяции для средних разрывов (только для дебага)
            # if 0.5 < gap < 5.0 and distance > 300:
            #     logger.info(f"  🎯 Interpolation decision: gap={gap:.2f}s, dist={distance:.0f}px → {'FLIGHT (parabolic)' if is_flight else 'LINEAR'}")
            
            for j in range(1, num_frames + 1):
                w_ratio = j / (num_frames + 1)
                new_ts = ts1 + j * frame_dt
                
                if is_flight:
                    # Используем вспомогательную функцию для полета
                    new_det = self._interpolate_between_points(
                        det1, det2, ts1, ts2, new_ts
                    )
                else:
                    # Простая линейная интерполяция для коротких движений
                    new_det = list(det1)
                    
                    # Интерполируем все координаты линейно
                    new_det[0] = det1[0] + (det2[0] - det1[0]) * w_ratio
                    new_det[1] = det1[1] + (det2[1] - det1[1]) * w_ratio
                    new_det[2] = det1[2] + (det2[2] - det1[2]) * w_ratio
                    new_det[3] = det1[3] + (det2[3] - det1[3]) * w_ratio
                    new_det[6] = det1[6] + (det2[6] - det1[6]) * w_ratio
                    new_det[7] = det1[7] + (det2[7] - det1[7]) * w_ratio
                    new_det[8] = det1[8] + (det2[8] - det1[8]) * w_ratio
                    new_det[9] = det1[9] + (det2[9] - det1[9]) * w_ratio
                    
                    # Уверенность снижаем для интерполированных точек
                    new_det[4] = min(det1[4], det2[4]) * 0.7
                
                interpolated[new_ts] = new_det
                added_count += 1
        
        if added_count > 0:
            logger.debug(f"Added {added_count} interpolated points")
            
        return interpolated
        
    def clean_detection_history(self, history, preserve_recent_seconds=0.5,
                               outlier_threshold=2.5, window_size=3):
        """Упрощённая дополнительная очистка."""
        if len(history) < 5:
            return history
        return history

    def get_last_confirmed_detection(self):
        """Получить последнюю confirmed детекцию (timestamp, позиция)."""
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

    def insert_backward_interpolation(self, start_ts, end_ts, start_pos, end_pos):
        """
        Вставляем синтетические детекции для плавного движения камеры.

        start_ts: время начала интерполяции (обычно end_ts - 1.0)
        end_ts: время обнаружения мяча
        start_pos: (x, y) откуда начинаем
        end_pos: (x, y) куда приезжаем
        """
        with self.history_lock:
            duration = end_ts - start_ts
            if duration <= 0:
                return

            # Генерируем 30 точек на секунду
            num_points = int(duration * 30)
            if num_points < 1:
                return

            inserted = 0
            for i in range(num_points):
                t = start_ts + (i / num_points) * duration
                alpha = i / num_points  # 0.0 → 1.0

                # Линейная интерполяция
                x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
                y = start_pos[1] + alpha * (end_pos[1] - start_pos[1])

                # Создаём синтетическую детекцию (формат как у реальной)
                synthetic_det = [
                    int(x), int(y),     # cx, cy (local)
                    15.0, 15.0,         # width, height (фиксированный размер)
                    0.5, 0,             # confidence, class_id
                    int(x), int(y),     # cx_global, cy_global
                    15.0, 15.0          # width_global, height_global
                ]

                # Вставляем в processed_future_history (чтобы попали в playback)
                self.processed_future_history[float(t)] = synthetic_det
                inserted += 1

            logger.info(f"🔄 BACKWARD INTERP: inserted {inserted} points from "
                       f"({start_pos[0]:.0f},{start_pos[1]:.0f}) to ({end_pos[0]:.0f},{end_pos[1]:.0f}) "
                       f"over {duration:.1f}s (ts {start_ts:.2f}→{end_ts:.2f})")


# =========================
# PLAYERS HISTORY (для мультикласса)
# =========================

class PlayersHistory:
    """История позиций игроков для синхронизации analysis→display."""

    def __init__(self, history_duration=10.0):
        self.history_duration = history_duration
        self.detections = {}  # {timestamp: [list of player detections]}

    def add_players(self, players_list, timestamp):
        """Сохранить список игроков для timestamp."""
        if players_list:
            self.detections[timestamp] = players_list
            self._cleanup_old(timestamp)

    def get_players_for_timestamp(self, ts):
        """Получить игроков для ближайшего timestamp."""
        if not self.detections:
            return None

        # Находим ближайший timestamp
        timestamps = list(self.detections.keys())
        closest_ts = min(timestamps, key=lambda t: abs(t - ts))

        # Если слишком старые данные
        if abs(closest_ts - ts) > 0.5:
            return None

        return self.detections[closest_ts]

    def calculate_center_of_mass(self, ts):
        """Вычисляет центр масс игроков для timestamp."""
        players = self.get_players_for_timestamp(ts)
        if not players or len(players) == 0:
            return None

        xs = [p['x'] for p in players]
        ys = [p['y'] for p in players]

        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _cleanup_old(self, current_ts):
        """Удаляет старые данные."""
        cutoff = current_ts - self.history_duration
        self.detections = {
            ts: players
            for ts, players in self.detections.items()
            if ts >= cutoff
        }


# =========================
# ОБРАБОТКА ТЕНЗОРОВ YOLO
# =========================

class TensorProcessor:
    """Постобработка YOLO-выходов."""
    
    def __init__(self, img_size=1024, conf_thresh=0.35, iou_thresh=0.45):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def postprocess_yolo_output(self, tensor_data, tile_offset=(0, 0, 1024, 1024), tile_id=0):
        """Обработка выхода YOLO."""
        try:
            # DEBUG: Логируем входные данные (TEMPORARY - только для первых тайлов)
            # if tile_id == 0:
            #     logger.info(f"[POSTPROCESS] tile_id={tile_id}, tensor_shape={tensor_data.shape}, tile_offset={tile_offset}")

            if len(tensor_data.shape) == 3:
                tensor_data = tensor_data[0]
            if tensor_data.shape[0] < tensor_data.shape[1]:
                tensor_data = tensor_data.transpose(1, 0)

            # if tile_id == 0:
            #     logger.info(f"[POSTPROCESS] After reshape: {tensor_data.shape}")

            if tensor_data.shape[1] < 9:
                # if tile_id == 0:
                #     logger.info(f"[POSTPROCESS] SKIP: shape[1]={tensor_data.shape[1]} < 9")
                return []

            # ========== MULTICLASS SUPPORT ==========
            # Извлекаем bbox и класс-скоры отдельно
            bbox_data = tensor_data[:, :4]  # x, y, w, h
            class_scores = tensor_data[:, 4:9]  # 5 классов: ball, player, staff, side_ref, main_ref

            # Для каждой детекции находим лучший класс
            class_ids = np.argmax(class_scores, axis=1)  # (21504,) - class_id 0-4
            confidences = np.max(class_scores, axis=1)   # (21504,) - max confidence

            # Фильтр по confidence
            mask = confidences > self.conf_thresh
            # if tile_id == 0:
            #     max_conf = np.max(confidences) if len(confidences) > 0 else 0.0
            #     logger.info(f"[POSTPROCESS] Confidence filter: {np.sum(mask)}/{len(mask)} passed (thresh={self.conf_thresh}), MAX_CONF={max_conf:.4f}")

            if not np.any(mask):
                return []

            # Применяем маску
            x = bbox_data[mask, 0]
            y = bbox_data[mask, 1]
            w = bbox_data[mask, 2]
            h = bbox_data[mask, 3]
            s = confidences[mask]
            cls_id = class_ids[mask]
            
            # Фильтрация по размеру
            size_mask = (w >= 8) & (h >= 8) & (w <= 120) & (h <= 120)
            if not np.any(size_mask):
                return []

            x = x[size_mask]
            y = y[size_mask]
            w = w[size_mask]
            h = h[size_mask]
            s = s[size_mask]
            cls_id = cls_id[size_mask]

            # Отбрасываем боксы у краёв
            edge = 20
            x1 = x - 0.5 * w
            y1 = y - 0.5 * h
            x2 = x + 0.5 * w
            y2 = y + 0.5 * h
            inb = (x1 >= edge) & (y1 >= edge) & (x2 <= (self.img_size - edge)) & (y2 <= (self.img_size - edge))
            if not np.any(inb):
                return []

            x = x[inb]
            y = y[inb]
            w = w[inb]
            h = h[inb]
            s = s[inb]
            cls_id = cls_id[inb]
            
            # Переводим в глобальные координаты
            off_x, off_y, tile_w, tile_h = tile_offset
            out = []
            for i in range(len(s)):
                cx_local = float(x[i])
                cy_local = float(y[i])
                cx_g = cx_local + float(off_x)
                cy_g = cy_local + float(off_y)

                out.append({
                    'x': cx_g,
                    'y': cy_g,
                    'width': float(w[i]),
                    'height': float(h[i]),
                    'confidence': float(s[i]),
                    'class_id': int(cls_id[i]),  # 0=ball, 1=player, 2=staff, 3=side_ref, 4=main_ref
                    'tile_id': int(tile_id)
                })
            return out
        except Exception as e:
            logger.error(f"postprocess error: {e}")
            return []


def get_tensor_as_numpy(layer_info):
    """Извлекаем numpy-массив из NvDsInferLayerInfo."""
    try:
        data_ptr = pyds.get_ptr(layer_info.buffer)
        dims = [layer_info.inferDims.d[i] for i in range(layer_info.inferDims.numDims)]
        
        if layer_info.dataType == 0:
            ctype_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float))
            np_dtype = np.float32
        elif layer_info.dataType == 1:
            ctype_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint16))
            np_dtype = np.float16
        elif layer_info.dataType == 2:
            ctype_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_int32))
            np_dtype = np.int32
        elif layer_info.dataType == 3:
            ctype_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_int8))
            np_dtype = np.int8
        else:
            raise TypeError(f"Unsupported dataType: {layer_info.dataType}")
            
        size = int(np.prod(dims))
        array = np.ctypeslib.as_array(ctype_ptr, shape=(size,)).copy()
        if np_dtype != np.float32:
            array = array.astype(np.float32)
        return array.reshape(dims)
    except Exception as e:
        logger.error(f"get_tensor_as_numpy: {e}")
        return np.array([])


# =========================
# NMS (Non-Maximum Suppression)
# =========================

def apply_nms(detections, iou_threshold=0.5):
    """
    Применяет Non-Maximum Suppression для удаления дублирующихся боксов.

    Args:
        detections: список словарей с ключами 'x', 'y', 'width', 'height', 'confidence'
        iou_threshold: порог IoU для подавления (default: 0.5)

    Returns:
        список детекций после NMS
    """
    if not detections:
        return []

    # Преобразуем в массивы для быстрой обработки
    boxes = []
    scores = []
    for d in detections:
        cx, cy, w, h = d['x'], d['y'], d['width'], d['height']
        # Конвертируем из center-format в corner-format
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(d['confidence'])

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Сортируем по confidence (descending)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Берём бокс с максимальной уверенностью
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Вычисляем IoU с остальными боксами
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])

        iou = inter / (area_i + area_others - inter)

        # Оставляем только те, у которых IoU < threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    # Возвращаем отфильтрованные детекции
    return [detections[i] for i in keep]


# =========================
# ОСНОВНОЙ КЛАСС
# =========================

class PanoramaWithVirtualCamera:
    """Панорама с двумя режимами отображения, записью и единой буферизацией."""
    
    def __init__(self,
                # Источники видео
                source_type: str = "files",
                video1: str = "left1.mp4",
                video2: str = "right1.mp4",
                config_path: str = None,
                buffer_duration: float = 5.0,
                enable_display: bool = True,
                display_mode: str = "panorama",  # "panorama", "virtualcam", "stream", "record"
                enable_analysis: bool = True,
                analysis_skip_interval: int = 5,
                confidence_threshold: float = 0.35,
                auto_zoom: bool = True,
                stream_key: str = None,
                stream_url: str = None,
                output_file: str = None,
                bitrate: int = 6000000):

        self.source_type = source_type
        self.video1 = video1
        self.video2 = video2
        self.buffer_duration = float(buffer_duration)
        self.enable_display = enable_display
        self.display_mode = display_mode
        self.enable_analysis = enable_analysis
        self.confidence_threshold = confidence_threshold
        self.auto_zoom = auto_zoom
        self.stream_key = stream_key
        self.stream_url = stream_url
        self.output_file = output_file
        self.bitrate = bitrate




        # Используем глобальные константы панорамы
        self.panorama_width = PANORAMA_WIDTH
        self.panorama_height = PANORAMA_HEIGHT

        # ROI конфигурация - используем предрассчитанные координаты тайлов
        # ВАЖНО: Координаты автоматически вычисляются из TILE_POSITIONS
        self.roi_configs = TILE_POSITIONS

        # Загружаем маску поля (добавить после инициализации history)
        self.field_mask = FieldMaskBinary(
        mask_path='field_mask.png',
        panorama_width=self.panorama_width,
        panorama_height=self.panorama_height
        )
        
        # История и фильтрация
        self.history = BallDetectionHistory(history_duration=10.0, cleanup_interval=1000)
        self.players_history = PlayersHistory(history_duration=10.0)  # История игроков для fallback

        self.tensor_processor = TensorProcessor(conf_thresh=confidence_threshold)
        
        # Для адаптивного фильтра
        self.last_ball_position = None
        self.frames_without_reliable_detection = 0

        # Хранилище всех детекций для отрисовки (синхронизация по timestamp)
        self.all_detections_history = {}  # {timestamp: {'ball': [...], 'player': [...], ...}}

        # EMA сглаживание для центра масс игроков
        self.players_center_mass_smoothed = None  # (x, y) - сглаженная позиция
        self.players_center_mass_alpha = 0.18  # Коэффициент сглаживания (меньше = плавнее, для буфера можем 0.15-0.2)

        # Буфер сырых позиций для обнаружения паттернов "туда-обратно"
        self.players_center_mass_history = []  # [(x, y), ...] последние 10 сырых позиций
        self.players_center_mass_history_max = 10
        
        # Статистика
        self.display_frame_count = 0
        self.analysis_frame_count = 0
        self.analysis_skip_counter = 0
        self.analysis_skip_interval = max(1, int(analysis_skip_interval))
        self.analysis_actual_frame = 0
        self.detection_count = 0
        self.start_time = None
        self.current_fps = 0.0

        # Timestamp для backward interpolation
        self.current_display_timestamp = 0.0  # Текущий timestamp playback

        # Параметры отображения
        self.framerate = 30

        self.buffer_lock = threading.RLock()
        
        # Пайплайны
        self.pipeline = None
        self.loop = GLib.MainLoop()
        
        # Потоки
        self.buffer_thread = None
        self.buffer_thread_running = False
        
        # Конфиг nvinfer
        self.config_path = config_path or self.create_inference_config()
        self.speed_zoom_enabled = True
        self.speed_history = deque(maxlen=5)  # История скоростей для сглаживания
        self.last_speed_calc_time = 0
        self.last_speed_calc_pos = None
        self.current_smooth_speed = 0.0
        self.speed_zoom_factor = 1.6

        # Интерполяция размера мяча для плавного зума
        self.smooth_ball_radius = 20.0  # Начальное значение
        self.radius_smooth_factor = 0.3  # Коэффициент сглаживания (0.3 = 30% нового значения)

        # Параметры поведения при потере мяча
        self.ball_lost = False
        self.ball_lost_frames = 0
        self.last_known_position = None  # (x, y, timestamp)
        self.lost_ball_fov_rate = 2.0    # Градусов в секунду увеличения FOV
        self.max_search_fov = 90.0       # Максимальный FOV при поиске
        self.ball_recovery_frames = 6   # Кадров для подтверждения восстановления
        
        # Настройки порогов скорости (пиксели/сек)
        self.speed_low_threshold = 300.0    # Начало реакции на скорость (было 400)
        self.speed_high_threshold = 1200.0  # Максимальная реакция (было 1500)
        self.speed_zoom_max_factor = 3.0    # Максимальное увеличение радиуса (тест: было 1.6→2.0, теперь 3.0x очень агрессивно)
        self.speed_smoothing = 0.3          # Коэффициент сглаживания скорости

        for log_file in ['ball_events.tsv', 'ball_raw_future.csv', 'ball_display_used.csv']:
            if os.path.exists(log_file):
                os.remove(log_file)
                logger.info(f"Удален старый лог: {log_file}")
        
    def create_inference_config(self, output_path="config_infer.txt"):
        """Создание конфига для YOLO (только если файл отсутствует или некорректный)."""

        # Список обязательных полей для валидации
        required_fields = [
            'gpu-id',
            'model-engine-file',
            'batch-size',
            'network-mode',
            'num-detected-classes',
            'network-type',
            'output-blob-names',
            'pre-cluster-threshold',
            'nms-iou-threshold'
        ]

        # Проверка существования и валидности конфига
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    content = f.read()

                # Проверка, что файл не пустой
                if len(content.strip()) == 0:
                    logger.warning(f"⚠️ Конфиг {output_path} пустой, будет пересоздан")
                else:
                    # Проверка наличия всех обязательных полей
                    missing_fields = []
                    for field in required_fields:
                        if field not in content:
                            missing_fields.append(field)

                    if missing_fields:
                        logger.warning(f"⚠️ Конфиг {output_path} неполный (отсутствуют: {', '.join(missing_fields)}), будет пересоздан")
                    else:
                        # Проверка наличия секций [property] и [class-attrs-all]
                        if '[property]' not in content or '[class-attrs-all]' not in content:
                            logger.warning(f"⚠️ Конфиг {output_path} без необходимых секций, будет пересоздан")
                        else:
                            # Конфиг валидный, используем существующий
                            logger.info(f"✅ Используется существующий конфиг: {output_path}")
                            return output_path

            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения конфига {output_path}: {e}, будет пересоздан")

        # Создаём новый конфиг (если не существует или невалидный)
        config = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=yolo11n_mixed_finetune_v7_int8.engine
batch-size=6
network-mode=2
num-detected-classes=1
interval=1
gie-unique-id=1
process-mode=1
network-type=100
maintain-aspect-ratio=1
symmetric-padding=1
output-blob-names=output0
output-tensor-meta=1

[class-attrs-all]
pre-cluster-threshold=0.25
topk=100
nms-iou-threshold=0.45
"""
        with open(output_path, 'w') as f:
            f.write(config)
        with open("labels.txt", "w") as f:
            f.write("ball\n")
        logger.info(f"✅ Создан новый конфиг nvinfer: {output_path}")
        return output_path

    def find_usb_audio_device(self):
        """Проверка доступности аудио захвата через PulseAudio."""
        try:
            # Используем PulseAudio вместо прямого доступа к hw:0,0
            test_pipe = """
                pulsesrc !
                audioconvert !
                audio/x-raw,format=S16LE,rate=44100,channels=2 !
                fakesink
            """
            test = Gst.parse_launch(test_pipe)
            test.set_state(Gst.State.PLAYING)
            time.sleep(0.2)
            state = test.get_state(0.1)
            test.set_state(Gst.State.NULL)

            if state[0] == Gst.StateChangeReturn.SUCCESS:
                self.audio_device = "pulse"  # Изменено с hw:0,0 на pulse
                logger.info("🎤 Микрофон готов через PulseAudio")
                return True

            logger.warning("⚠️ Микрофон не доступен")
            self.audio_device = None
            return False

        except Exception as e:
            logger.error(f"Ошибка проверки микрофона: {e}")
            self.audio_device = None
            return False

    # ========================================================================
    # МОДУЛЬНЫЕ ФУНКЦИИ DISPLAY BRANCH'ЕЙ (для каждого режима)
    # ========================================================================

    def _get_display_branch_base(self, buffer_size: int, buffer_time_ns: int, buffer_frames: int) -> str:
        """
        Общая часть display branch для всех режимов.
        Возвращает pipeline string от main_tee до ring buffer.
        """
        frame_size_bytes = 5700 * 1900 * 4  # 43,320,000 bytes per frame (panorama)
        ring_buffer_size = buffer_frames * frame_size_bytes

        return f"""
            main_tee. !
            queue name=display_queue
                max-size-buffers={buffer_size}
                max-size-time={buffer_time_ns}
                leaky=0 !
            nvvideoconvert name=display_convert compute-hw=1 !
            capsfilter caps="video/x-raw(memory:NVMM),format=RGBA" !
            nvdsringbuf
                ring-bytes={ring_buffer_size}
                min-slots={buffer_frames}
                chunk=1 !
        """

    def _create_display_panorama(self, buffer_size: int, buffer_time_ns: int, buffer_frames: int) -> str:
        """
        Panorama display branch: ring buffer → nvdsosd → nveglglessink
        Полная панорама (5700×1900) с отрисовкой детекций
        """
        base = self._get_display_branch_base(buffer_size, buffer_time_ns, buffer_frames)

        pipeline_str = base + """
            nvdsosd name=draw_osd process-mode=0 !
            nvvideoconvert name=display_convert_out compute-hw=1 nvbuf-memory-type=0 !
            nveglglessink name=display_sink sync=false async=false enable-last-sample=false
        """

        logger.info("🎬 Режим панорамы: вывод в окно (5700×1900)")
        return pipeline_str

    def _create_display_virtualcam(self, buffer_size: int, buffer_time_ns: int, buffer_frames: int) -> str:
        """
        Virtual Camera display branch: ring buffer → nvdsvirtualcam → xvimagesink
        Динамическая камера (1920×1080) следящая за мячом
        """
        base = self._get_display_branch_base(buffer_size, buffer_time_ns, buffer_frames)

        pipeline_str = base + f"""
            nvdsvirtualcam name=vcam
                output-width=1920
                output-height=1080
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT}
                yaw=0 pitch=15 roll=0 fov=65
                auto-follow=true
                smooth-factor=0.15 !
            nvvideoconvert !
            video/x-raw,format=RGBA !
            videoconvert !
            xvimagesink name=vcam_sink sync=false
        """

        logger.info("📹 Режим виртуальной камеры: вывод в окно (1920×1080)")
        return pipeline_str

    def _create_display_stream(self, buffer_size: int, buffer_time_ns: int, buffer_frames: int) -> str:
        """
        Stream display branch: ring buffer → nvdsvirtualcam → H.264 → RTMP + опциональная запись
        Стриминг на YouTube Live с опциональной локальной записью
        """
        base = self._get_display_branch_base(buffer_size, buffer_time_ns, buffer_frames)

        bitrate_bps = self.bitrate

        # Основная часть: nvdsvirtualcam + H.264
        pipeline_str = base + f"""
            nvdsvirtualcam name=vcam
                output-width=1920
                output-height=1080
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT}
                yaw=0 pitch=10 roll=0 fov=65
                auto-follow=true
                smooth-factor=0.15 !
            video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080 !
            nvvideoconvert compute-hw=1 !
            video/x-raw(memory:NVMM),format=NV12 !
            nvv4l2h264enc
                bitrate={bitrate_bps}
                preset-level=2
                insert-sps-pps=1
                iframeinterval=50
                maxperf-enable=true !
            h264parse !
        """

        # Если нужна запись - добавляем tee
        if self.output_file:
            pipeline_str += f"""
            tee name=t !
            queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
            flvmux name=flvmux streamable=true !
            rtmpsink
                location={self.stream_url}{self.stream_key}
                sync=false
                async=false

            t. !
            queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
            flvmux streamable=true !
            filesink location={self.output_file} sync=false async=false
            """
            logger.info(f"💾 Запись в FLV включена: {self.output_file}")
        else:
            # Только стриминг без записи
            pipeline_str += f"""
            queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
            flvmux name=flvmux streamable=true !
            rtmpsink
                location={self.stream_url}{self.stream_key}
                sync=false
                async=false
            """

        # Аудио (тишина - микрофон не используется)
        pipeline_str += """
            audiotestsrc wave=silence is-live=true !
            audio/x-raw,rate=44100,channels=2 !
            audioconvert !
            voaacenc bitrate=128000 !
            aacparse !
            queue !
            flvmux.
        """

        bitrate_mbps = bitrate_bps / 1000000.0
        logger.info(f"🔴 Стриминг на {self.stream_url}")
        logger.info(f"⚡ H.264 bitrate: {bitrate_mbps:.1f} Mbps")

        return pipeline_str

    def _create_display_record(self, buffer_size: int, buffer_time_ns: int, buffer_frames: int) -> str:
        """
        Record display branch: ring buffer → nvdsvirtualcam → H.264 → файл
        Запись видео в файл (FLV/MP4/MKV) без окна
        """
        base = self._get_display_branch_base(buffer_size, buffer_time_ns, buffer_frames)

        bitrate_bps = self.bitrate

        # Выбор формата по расширению файла
        use_flv = self.output_file.endswith('.flv')
        use_mp4 = self.output_file.endswith('.mp4')

        if use_flv:
            muxer = "flvmux streamable=true"
            format_name = "FLV (рекомендуется, как у YouTube)"
        elif use_mp4:
            muxer = "mp4mux"
            format_name = "MP4"
        else:
            muxer = 'matroskamux streamable=false writing-app="DeepStream Football Tracker"'
            format_name = "Matroska (MKV)"

        pipeline_str = base + f"""
            nvdsvirtualcam name=vcam
                output-width=1920
                output-height=1080
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT}
                yaw=0 pitch=10 roll=0 fov=65
                auto-follow=true
                smooth-factor=0.15 !
            video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080 !
            nvvideoconvert compute-hw=1 !
            video/x-raw(memory:NVMM),format=NV12 !
            nvv4l2h264enc
                bitrate={bitrate_bps}
                preset-level=2
                insert-sps-pps=1
                iframeinterval=50
                maxperf-enable=true !
            h264parse !
            queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
            {muxer} !
            filesink location={self.output_file} sync=false async=false
        """

        bitrate_mbps = bitrate_bps / 1000000.0
        logger.info(f"💾 Режим записи: {self.output_file}")
        logger.info(f"⚡ Параметры: bitrate={bitrate_mbps:.1f}Mbps, preset=2, iframe=50")
        logger.info(f"📦 Формат: {format_name}")

        return pipeline_str

    def create_pipeline(self) -> bool:
        """Создание основного pipeline с поддержкой камер и файлов."""
        try:
            buffer_size = int(self.framerate * self.buffer_duration)
            buffer_time_ns = int(self.buffer_duration * 1e9)
            
            # Определяем источники в зависимости от типа
            if self.source_type == "cameras":
                # Используем nvarguscamerasrc для камер
                left_cam = int(self.video1)
                right_cam = int(self.video2)
                
                logger.info(f"📷 Используем камеры: левая={left_cam}, правая={right_cam}")
                
                sources_str = f"""
                    nvarguscamerasrc sensor-id={left_cam} !
                    video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_0

                    nvarguscamerasrc sensor-id={right_cam} !
                    video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_1
                """
                
                # Для камер используем live-source=1
                mux_config = """
                    nvstreammux name=mux
                        batch-size=2
                        width=3840
                        height=2160
                        live-source=1
                        batched-push-timeout=33333 !
                """
            else:
                # Используем filesrc для файлов
                logger.info(f"📁 Используем файлы: {self.video1}, {self.video2}")
                
                sources_str = f"""
                    filesrc location={self.video1} !
                    decodebin !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_0

                    filesrc location={self.video2} !
                    decodebin !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_1
                """
                
                # Для файлов используем live-source=0
                mux_config = """
                    nvstreammux name=mux
                        batch-size=2
                        width=3840
                        height=2160
                        live-source=0
                        batched-push-timeout=40000 !
                """
            
            # Общая часть pipeline
            common_str = f"""
                nvdsstitch
                    left-source-id=0
                    right-source-id=1
                    gpu-id=0
                    use-egl=true
                    panorama-width={PANORAMA_WIDTH}
                    panorama-height={PANORAMA_HEIGHT} !

                tee name=main_tee
            """
            
            # Базовый pipeline
            pipeline_str = sources_str + mux_config + common_str

            # ========================================================================
            # ВЫБОР DISPLAY BRANCH В ЗАВИСИМОСТИ ОТ РЕЖИМА
            # ========================================================================
            if self.enable_display:
                buffer_frames = int(self.buffer_duration * self.framerate)

                # Выбираем нужный режим и получаем соответствующий pipeline string
                if self.display_mode == "panorama":
                    pipeline_str += self._create_display_panorama(buffer_size, buffer_time_ns, buffer_frames)

                elif self.display_mode == "virtualcam":
                    pipeline_str += self._create_display_virtualcam(buffer_size, buffer_time_ns, buffer_frames)

                elif self.display_mode == "stream":
                    # Для stream требуется --stream-url и --stream-key
                    if not self.stream_url or not self.stream_key:
                        logger.error("❌ Для режима stream требуются --stream-url и --stream-key")
                        return False
                    pipeline_str += self._create_display_stream(buffer_size, buffer_time_ns, buffer_frames)

                elif self.display_mode == "record":
                    # Для record требуется --output
                    if not self.output_file:
                        logger.error("❌ Для режима record требуется --output <файл>")
                        return False
                    pipeline_str += self._create_display_record(buffer_size, buffer_time_ns, buffer_frames)

                else:
                    logger.error(f"❌ Неизвестный режим: {self.display_mode}")
                    return False
                
            # Ветка анализа
            if self.enable_analysis:
                pipeline_str += """
                    main_tee. !
                    queue name=analysis_queue max-size-buffers=2 leaky=downstream !
                    tee name=tiles_tee
                """
                
            logger.info(f"Создаём основной pipeline для режима: {self.display_mode}, источник: {self.source_type}")
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # nveglglessink отображает окно напрямую - callback'ы не нужны

            # Подключаем draw_probe для режима панорамы
            if self.enable_display and self.display_mode == "panorama":
                draw_osd = self.pipeline.get_by_name("draw_osd")
                if draw_osd:
                    osd_sink_pad = draw_osd.get_static_pad("sink")
                    if osd_sink_pad:
                        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.draw_probe, 0)
                        logger.info("✅ draw_probe подключен к nvdsosd")

            # Создаем тайлы для анализа если нужно
            if self.enable_analysis:
                self._create_analysis_tiles()
                
            logger.info("✅ Основной pipeline создан успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка create_pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

            
    def _create_analysis_tiles(self):
        """Создание 6 тайлов для анализа."""
        tiles_tee = self.pipeline.get_by_name("tiles_tee")
        if not tiles_tee:
            logger.error("tiles_tee не найден")
            return
            
        # Identity для пропуска кадров
        frame_filter = Gst.ElementFactory.make("identity", "frame-filter")
        frame_filter.set_property("sync", False)
        self.pipeline.add(frame_filter)
        
        tee_src = tiles_tee.request_pad_simple("src_%u")
        filter_sink = frame_filter.get_static_pad("sink")
        tee_src.link(filter_sink)
        
        filter_src = frame_filter.get_static_pad("src")
        filter_src.add_probe(Gst.PadProbeType.BUFFER, self.frame_skip_probe, 0)
        logger.info(f"Добавлен frame_skip_probe (каждый {self.analysis_skip_interval}-й кадр)")

        # ============================================================
        # НОВЫЙ КОД: nvtilebatcher вместо filtered_tee + 6×crop + mux
        # ============================================================

        # Создаем nvtilebatcher плагин
        tilebatcher = Gst.ElementFactory.make("nvtilebatcher", "tilebatcher")
        if not tilebatcher:
            logger.error("❌ nvtilebatcher плагин не найден!")
            logger.error("Установите плагин: cd /home/nvidia/deep_cv_football/my_tile_batcher/src && make install")
            return

        # Настройки плагина
        tilebatcher.set_property("gpu-id", 0)
        tilebatcher.set_property("panorama-width", PANORAMA_WIDTH)
        tilebatcher.set_property("panorama-height", PANORAMA_HEIGHT)
        tilebatcher.set_property("tile-offset-y", TILE_OFFSET_Y)  # Вертикальный offset из field_mask.png
        # 6 тайлов БЕЗ ПРОПУСКОВ, вырезаются на основе field_mask.png
        # Y позиция: не симметричное центрирование, а рассчитанное из маски поля!

        self.pipeline.add(tilebatcher)

        # Связываем: frame_filter → tilebatcher
        frame_filter.link(tilebatcher)

        logger.info(f"✅ nvtilebatcher создан ({TILES_COUNT} тайлов БЕЗ пропусков из центра)")
        logger.info(f"   Координаты тайлов (отступ по бокам {TILE_OFFSET_X}px, вертикальный {TILE_OFFSET_Y}px):")
        for tile_id, (x, y, w, h) in enumerate(TILE_POSITIONS):
            logger.info(f"   Тайл {tile_id}: x={x}, y={y}, size={w}×{h}")

        # ============================================================
        # nvinfer напрямую после tilebatcher (БЕЗ nvstreammux!)
        # ============================================================

        pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
        pgie.set_property("config-file-path", self.config_path)
        pgie.set_property("batch-size", 6)  # ВАЖНО: должно соответствовать TILES_PER_BATCH
        pgie.set_property("gpu-id", 0)
        self.pipeline.add(pgie)

        # Связываем: tilebatcher → nvinfer
        tilebatcher.link(pgie)

        logger.info("✅ nvinfer подключен после nvtilebatcher")
        
        # fakesink
        sink_inf = Gst.ElementFactory.make("fakesink", "sink-infer")
        sink_inf.set_property("sync", False)
        sink_inf.set_property("async", False)
        self.pipeline.add(sink_inf)
        pgie.link(sink_inf)
        
        # Probe после инференса
        pgie_src = pgie.get_static_pad("src")
        if pgie_src:
            pgie_src.add_probe(Gst.PadProbeType.BUFFER, self.analysis_probe, 0)
            logger.info("Добавлен analysis_probe")
            
            
    def frame_skip_probe(self, pad, info, u_data):
        """Пропуск кадров для анализа."""
        self.analysis_skip_counter += 1
        if self.analysis_skip_counter % self.analysis_skip_interval != 0:
            return Gst.PadProbeReturn.DROP
        self.analysis_actual_frame += 1
        return Gst.PadProbeReturn.OK
        
    def analysis_probe(self, pad, info, u_data):
        """Обработка YOLO с фильтрацией."""
        try:
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK
                
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
                
            self.analysis_frame_count = self.analysis_actual_frame * self.analysis_skip_interval
            
            # Собираем детекции
            per_ts = defaultdict(list)
            per_ts_fnum = {}

            # Дебаг: считаем количество тайлов и тензоров
            tiles_processed = []
            tensor_found_tiles = []

            # ВАЖНО: tile_id считаем вручную, т.к. pad_index ВСЕГДА 0!
            tile_counter = 0

            l_frame = batch_meta.frame_meta_list
            while l_frame:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                if not fm:
                    l_frame = l_frame.next
                    continue

                # Используем счетчик вместо pad_index!
                tile_id = tile_counter
                tile_counter += 1

                tiles_processed.append(tile_id)
                frame_num = fm.frame_num
                ts_sec = float(fm.buf_pts) / float(Gst.SECOND)

                l_user = fm.frame_user_meta_list
                while l_user:
                    um = pyds.NvDsUserMeta.cast(l_user.data)
                    if um and um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tensor_found_tiles.append(tile_id)
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                        for i in range(tensor_meta.num_output_layers):
                            layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                            td = get_tensor_as_numpy(layer)
                            if td.size == 0:
                                continue

                            if tile_id < len(self.roi_configs):
                                tile_off = self.roi_configs[tile_id]
                            else:
                                tile_off = (0, 0, 1024, 1024)

                            dets = self.tensor_processor.postprocess_yolo_output(td, tile_off, tile_id)

                            if dets:
                                per_ts[ts_sec].extend(dets)
                                per_ts_fnum[ts_sec] = frame_num
                                self.detection_count += len(dets)
                                
                    l_user = l_user.next
                l_frame = l_frame.next
                
            # Дебаг лог (каждые 10 кадров)
            if self.analysis_frame_count % 10 == 0:
                logger.info(f"🔍 Tiles: processed={tiles_processed}, tensor_found={tensor_found_tiles}")

            # Обрабатываем детекции
            for ts, det_list in per_ts.items():
                if not det_list:
                    continue

                # ========== MULTICLASS: Разделяем по классам ==========
                ball_detections = [d for d in det_list if d.get('class_id', 0) == 0]
                player_detections = [d for d in det_list if d.get('class_id', 0) == 1]
                staff_detections = [d for d in det_list if d.get('class_id', 0) == 2]
                side_ref_detections = [d for d in det_list if d.get('class_id', 0) == 3]
                main_ref_detections = [d for d in det_list if d.get('class_id', 0) == 4]

                # Логируем детальную статистику по всем классам (каждые 10 кадров)
                # if self.analysis_frame_count % 10 == 0:
                #     logger.info(f"📊 RAW Detections per class:")
                #     logger.info(f"   🔴 Ball: {len(ball_detections)}")
                #     logger.info(f"   🟢 Players: {len(player_detections)}")
                #     logger.info(f"   🟡 Staff: {len(staff_detections)}")
                #     logger.info(f"   🔵 Side Refs: {len(side_ref_detections)}")
                #     logger.info(f"   🟣 Main Refs: {len(main_ref_detections)}")
                #     logger.info(f"   📦 TOTAL: {len(det_list)}")

                # Обрабатываем игроков (простая фильтрация + NMS, сохраняем в историю)
                valid_players = []
                filtered_players = []
                if player_detections:
                    # Простые фильтры для игроков
                    filtered_players = [p for p in player_detections
                                       if p['confidence'] >= 0.45  # Повышенный порог для фильтрации ложных детекций
                                       and self.field_mask.is_inside_field(p['x'], p['y'])
                                       and 5 <= p['width'] <= 250
                                       and 20 <= p['height'] <= 350]  # Увеличен для высоких игроков

                    # Применяем NMS для удаления дублирующихся детекций
                    if filtered_players:
                        valid_players = apply_nms(filtered_players, iou_threshold=0.5)

                    if valid_players:
                        self.players_history.add_players(valid_players, ts)

                # Сохраняем ВСЕ валидные детекции для отрисовки
                if ts not in self.all_detections_history:
                    self.all_detections_history[ts] = {'ball': [], 'player': [], 'staff': [], 'referee': []}

                # Добавляем игроков (если были валидные)
                if valid_players:
                    self.all_detections_history[ts]['player'] = valid_players

                # Добавляем staff (class_id=2) - не добавляем в историю, т.к. не рисуем
                # Добавляем referees (class_id=3,4) - не добавляем в историю, т.к. не рисуем

                # Логируем статистику после фильтрации (каждые 10 кадров)
                if self.analysis_frame_count % 10 == 0:
                    logger.info(f"✅ After filters (for drawing):")
                    logger.info(f"   🟢 Players: {len(player_detections)} raw → {len(filtered_players)} filtered → {len(valid_players)} after NMS")
                    # Мяч будет залогирован ниже после своих фильтров

                # Продолжаем обработку МЯЧА с текущими фильтрами
                det_list = ball_detections
                count_initial = len(det_list)

                # Показываем с каких тайлов детекции (каждые 10 кадров)
                if self.analysis_frame_count % 10 == 0 and det_list:
                    tiles_count = {}
                    # for d in det_list:
                    #     tile_id = d.get('tile_id', '?')
                    #     tiles_count[tile_id] = tiles_count.get(tile_id, 0) + 1
                    # logger.info(f"📍 BALL detections by tile: {dict(sorted(tiles_count.items()))}")

                # 1. Фильтр по уверенности
                det_list = [d for d in det_list if d['confidence'] >= self.confidence_threshold]
                count_after_confidence = len(det_list)
                if not det_list:
                    if self.analysis_frame_count % 10 == 0:
                        logger.info(f"  ❌ Filter 1 (confidence): {count_initial} → 0 (threshold={self.confidence_threshold})")
                    continue

                # 2. Фильтр по маске поля
                det_list_before_field = det_list.copy()
                det_list = [d for d in det_list
                            if self.field_mask.is_inside_field(d['x'], d['y'])]
                count_after_field = len(det_list)
                if not det_list:
                    if self.analysis_frame_count % 10 == 0:
                        # Показываем примеры отброшенных детекций
                        examples = [f"({d['x']:.0f},{d['y']:.0f},tile={d.get('tile_id','?')})" for d in det_list_before_field[:3]]
                        logger.info(f"  ❌ Filter 2 (field mask): {count_after_confidence} → 0. Examples: {examples}")
                    continue

                # 3. Фильтр permanent ban (выбросы траектории)
                filtered_from_ban = []
                banned_examples = []
                for d in det_list:
                    if not self.history.is_point_banned(d['x'], d['y']):
                        filtered_from_ban.append(d)
                    else:
                        banned_examples.append(f"({d['x']:.0f},{d['y']:.0f})")
                det_list = filtered_from_ban
                count_after_ban = len(det_list)

                if not det_list:
                    if self.analysis_frame_count % 10 == 0:
                        logger.info(f"  ❌ Filter 3 (banned zones): {count_after_field} → 0. Banned: {banned_examples[:3]}")
                    continue

                # 4. Фильтр по форме (близко к квадрату)
                shape_rejected = []
                valid_dets = []
                for d in det_list:
                    ratio = d['width'] / (d['height'] + 0.001)
                    if 0.7 <= ratio <= 1.3:
                        valid_dets.append(d)
                    else:
                        shape_rejected.append(f"({d['x']:.0f},{d['y']:.0f},w={d['width']:.0f},h={d['height']:.0f},ratio={ratio:.2f})")
                count_after_shape = len(valid_dets)

                if not valid_dets:
                    if self.analysis_frame_count % 10 == 0:
                        logger.info(f"  ❌ Filter 4 (shape ratio): {count_after_ban} → 0. Rejected: {shape_rejected[:3]}")
                    self.frames_without_reliable_detection += 1
                    continue

                # 5. Адаптивный фильтр по расстоянию
                count_before_distance = len(valid_dets)
                if self.last_ball_position and valid_dets:
                    valid_dets = self._apply_adaptive_distance_filter(
                        valid_dets,
                        self.last_ball_position,
                        self.frames_without_reliable_detection
                    )
                count_after_distance = len(valid_dets)

                if not valid_dets:
                    if self.analysis_frame_count % 10 == 0:
                        logger.info(f"  ❌ Filter 5 (distance): {count_before_distance} → 0 (last_pos={self.last_ball_position})")
                    self.frames_without_reliable_detection += 1
                    continue

                # Логируем успешное прохождение всех фильтров
                if self.analysis_frame_count % 10 == 0:
                    logger.info(f"  ✅ Filters passed: {count_initial} → conf:{count_after_confidence} → field:{count_after_field} → ban:{count_after_ban} → shape:{count_after_shape} → dist:{count_after_distance}")

                # ========== ПРИОРИТЕТ МЯЧА ПО БЛИЗОСТИ К ИГРОКАМ ==========
                # Если детекций >= 2, выбираем мяч ближайший к центру масс игроков
                # даже если его confidence ниже
                best = None
                if len(valid_dets) >= 2:
                    # Вычисляем центр масс игроков
                    players_center = self.players_history.calculate_center_of_mass(ts)

                    if players_center:
                        # Сортируем по расстоянию до центра масс игроков
                        def distance_to_players(det):
                            dx = det['x'] - players_center[0]
                            dy = det['y'] - players_center[1]
                            return (dx*dx + dy*dy) ** 0.5

                        closest_to_players = min(valid_dets, key=distance_to_players)
                        highest_conf = max(valid_dets, key=lambda d: d['confidence'])

                        # Логируем выбор
                        dist_closest = distance_to_players(closest_to_players)
                        dist_highest = distance_to_players(highest_conf)

                        if self.analysis_frame_count % 10 == 0:
                            logger.info(f"  🎯 Ball priority: {len(valid_dets)} candidates")
                            logger.info(f"     Closest to players: conf={closest_to_players['confidence']:.3f}, dist={dist_closest:.0f}px")
                            logger.info(f"     Highest confidence: conf={highest_conf['confidence']:.3f}, dist={dist_highest:.0f}px")

                        # Выбираем ближайший к игрокам
                        best = closest_to_players

                        if closest_to_players != highest_conf and self.analysis_frame_count % 10 == 0:
                            logger.info(f"  ✨ Picked ball closer to players (Δconf={highest_conf['confidence'] - closest_to_players['confidence']:.3f})")
                    else:
                        # Нет данных об игроках - выбираем по confidence
                        best = max(valid_dets, key=lambda d: d['confidence'])
                else:
                    # Только одна детекция - берём её
                    best = max(valid_dets, key=lambda d: d['confidence'])

                # Обновляем последнюю позицию
                self.last_ball_position = (best['x'], best['y'])
                self.frames_without_reliable_detection = 0

                # Формируем вектор детекции
                cx_g = float(best['x'])
                cy_g = float(best['y'])
                w_g = float(best['width'])
                h_g = float(best['height'])
                conf = float(best['confidence'])

                # Логируем каждую сырую детекцию мяча
                if self.analysis_frame_count % 5 == 0:
                    logger.info(f"🔴 RAW BALL: ts={ts:.2f}, pos=({cx_g:.0f},{cy_g:.0f}), conf={conf:.3f}, size={w_g:.0f}x{h_g:.0f}")

                det_vec = [
                    int(cx_g), int(cy_g),
                    float(w_g), float(h_g),
                    conf, 0,
                    int(cx_g), int(cy_g),
                    float(w_g), float(h_g)
                ]

                # Добавляем в историю
                self.history.add_detection(det_vec, ts, per_ts_fnum.get(ts, 0))

                # ========== BACKWARD INTERPOLATION при долгих потерях ==========
                # Проверяем: это восстановление после долгой потери?
                last_detection = self.history.get_last_confirmed_detection()

                if last_detection:
                    gap_duration = ts - last_detection['timestamp']

                    # Применяем backward interpolation если gap >= 5 секунд
                    if gap_duration >= 5.0:
                        # Вычисляем расстояние между позициями
                        distance = math.sqrt(
                            (cx_g - last_detection['x'])**2 +
                            (cy_g - last_detection['y'])**2
                        )

                        # Проверяем: playback ещё не достиг точки интерполяции?
                        interpolation_start = ts - 1.0  # За 1 секунду до обнаружения

                        if self.current_display_timestamp < interpolation_start:
                            # Применяем backward interpolation!
                            self.history.insert_backward_interpolation(
                                start_ts=interpolation_start,
                                end_ts=ts,
                                start_pos=(last_detection['x'], last_detection['y']),
                                end_pos=(cx_g, cy_g)
                            )

                            logger.info(f"📍 GAP DETECTED: {gap_duration:.1f}s, distance={distance:.0f}px, "
                                       f"from ({last_detection['x']:.0f},{last_detection['y']:.0f}) "
                                       f"to ({cx_g:.0f},{cy_g:.0f})")
                        else:
                            logger.warning(f"⚠️ Cannot apply backward interp: playback already at "
                                         f"{self.current_display_timestamp:.2f}s, need {interpolation_start:.2f}s "
                                         f"(gap={gap_duration:.1f}s)")

                # Добавляем мяч в all_detections_history для отрисовки
                if ts not in self.all_detections_history:
                    self.all_detections_history[ts] = {'ball': [], 'player': [], 'staff': [], 'referee': []}
                self.all_detections_history[ts]['ball'].append(best)

                # Дебаг: показываем размер истории (правильный способ!)
                history_size = len(self.history.raw_future_history) + len(self.history.processed_future_history) + len(self.history.confirmed_history)

                # Логирование успешной детекции каждые 10 кадров
                # if self.analysis_frame_count % 10 == 0:
                #     logger.info(f"   🔴 Valid Ball: 1 (passed all filters)")
                #     logger.info(f"🎯 Ball Detection: pos=({cx_g:.0f},{cy_g:.0f}), conf={conf:.3f}, tile={best.get('tile_id', '?')}, history_size={history_size}")

            # Статистика каждые 40 кадров (было 10, оптимизировано)
            # if self.analysis_frame_count % 40 == 0 and self.start_time:
            #     elapsed = max(1e-6, time.time() - self.start_time)
            #     fps_a = self.analysis_frame_count / elapsed
            #     # Подсчитываем сколько детекций в истории (все 3 словаря)
            #     valid_count = len(self.history.raw_future_history) + len(self.history.processed_future_history) + len(self.history.confirmed_history)
            #     logger.info(f"[Analysis] frame={self.analysis_frame_count}, fps≈{fps_a:.2f}, "
            #                f"raw_detections={self.detection_count}, valid_after_filters={valid_count}")
                
        except Exception as e:
            logger.error(f"analysis_probe error: {e}")
            import traceback
            traceback.print_exc()
            
        return Gst.PadProbeReturn.OK

    def draw_probe(self, pad, info, u_data):
        """Отрисовка детекций на nvdsosd для режима панорамы."""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            pts_sec = float(gst_buffer.pts) / float(Gst.SECOND)

            # Обновляем display timestamp
            self.history.update_display_timestamp(pts_sec)

            # Получаем детекцию мяча для текущего момента
            det = self.history.get_detection_for_timestamp(pts_sec, max_delta=0.12)

            # Получаем ВСЕ детекции для отрисовки
            all_detections = self.get_all_detections_for_timestamp(pts_sec, max_delta=0.12)

            # В режиме панорамы показываем интерполяцию мяча
            if det:
                if all_detections is None:
                    all_detections = {'ball': [], 'player': [], 'staff': [], 'referee': []}

                ball_det = {
                    'x': det[6] if len(det) > 6 else det[0],
                    'y': det[7] if len(det) > 7 else det[1],
                    'width': det[8] if len(det) > 8 else det[2],
                    'height': det[9] if len(det) > 9 else det[3],
                    'confidence': det[4] if len(det) > 4 else 0.5,
                    'is_interpolated': True
                }
                all_detections['ball'].append(ball_det)

            # Вычисляем центр масс
            self._compute_smoothed_center_of_mass(pts_sec)

            # Рисуем на nvdsosd
            l_frame = batch_meta.frame_meta_list
            while l_frame:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                if not fm:
                    l_frame = l_frame.next
                    continue

                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                if not display_meta:
                    l_frame = l_frame.next
                    continue

                # FPS и информация
                elapsed = max(1e-6, time.time() - self.start_time) if self.start_time else 0
                self.current_fps = float(self.display_frame_count) / elapsed if elapsed > 0 else 0
                text = f"FPS:{self.current_fps:.1f}"

                self.display_frame_count += 1

                # Подсчитываем прямоугольники
                num_detection_rects = 0
                if all_detections:
                    num_detection_rects = (len(all_detections.get('ball', [])) +
                                          len(all_detections.get('player', [])))

                max_available_rects = 16
                if num_detection_rects > max_available_rects:
                    num_detection_rects = max_available_rects

                display_meta.num_rects = num_detection_rects
                display_meta.num_labels = 1

                # Рисуем детекции
                rect_idx = 0

                if all_detections:
                    class_colors = {
                        'ball': (1.0, 0.0, 0.0, 1.0),      # Красный
                        'player': (0.0, 1.0, 0.0, 1.0)      # Зелёный
                    }

                    class_widths = {
                        'ball': 3,
                        'player': 2
                    }

                    for class_name, color in class_colors.items():
                        detections_list = all_detections.get(class_name, [])

                        # Для игроков - центр масс
                        if class_name == 'player' and self.players_center_mass_smoothed:
                            center_x, center_y = self.players_center_mass_smoothed
                            cm_box_size = 100

                            if rect_idx < num_detection_rects:
                                try:
                                    left = int(center_x - cm_box_size / 2)
                                    top = int(center_y - cm_box_size / 2)

                                    rect = display_meta.rect_params[rect_idx]
                                    rect.left = max(0, left)
                                    rect.top = max(0, top)
                                    rect.width = cm_box_size
                                    rect.height = cm_box_size
                                    rect.border_width = 4
                                    rect.border_color.set(*color)
                                    rect.has_bg_color = 0

                                    rect_idx += 1
                                except (IndexError, Exception):
                                    break

                        # Для мяча - обычные боксы
                        elif class_name == 'ball':
                            for d in detections_list:
                                if rect_idx >= num_detection_rects:
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

                                    if is_interp:
                                        rect.border_width = class_widths[class_name]
                                        rect.border_color.set(1.0, 1.0, 0.0, 1.0)  # Жёлтый
                                    else:
                                        rect.border_width = class_widths[class_name]
                                        rect.border_color.set(*color)

                                    rect.has_bg_color = 0
                                    rect_idx += 1
                                except (IndexError, Exception):
                                    break

                        if rect_idx >= num_detection_rects:
                            break

                # Добавляем текст
                if det is not None:
                    cx, cy, w, h, conf = det[0:5]
                    text += f" | Ball:({int(cx)},{int(cy)}) conf={conf:.2f}"

                if all_detections:
                    text += f" | Ball={len(all_detections.get('ball', []))}, Players={len(all_detections.get('player', []))}"

                lbl = display_meta.text_params[0]
                lbl.display_text = text
                lbl.x_offset = 10
                lbl.y_offset = 10
                lbl.font_params.font_name = "Serif"
                lbl.font_params.font_size = 20
                lbl.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                lbl.set_bg_clr = 1
                lbl.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

                pyds.nvds_add_display_meta_to_frame(fm, display_meta)
                break

        except Exception as e:
            logger.error(f"draw_probe error: {e}")
            import traceback
            traceback.print_exc()

        return Gst.PadProbeReturn.OK

    def _apply_adaptive_distance_filter(self, detections, last_position, frames_missed):
        """Адаптивный фильтр по расстоянию."""
        if not last_position:
            return detections
            
        base_radius = 100
        search_radius = base_radius * (1 + frames_missed * 0.2)
        search_radius = min(search_radius, 500)
        
        filtered = []
        for d in detections:
            dist = math.sqrt((d['x'] - last_position[0])**2 + 
                           (d['y'] - last_position[1])**2)
            if dist <= search_radius:
                filtered.append(d)
                
        return filtered if filtered else detections
        
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

    def _emergency_shutdown(self):
        """Экстренное завершение для освобождения камер."""
        logger.error("⚠️ ЭКСТРЕННОЕ ЗАВЕРШЕНИЕ - освобождаем камеры...")
        
        try:
            # Сначала останавливаем источники (камеры)
            if self.pipeline:
                # Находим и останавливаем nvarguscamerasrc элементы
                it = self.pipeline.iterate_elements()
                while True:
                    ret, element = it.next()
                    if ret != Gst.IteratorResult.OK:
                        break
                    if element and element.get_factory():
                        if "nvarguscamerasrc" in element.get_factory().get_name():
                            element.set_state(Gst.State.NULL)
                            logger.info(f"Камера остановлена: {element.get_name()}")
        except:
            pass
        
        # Теперь останавливаем основной пайплайн
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
        except:
            pass
        
        # Завершаем main loop
        if self.loop.is_running():
            self.loop.quit()
        
        # Форсированный выход через 1 секунду если что-то зависло
        def force_exit():
            logger.error("🛑 ФОРСИРОВАННЫЙ ВЫХОД!")
            os._exit(1)  # Жёсткий выход
        
        GLib.timeout_add(1000, force_exit)
        return False
            
    def _buffer_loop(self):
        """Фоновый поток для мониторинга (single pipeline с ring buffer)."""
        logger.info("[BUFFER] поток мониторинга запущен")

        # Для single pipeline с ring buffer: просто мониторим работу
        while self.buffer_thread_running:
            time.sleep(1.0)
            # Можно добавить логирование статистики здесь если нужно

        logger.info("[BUFFER] поток мониторинга завершён")

        
    def _on_bus_message(self, bus, message):
        """Обработка сообщений шины."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[BUS] ERROR: {err}; debug: {debug}")
            self.stop()
        elif t == Gst.MessageType.EOS:
            logger.info("[BUS] EOS")
            self.stop()
        return True
        
    def run(self) -> bool:
        """Запуск приложения."""
        if not self.create_pipeline():
            return False

        if self.display_mode == "stream_url":
            logger.info(f"🚀 Запуск stream_url стриминга с виртуальной камерой")
            logger.info(f"🔑 Ключ: {self.stream_key[:4]}...{self.stream_key[-4:]}")
            logger.info(f"📺 URL: {self.stream_url}")
            logger.info(f"📷 Виртуальная камера будет следить за мячом")
        else:
            logger.info(f"Запуск основного пайплайна в режиме {self.display_mode}…")
                
        main_bus = self.pipeline.get_bus()
        main_bus.add_signal_watch()
        main_bus.connect("message", self._on_bus_message)
        
        logger.info(f"Запуск основного пайплайна в режиме {self.display_mode}…")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.start_time = time.time()
        
        self.buffer_thread_running = True
        self.buffer_thread = threading.Thread(target=self._buffer_loop, daemon=True)
        self.buffer_thread.start()
        
        try:
            logger.info("Главный цикл запущен. Нажмите Ctrl+C для выхода.")
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Остановлено пользователем (Ctrl+C).")
        finally:
            self.stop()
            
        return True
        
    def stop(self):
        """Корректное завершение."""
        try:
            self.buffer_thread_running = False
            time.sleep(0.3)
        except:
            pass

        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
        except:
            pass

        try:
            if self.loop.is_running():
                self.loop.quit()
        except:
            pass

        logger.info(f"[STATS] display_frames={self.display_frame_count}")
        logger.info("Остановлено.")


# =========================
# MAIN
# =========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Панорама с виртуальной камерой и записью')
    
    # Выбор источника
    parser.add_argument('--source', choices=['files', 'cameras'], default='files',
                       help='Источник видео: files (видеофайлы) или cameras (MIPI CSI камеры)')
    
    # Источники
    parser.add_argument('--video1', default="left.mp4", 
                       help="Левое видео (путь к файлу или ID камеры)")
    parser.add_argument('--video2', default="right.mp4", 
                       help="Правое видео (путь к файлу или ID камеры)")
    
    parser.add_argument('--config', default=None, help="Путь к конфигу nvinfer")
    parser.add_argument('--buffer', type=float, default=7.0, help="Длительность буфера (сек)")
    
    parser.add_argument('--mode', choices=['panorama', 'virtualcam', 'stream', 'record'],
                       default='virtualcam',
                       help='Режим: panorama=окно панорамы, virtualcam=окно камеры, stream=стрим на YouTube, record=только запись в файл')

    parser.add_argument('--output', type=str, default=None,
                       help='Путь к файлу для записи (работает только в режимах stream и record)')

    parser.add_argument('--stream-url', default='rtmp://a.rtmp.youtube.com/live2/',
                       help='RTMP URL для стриминга (например: rtmp://live.twitch.tv/live)')
    parser.add_argument('--stream-key', default='ufpj-dffk-f1de-8ya6-crq5',
                       help='Ключ стрима stream')
    parser.add_argument('--bitrate', type=int, default=6000000,
                       help='Битрейт видео в bps (3500000=3.5Mbps для слабого 4G, 4500000=4.5Mbps для среднего 4G, 6000000=6Mbps для хорошего WiFi/4G)')
    parser.add_argument('--skip-interval', type=int, default=8,
                       help='Анализировать каждый N-й кадр')
    parser.add_argument('--confidence', type=float, default=0.35, 
                       help='Порог уверенности детекции')
    parser.add_argument('--no-zoom', action='store_true', 
                       help='Отключить автозум в режиме virtualcam')
    parser.add_argument('--disable-display', action='store_true', 
                       help='Отключить отображение')
    parser.add_argument('--disable-analysis', action='store_true', 
                       help='Отключить анализ')
    
    args = parser.parse_args()

    # Валидация параметров записи
    if args.output:
        # --output работает только с режимами stream и record
        if args.mode not in ['stream', 'record']:
            logger.error("Ошибка: --output работает только с режимами 'stream' или 'record'")
            logger.error(f"Текущий режим: {args.mode}")
            return 1

        # Для режима record параметр --output обязателен
        if args.mode == 'record' and not args.output:
            logger.error("Ошибка: для режима 'record' необходимо указать --output <файл>")
            return 1

    # Проверка режима record без --output
    if args.mode == 'record' and not args.output:
        logger.error("Ошибка: для режима 'record' необходимо указать --output <файл>")
        return 1

    # Автоматический выбор оптимального битрейта для режима record
    # Если пользователь НЕ указал --bitrate явно, используем максимум для записи
    if args.mode == 'record' and args.bitrate == 6000000:  # Значение по умолчанию
        args.bitrate = 8000000  # 8 Mbps для максимального качества записи
        logger.info("📹 Режим записи: автоматически установлен битрейт 8 Mbps (максимальное качество)")
        logger.info("   Для изменения используйте: --bitrate <значение>")

    # Проверка источников
    if args.source == "files":
        # Проверяем существование файлов
        for vf in [args.video1, args.video2]:
            if not os.path.exists(vf):
                logger.error(f"Файл не найден: {vf}")
                return 1
    else:
        # Для камер преобразуем в числа
        try:
            cam1 = int(args.video1)
            cam2 = int(args.video2)
            args.video1 = str(cam1)
            args.video2 = str(cam2)
            logger.info(f"Используем камеры: {cam1} и {cam2}")
        except ValueError:
            logger.error("Для камер укажите числовые ID (например: --video1 0 --video2 1)")
            return 1
            
    app = PanoramaWithVirtualCamera(
        source_type=args.source,
        video1=args.video1,
        video2=args.video2,
        config_path=args.config,
        buffer_duration=args.buffer,
        enable_display=not args.disable_display,
        display_mode=args.mode,
        enable_analysis=not args.disable_analysis,
        analysis_skip_interval=args.skip_interval,
        confidence_threshold=args.confidence,
        auto_zoom=not args.no_zoom,
        stream_url=args.stream_url,
        stream_key=args.stream_key,
        output_file=args.output,
        bitrate=args.bitrate
    )
    
    ok = app.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())