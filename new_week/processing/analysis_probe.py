#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Detection Processing Handler

This module handles all YOLO detection processing from the analysis pipeline,
including multi-class detection filtering, NMS application, and ball tracking.
"""

import math
import logging
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import pyds
from gi.repository import Gst
import numpy as np

from .tensor_processor import get_tensor_as_numpy

logger = logging.getLogger("panorama-virtualcam")


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


class AnalysisProbeHandler:
    """
    Handles YOLO detection processing from the analysis pipeline.

    This class processes multi-class YOLO detections (ball, players, staff, referees),
    applies various filters, NMS, and maintains detection history.
    """

    def __init__(self,
                 ball_history,
                 players_history,
                 field_mask,
                 tensor_processor,
                 roi_configs: List[Tuple[int, int, int, int]],
                 all_detections_history: Dict[float, Dict[str, List]],
                 panorama_width: int,
                 panorama_height: int,
                 confidence_threshold: float = 0.35,
                 analysis_skip_interval: int = 5,
                 history_manager=None):
        """
        Initialize the analysis probe handler.

        Args:
            ball_history: BallDetectionHistory instance for ball tracking
            players_history: PlayersHistory instance for player tracking
            field_mask: FieldMaskBinary instance for field boundary checking
            tensor_processor: TensorProcessor instance for YOLO output processing
            roi_configs: List of ROI configurations (x, y, width, height)
            all_detections_history: Shared dict for all detections by timestamp
            panorama_width: Width of panorama for coordinate validation
            panorama_height: Height of panorama for coordinate validation
            confidence_threshold: Minimum confidence for ball detection
            analysis_skip_interval: Frame skip interval for analysis
            history_manager: HistoryManager instance for timer-based trajectory updates
        """
        # Core dependencies
        self.history = ball_history
        self.players_history = players_history
        self.field_mask = field_mask
        self.tensor_processor = tensor_processor
        self.roi_configs = roi_configs
        self.panorama_width = panorama_width
        self.panorama_height = panorama_height
        self.history_manager = history_manager

        # Configuration
        self.confidence_threshold = confidence_threshold
        self.analysis_skip_interval = max(1, int(analysis_skip_interval))

        # State tracking
        self.analysis_actual_frame = 0
        self.analysis_frame_count = 0
        self.detection_count = 0
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.frames_without_reliable_detection = 0
        self.current_display_timestamp = 0.0

        # Detection history for rendering (shared reference, not a new dict)
        self.all_detections_history = all_detections_history

        # Timing (optional, for statistics)
        self.start_time = None

    def handle_analysis_probe(self, pad, info, user_data):
        """
        YOLO detection processing probe callback.

        This probe is attached to nvinfer src pad and processes raw tensor outputs
        for multi-class object detection (ball, players, staff, referees).

        Args:
            pad: GStreamer pad (nvinfer src pad)
            info: GstPadProbeInfo containing buffer
            user_data: User data (unused)

        Returns:
            Gst.PadProbeReturn.OK - Always continue processing.
            We never drop buffers as all frames are needed for detection history.

        Reference:
            DeepStream SDK 7.1 - /ds_doc/7.1/text/DS_Zero_Coding_DS_Components.html
        """
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

            # CRITICAL FIX: Proper metadata iteration with StopIteration handling
            # Reference: DeepStream SDK 7.1 documentation - /ds_doc/7.1/python-api/PYTHON_API/NvDsMeta/
            # All metadata list iteration MUST use try/except StopIteration blocks
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

                # Используем счетчик вместо pad_index!
                tile_id = tile_counter
                tile_counter += 1

                tiles_processed.append(tile_id)
                frame_num = fm.frame_num
                ts_sec = float(fm.buf_pts) / float(Gst.SECOND)

                l_user = fm.frame_user_meta_list
                while l_user is not None:
                    try:
                        um = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if um and um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        # Validate user metadata has data (best practice)
                        if not um.user_meta_data:
                            logger.warning("User metadata has no data, skipping")
                            try:
                                l_user = l_user.next
                            except StopIteration:
                                break
                            continue

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

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

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

        # ===== CRITICAL: Update camera trajectory on every YOLO detection (every ~0.5s) =====
        # This is independent of whether ball was detected!
        # Ensures trajectory updates for:
        # 1. Initial ball detection (no ball at startup)
        # 2. Ball lost 7+ seconds (history cleared, only player COM available)
        # 3. Regular updates with interpolated ball positions
        if self.history_manager:
            try:
                self.history_manager.update_camera_trajectory_on_timer()
            except Exception as e:
                logger.warning(f"⚠️ Failed to update camera trajectory on timer: {e}")

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
