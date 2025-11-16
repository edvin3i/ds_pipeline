#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO tensor post-processing."""

import numpy as np
import ctypes
import logging
import pyds

logger = logging.getLogger("panorama-virtualcam")


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
