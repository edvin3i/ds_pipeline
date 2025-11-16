#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-Maximum Suppression utilities."""

import numpy as np


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
