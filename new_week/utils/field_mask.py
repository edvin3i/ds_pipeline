#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary field mask for filtering detections."""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("panorama-virtualcam")


class FieldMaskBinary:
    """Бинарная маска поля для фильтрации детекций."""

    def __init__(self, mask_path='field_mask.png', panorama_width=5700, panorama_height=1900):
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
