#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV logging utilities for ball detections."""

import os
import csv
import time
import logging

logger = logging.getLogger("panorama-virtualcam")


def save_detection_to_csv(detection, timestamp, frame_num, file_path=None):
    """Запись детекции в TSV файл."""
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
