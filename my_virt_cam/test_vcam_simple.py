#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощённый тест виртуальной камеры - создаёт несколько изображений с разными параметрами
"""

import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("vcam_test")

Gst.init(None)

def test_vcam(yaw, pitch, fov, name):
    """Создаёт одно изображение с заданными параметрами"""

    output_file = f"test_{name}.jpg"

    pipeline_str = f"""
        multifilesrc location=left.jpg !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        queue !
        nvstreammux0.sink_0

        multifilesrc location=right.jpg !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        queue !
        nvstreammux0.sink_1

        nvstreammux name=nvstreammux0 batch-size=2 width=3840 height=2160 !

        nvdsstitch
            left-source-id=0
            right-source-id=1
            panorama-width=6528
            panorama-height=1800 !

        nvdsvirtualcam
            yaw={yaw}
            pitch={pitch}
            fov={fov}
            output-width=1920
            output-height=1080
            panorama-width=6528
            panorama-height=1800 !

        nvvideoconvert !
        jpegenc !
        filesink location={output_file}
    """

    logger.info(f"Создание {output_file}: yaw={yaw}, pitch={pitch}, fov={fov}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        pipeline.set_state(Gst.State.PLAYING)

        # Ждём немного
        time.sleep(2)

        pipeline.set_state(Gst.State.NULL)
        logger.info(f"✓ {output_file} создан")
        return True

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return False

# Установка путей к плагинам
os.environ['GST_PLUGIN_PATH'] = "/home/nvidia/deep_cv_football/my_virt_cam/src:/home/nvidia/deep_cv_football/my_steach"

# Тестируем несколько конфигураций
tests = [
    (0, 0, 50, "center"),
    (-45, 0, 50, "left"),
    (45, 0, 50, "right"),
    (0, 0, 40, "zoom_in"),
    (0, 0, 60, "zoom_out"),
]

logger.info("=== ТЕСТ ВИРТУАЛЬНОЙ КАМЕРЫ ===")
for yaw, pitch, fov, name in tests:
    test_vcam(yaw, pitch, fov, name)
    time.sleep(1)

logger.info("=== ТЕСТ ЗАВЕРШЁН ===")
logger.info("Проверьте созданные файлы: test_*.jpg")