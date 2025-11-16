#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отладочный тест виртуальной камеры с детальной проверкой
"""

import os
import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vcam_debug")

# Инициализация GStreamer
Gst.init(None)

# Проверяем рабочую директорию
work_dir = Path("/home/nvidia/deep_cv_football/my_virt_cam")
os.chdir(work_dir)
logger.info(f"Рабочая директория: {os.getcwd()}")

# Установка путей к плагинам
plugin_paths = [
    "/home/nvidia/deep_cv_football/my_virt_cam/src",
    "/home/nvidia/deep_cv_football/my_steach"
]
os.environ['GST_PLUGIN_PATH'] = ":".join(plugin_paths)
logger.info(f"GST_PLUGIN_PATH: {os.environ['GST_PLUGIN_PATH']}")

# Проверяем наличие исходных файлов
left_jpg = work_dir / "left.jpg"
right_jpg = work_dir / "right.jpg"

if not left_jpg.exists():
    logger.error(f"Файл не найден: {left_jpg}")
    sys.exit(1)
else:
    logger.info(f"✓ Найден: {left_jpg} ({left_jpg.stat().st_size} байт)")

if not right_jpg.exists():
    logger.error(f"Файл не найден: {right_jpg}")
    sys.exit(1)
else:
    logger.info(f"✓ Найден: {right_jpg} ({right_jpg.stat().st_size} байт)")

# Проверяем наличие плагинов
def check_plugin(plugin_name):
    """Проверка наличия GStreamer плагина"""
    registry = Gst.Registry.get()
    plugin = registry.find_plugin(plugin_name)
    if plugin:
        logger.info(f"✓ Плагин {plugin_name} найден")
        return True
    else:
        logger.warning(f"✗ Плагин {plugin_name} НЕ найден")
        return False

# Проверяем необходимые плагины
check_plugin("nvdsstitch")
check_plugin("nvdsvirtualcam")

def test_simple_pipeline():
    """Тест простого pipeline без виртуальной камеры"""

    logger.info("\n=== ТЕСТ 1: Простое копирование изображения ===")
    output_file = work_dir / "test_copy.jpg"

    pipeline_str = f"""
        filesrc location={left_jpg} !
        jpegdec !
        jpegenc !
        filesink location={output_file}
    """

    logger.debug(f"Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)

        # Добавляем обработчик сообщений
        bus = pipeline.get_bus()

        pipeline.set_state(Gst.State.PLAYING)
        logger.info("Pipeline запущен...")

        # Ждём завершения
        msg = bus.timed_pop_filtered(
            5 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"Ошибка: {err}, {debug}")
                return False
            elif msg.type == Gst.MessageType.EOS:
                logger.info("Pipeline завершён (EOS)")
        else:
            logger.warning("Таймаут ожидания")

        pipeline.set_state(Gst.State.NULL)

        # Проверяем результат
        if output_file.exists():
            size = output_file.stat().st_size
            logger.info(f"✓ Файл создан: {output_file} ({size} байт)")
            return size > 0
        else:
            logger.error(f"✗ Файл НЕ создан: {output_file}")
            return False

    except Exception as e:
        logger.error(f"Исключение: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stitch_only():
    """Тест только склейки без виртуальной камеры"""

    logger.info("\n=== ТЕСТ 2: Склейка двух изображений ===")
    output_file = work_dir / "test_stitch.jpg"

    pipeline_str = f"""
        filesrc location={left_jpg} !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        nvstreammux0.sink_0

        filesrc location={right_jpg} !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        nvstreammux0.sink_1

        nvstreammux name=nvstreammux0
            batch-size=2
            width=3840
            height=2160
            live-source=0
            num-surfaces-per-frame=1 !

        nvdsstitch
            left-source-id=0
            right-source-id=1
            panorama-width=6528
            panorama-height=1800 !

        nvvideoconvert !
        video/x-raw,format=RGBA !
        videoconvert !
        jpegenc !
        filesink location={output_file}
    """

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        bus = pipeline.get_bus()

        pipeline.set_state(Gst.State.PLAYING)
        logger.info("Stitch pipeline запущен...")

        msg = bus.timed_pop_filtered(
            10 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"Ошибка: {err}")
                logger.error(f"Debug: {debug}")
                return False
            elif msg.type == Gst.MessageType.EOS:
                logger.info("Stitch pipeline завершён (EOS)")
        else:
            logger.warning("Таймаут stitch pipeline")

        pipeline.set_state(Gst.State.NULL)

        if output_file.exists():
            size = output_file.stat().st_size
            logger.info(f"✓ Панорама создана: {output_file} ({size} байт)")
            return size > 0
        else:
            logger.error(f"✗ Панорама НЕ создана")
            return False

    except Exception as e:
        logger.error(f"Исключение в stitch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_virtual_camera(yaw=0, pitch=0, fov=50):
    """Тест полного pipeline с виртуальной камерой"""

    logger.info(f"\n=== ТЕСТ 3: Виртуальная камера (yaw={yaw}, pitch={pitch}, fov={fov}) ===")
    output_file = work_dir / f"test_vcam_y{yaw}_p{pitch}_f{fov}.jpg"

    pipeline_str = f"""
        filesrc location={left_jpg} !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        nvstreammux0.sink_0

        filesrc location={right_jpg} !
        jpegdec !
        videoconvert !
        videoscale !
        video/x-raw,width=3840,height=2160,format=RGBA !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA !
        nvstreammux0.sink_1

        nvstreammux name=nvstreammux0
            batch-size=2
            width=3840
            height=2160
            live-source=0
            num-surfaces-per-frame=1 !

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
        video/x-raw,format=RGBA !
        videoconvert !
        jpegenc !
        filesink location={output_file}
    """

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        bus = pipeline.get_bus()

        pipeline.set_state(Gst.State.PLAYING)
        logger.info("Virtual camera pipeline запущен...")

        msg = bus.timed_pop_filtered(
            10 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"Ошибка: {err}")
                logger.error(f"Debug info: {debug}")
                return False
            elif msg.type == Gst.MessageType.EOS:
                logger.info("Virtual camera pipeline завершён (EOS)")
        else:
            logger.warning("Таймаут virtual camera pipeline")

        pipeline.set_state(Gst.State.NULL)

        if output_file.exists():
            size = output_file.stat().st_size
            logger.info(f"✓ Virtual camera создана: {output_file} ({size} байт)")
            return size > 0
        else:
            logger.error(f"✗ Virtual camera НЕ создана")
            return False

    except Exception as e:
        logger.error(f"Исключение в virtual camera: {e}")
        import traceback
        traceback.print_exc()
        return False

# Запускаем тесты
logger.info("="*60)
logger.info("НАЧАЛО ОТЛАДОЧНОГО ТЕСТИРОВАНИЯ")
logger.info("="*60)

# Тест 1: простое копирование
test_simple_pipeline()
time.sleep(1)

# Тест 2: склейка
test_stitch_only()
time.sleep(1)

# Тест 3: виртуальная камера
test_virtual_camera(0, 0, 50)
time.sleep(1)
test_virtual_camera(-45, 0, 50)

# Итоговая проверка
logger.info("\n" + "="*60)
logger.info("ИТОГОВАЯ ПРОВЕРКА СОЗДАННЫХ ФАЙЛОВ:")
logger.info("="*60)

test_files = list(work_dir.glob("test_*.jpg"))
if test_files:
    for f in sorted(test_files):
        size = f.stat().st_size
        logger.info(f"  {f.name}: {size:,} байт")
else:
    logger.error("НЕТ СОЗДАННЫХ ТЕСТОВЫХ ФАЙЛОВ!")

logger.info("\nТестирование завершено.")