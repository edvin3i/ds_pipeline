#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест виртуальной камеры с использованием изображений JPG
Создаёт панораму из двух изображений и применяет виртуальную камеру
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
from pathlib import Path
import logging
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("virtual_camera_images")

# Инициализация GStreamer
Gst.init(None)

# Константы размеров панорамы
PANORAMA_WIDTH = 6528
PANORAMA_HEIGHT = 1800

# Размер выходного изображения виртуальной камеры
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

class VirtualCameraImageTest:
    """Тестирование виртуальной камеры с изображениями"""

    def __init__(self, left_image, right_image, output_dir="output"):
        self.left_image = left_image
        self.right_image = right_image
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Параметры для тестирования
        self.test_params = [
            # (yaw, pitch, fov, описание)
            (0, 0, 50, "center"),
            (-45, 0, 50, "left_45"),
            (45, 0, 50, "right_45"),
            (-90, 0, 50, "left_90"),
            (90, 0, 50, "right_90"),
            (0, -15, 50, "down_15"),
            (0, 15, 50, "up_15"),
            (0, 0, 40, "zoom_in"),
            (0, 0, 60, "zoom_out"),
            (0, 0, 75, "wide_angle"),
            (-30, 10, 55, "left_up"),
            (30, -10, 45, "right_down"),
        ]

    def create_pipeline_for_params(self, yaw, pitch, fov, output_file):
        """Создаёт pipeline для конкретных параметров"""

        pipeline_str = f"""
            multifilesrc location={self.left_image} caps="image/jpeg,framerate=1/1" !
            jpegdec !
            videoconvert !
            videoscale !
            video/x-raw,width=3840,height=2160,format=RGBA !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=1 !
            nvstreammux0.sink_0

            multifilesrc location={self.right_image} caps="image/jpeg,framerate=1/1" !
            jpegdec !
            videoconvert !
            videoscale !
            video/x-raw,width=3840,height=2160,format=RGBA !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=1 !
            nvstreammux0.sink_1

            nvstreammux name=nvstreammux0
                batch-size=2
                width=3840
                height=2160
                batched-push-timeout=40000
                num-surfaces-per-frame=1
                live-source=0 !

            nvdsstitch name=stitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=1 !

            nvdsvirtualcam name=vcam
                yaw={yaw}
                pitch={pitch}
                roll=0
                fov={fov}
                output-width={OUTPUT_WIDTH}
                output-height={OUTPUT_HEIGHT}
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=1 !

            nvvideoconvert !
            video/x-raw,format=RGBA !
            videoconvert !
            video/x-raw,format=RGB !
            jpegenc quality=95 !
            filesink location={output_file}
        """

        return Gst.parse_launch(pipeline_str)

    def process_single_configuration(self, yaw, pitch, fov, name):
        """Обрабатывает одну конфигурацию параметров"""

        output_file = self.output_dir / f"vcam_{name}_y{yaw}_p{pitch}_f{fov}.jpg"

        logger.info(f"Обработка: {name} (yaw={yaw}, pitch={pitch}, fov={fov})")

        try:
            # Создаём pipeline
            pipeline = self.create_pipeline_for_params(yaw, pitch, fov, str(output_file))

            # Настраиваем обработку сообщений
            bus = pipeline.get_bus()

            # Запускаем pipeline
            pipeline.set_state(Gst.State.PLAYING)

            # Ждём завершения
            msg = bus.timed_pop_filtered(
                10 * Gst.SECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )

            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    logger.error(f"Ошибка: {err}, {debug}")
                    return False
                elif msg.type == Gst.MessageType.EOS:
                    logger.info(f"✓ Сохранено: {output_file}")
            else:
                logger.warning(f"Таймаут для {name}")

            # Останавливаем pipeline
            pipeline.set_state(Gst.State.NULL)

            return True

        except Exception as e:
            logger.error(f"Ошибка при обработке {name}: {e}")
            return False

    def create_comparison_grid(self):
        """Создаёт сетку для сравнения всех результатов"""

        logger.info("Создание сравнительной сетки...")

        images = []
        labels = []

        # Загружаем все созданные изображения
        for yaw, pitch, fov, name in self.test_params:
            img_path = self.output_dir / f"vcam_{name}_y{yaw}_p{pitch}_f{fov}.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Масштабируем для сетки
                    img_small = cv2.resize(img, (480, 270))

                    # Добавляем подпись
                    cv2.putText(img_small, f"{name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img_small, f"Y:{yaw} P:{pitch} F:{fov}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    images.append(img_small)
                    labels.append(name)

        if not images:
            logger.warning("Нет изображений для создания сетки")
            return

        # Создаём сетку 4x3
        cols = 4
        rows = (len(images) + cols - 1) // cols

        grid_width = 480 * cols
        grid_height = 270 * rows
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            y1 = row * 270
            y2 = y1 + 270
            x1 = col * 480
            x2 = x1 + 480
            grid[y1:y2, x1:x2] = img

        # Сохраняем сетку
        grid_path = self.output_dir / "comparison_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        logger.info(f"✓ Сетка сохранена: {grid_path}")

    def test_animation_sequence(self):
        """Создаёт последовательность кадров для анимации"""

        logger.info("Создание анимационной последовательности...")

        # Параметры анимации - плавное движение камеры
        animation_params = []

        # Горизонтальное панорамирование
        for i in range(-90, 91, 10):
            animation_params.append((i, 0, 50, f"pan_{i:03d}"))

        # Вертикальное панорамирование
        for i in range(-30, 23, 5):
            animation_params.append((0, i, 50, f"tilt_{i:03d}"))

        # Изменение зума
        for i in range(40, 76, 5):
            animation_params.append((0, 0, i, f"zoom_{i:03d}"))

        # Обрабатываем каждый кадр
        for idx, (yaw, pitch, fov, name) in enumerate(animation_params):
            output_file = self.output_dir / "animation" / f"frame_{idx:04d}_{name}.jpg"
            output_file.parent.mkdir(exist_ok=True)

            logger.info(f"Кадр {idx+1}/{len(animation_params)}: {name}")

            try:
                pipeline = self.create_pipeline_for_params(yaw, pitch, fov, str(output_file))
                bus = pipeline.get_bus()
                pipeline.set_state(Gst.State.PLAYING)

                msg = bus.timed_pop_filtered(
                    5 * Gst.SECOND,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS
                )

                pipeline.set_state(Gst.State.NULL)

            except Exception as e:
                logger.error(f"Ошибка кадра {idx}: {e}")

        logger.info("✓ Анимационная последовательность создана")

    def run(self):
        """Запуск всех тестов"""

        logger.info("="*60)
        logger.info("ТЕСТИРОВАНИЕ ВИРТУАЛЬНОЙ КАМЕРЫ С ИЗОБРАЖЕНИЯМИ")
        logger.info(f"Левое изображение: {self.left_image}")
        logger.info(f"Правое изображение: {self.right_image}")
        logger.info(f"Выходная директория: {self.output_dir}")
        logger.info("="*60)

        # Обрабатываем все конфигурации
        success_count = 0
        for yaw, pitch, fov, name in self.test_params:
            if self.process_single_configuration(yaw, pitch, fov, name):
                success_count += 1

        logger.info(f"\nОбработано успешно: {success_count}/{len(self.test_params)}")

        # Создаём сравнительную сетку
        self.create_comparison_grid()

        # Опционально: создаём анимацию
        if input("\nСоздать анимационную последовательность? (y/n): ").lower() == 'y':
            self.test_animation_sequence()

        logger.info("\n✓ Тестирование завершено!")
        logger.info(f"Результаты сохранены в: {self.output_dir}")


def extract_frame_from_video(video_path, output_path, frame_number=100):
    """Извлекает кадр из видео и сохраняет как JPG"""

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(output_path), frame)
        logger.info(f"Кадр {frame_number} извлечён: {output_path}")
        return True
    else:
        logger.error(f"Не удалось извлечь кадр из {video_path}")
        return False

    cap.release()


def main():
    """Точка входа"""

    if len(sys.argv) < 3:
        print("Использование:")
        print("  1) С изображениями: python3 test_virtual_camera_images.py left.jpg right.jpg [output_dir]")
        print("  2) С видео (извлечь кадры): python3 test_virtual_camera_images.py left.mp4 right.mp4 --extract")
        sys.exit(1)

    left_input = sys.argv[1]
    right_input = sys.argv[2]

    # Проверяем, нужно ли извлечь кадры из видео
    if len(sys.argv) > 3 and sys.argv[3] == "--extract":
        # Извлекаем кадры из видео
        left_image = "left_frame.jpg"
        right_image = "right_frame.jpg"

        if not extract_frame_from_video(left_input, left_image, 100):
            sys.exit(1)
        if not extract_frame_from_video(right_input, right_image, 100):
            sys.exit(1)

        output_dir = "output_images"
    else:
        # Используем существующие изображения
        left_image = left_input
        right_image = right_input
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "output_images"

    # Проверяем существование файлов
    if not Path(left_image).exists():
        logger.error(f"Левое изображение не найдено: {left_image}")
        sys.exit(1)

    if not Path(right_image).exists():
        logger.error(f"Правое изображение не найдено: {right_image}")
        sys.exit(1)

    # Устанавливаем путь к плагинам
    plugin_path = "/home/nvidia/deep_cv_football/my_virt_cam/src"
    stitch_path = "/home/nvidia/deep_cv_football/my_steach"
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{stitch_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

    logger.info(f"GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH')}")

    try:
        # Создаём и запускаем тестер
        tester = VirtualCameraImageTest(left_image, right_image, output_dir)
        tester.run()

    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()