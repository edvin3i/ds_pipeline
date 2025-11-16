#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивный тест виртуальной камеры с ползунками управления
Использует nvdsstitch для склейки и nvdsvirtualcam для виртуальной камеры
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import cv2
import numpy as np
import threading
import time
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("virtual_camera_test")

# Инициализация GStreamer
Gst.init(None)

# Константы размеров панорамы
PANORAMA_WIDTH = 6528
PANORAMA_HEIGHT = 1800

# Размер выходного изображения виртуальной камеры
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

class VirtualCameraController:
    """Контроллер виртуальной камеры с интерактивным управлением"""

    def __init__(self, left_file, right_file):
        self.left_file = left_file
        self.right_file = right_file

        # Параметры виртуальной камеры (согласно ограничениям плагина)
        self.yaw = 0.0      # Горизонтальный поворот (-90 до 90)
        self.pitch = 0.0    # Вертикальный поворот (-32 до 22)
        self.roll = 0.0     # Крен (-23 до 23)
        self.fov = 50.0     # Поле зрения (40 до 75)

        # GStreamer компоненты
        self.pipeline = None
        self.loop = None
        self.vcam_element = None
        self.is_running = False

        # OpenCV окно для отображения
        self.window_name = "Виртуальная камера - Управление"
        self.control_window = "Панель управления"

        # Буфер для кадров
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Event()

        # Счётчики для статистики
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_fps_count = 0

    def create_pipeline(self):
        """Создаёт GStreamer pipeline для склейки и виртуальной камеры"""

        pipeline_str = f"""
            filesrc location={self.left_file} !
            qtdemux ! h264parse ! nvv4l2decoder name=dec0 !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_0

            filesrc location={self.right_file} !
            qtdemux ! h264parse ! nvv4l2decoder name=dec1 !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_1

            nvstreammux name=nvstreammux0
                batch-size=2
                width=3840
                height=2160
                batched-push-timeout=40000
                live-source=0 !

            nvdsstitch name=stitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=3 !

            nvdsvirtualcam name=vcam
                yaw={self.yaw}
                pitch={self.pitch}
                roll={self.roll}
                fov={self.fov}
                output-width={OUTPUT_WIDTH}
                output-height={OUTPUT_HEIGHT}
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=3 !

            nvvideoconvert !
            video/x-raw,format=BGRx !
            videoconvert !
            video/x-raw,format=BGR !
            appsink name=appsink
                emit-signals=true
                sync=false
                max-buffers=1
                drop=true
        """

        logger.info("Создание pipeline...")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Получаем элемент виртуальной камеры для динамического управления
        self.vcam_element = self.pipeline.get_by_name("vcam")

        # Настраиваем appsink для получения кадров
        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)

        # Настраиваем обработку сообщений
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        logger.info("Pipeline создан успешно")

    def on_new_sample(self, sink):
        """Обработчик новых кадров из pipeline"""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Получаем размеры изображения
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")

            # Извлекаем данные из буфера
            result, mapinfo = buffer.map(Gst.MapFlags.READ)
            if result:
                # Конвертируем в numpy array
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=mapinfo.data
                )

                # Сохраняем кадр
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1

                self.frame_ready.set()
                buffer.unmap(mapinfo)

                # Обновляем FPS
                self.update_fps()

        return Gst.FlowReturn.OK

    def update_fps(self):
        """Обновляет счётчик FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = (self.frame_count - self.last_fps_count) / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
            self.last_fps_count = self.frame_count

    def on_message(self, bus, message):
        """Обработчик сообщений pipeline"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("Конец потока, перезапуск...")
            self.restart_pipeline()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Ошибка: {err}, {debug}")
            self.loop.quit()

    def restart_pipeline(self):
        """Перезапускает pipeline для зацикливания"""
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.set_state(Gst.State.PLAYING)

    def update_camera_params(self):
        """Обновляет параметры виртуальной камеры в реальном времени"""
        if self.vcam_element:
            self.vcam_element.set_property("yaw", self.yaw)
            self.vcam_element.set_property("pitch", self.pitch)
            self.vcam_element.set_property("roll", self.roll)
            self.vcam_element.set_property("fov", self.fov)

    def create_control_ui(self):
        """Создаёт UI с ползунками управления"""
        try:
            # Сначала создаём окна
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)

            # Показываем пустые окна чтобы они инициализировались
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imshow(self.window_name, empty_frame)

            control_panel = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.imshow(self.control_window, control_panel)

            # Даём окнам время появиться
            cv2.waitKey(1)

            # Теперь изменяем размер
            cv2.resizeWindow(self.control_window, 400, 300)

            # Создаём ползунки после того как окна созданы (с учётом ограничений плагина)
            cv2.createTrackbar('Yaw (Горизонт.)', self.control_window, 90, 180, self.on_yaw_change)
            cv2.createTrackbar('Pitch (Вертик.)', self.control_window, 32, 54, self.on_pitch_change)
            cv2.createTrackbar('FOV (Зум)', self.control_window, int(self.fov), 75, self.on_fov_change)
            cv2.createTrackbar('Roll (Крен)', self.control_window, 23, 46, self.on_roll_change)

            # Устанавливаем начальные значения
            cv2.setTrackbarPos('Yaw (Горизонт.)', self.control_window, int(self.yaw + 90))
            cv2.setTrackbarPos('Pitch (Вертик.)', self.control_window, int(self.pitch + 32))
            cv2.setTrackbarPos('FOV (Зум)', self.control_window, int(self.fov))
            cv2.setTrackbarPos('Roll (Крен)', self.control_window, int(self.roll + 23))

            logger.info("UI создан успешно")
        except Exception as e:
            logger.error(f"Ошибка создания UI: {e}")

    def on_yaw_change(self, value):
        """Обработчик изменения горизонтального угла"""
        self.yaw = value - 90  # Преобразуем в диапазон -90 до 90
        self.update_camera_params()

    def on_pitch_change(self, value):
        """Обработчик изменения вертикального угла"""
        self.pitch = value - 32  # Преобразуем в диапазон -32 до 22
        self.update_camera_params()

    def on_fov_change(self, value):
        """Обработчик изменения поля зрения"""
        self.fov = max(40, value)  # Минимум 40 градусов (согласно ограничениям плагина)
        self.update_camera_params()

    def on_roll_change(self, value):
        """Обработчик изменения крена"""
        self.roll = value - 23  # Преобразуем в диапазон -23 до 23
        self.update_camera_params()

    def draw_control_panel(self):
        """Рисует панель с информацией"""
        panel = np.zeros((300, 400, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # Тёмно-серый фон

        # Заголовок
        cv2.putText(panel, "Управление виртуальной камерой", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Текущие параметры
        y_offset = 70
        params = [
            f"Yaw (гориз.): {self.yaw:.1f}°",
            f"Pitch (верт.): {self.pitch:.1f}°",
            f"Roll (крен): {self.roll:.1f}°",
            f"FOV (зум): {self.fov:.1f}°",
            "",
            f"FPS: {self.fps:.1f}",
            f"Кадров: {self.frame_count}",
        ]

        for param in params:
            cv2.putText(panel, param, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

        # Инструкции
        y_offset = 230
        instructions = [
            "Клавиши:",
            "Space - Пауза/Воспроизведение",
            "R - Сброс параметров",
            "S - Сохранить кадр",
            "Q - Выход"
        ]

        for instruction in instructions:
            cv2.putText(panel, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += 20

        return panel

    def reset_params(self):
        """Сбрасывает параметры камеры к значениям по умолчанию"""
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.fov = 50.0

        # Обновляем ползунки (с учётом смещений)
        cv2.setTrackbarPos('Yaw (Горизонт.)', self.control_window, 90)  # 0 + 90
        cv2.setTrackbarPos('Pitch (Вертик.)', self.control_window, 32)  # 0 + 32
        cv2.setTrackbarPos('FOV (Зум)', self.control_window, 50)
        cv2.setTrackbarPos('Roll (Крен)', self.control_window, 23)  # 0 + 23

        self.update_camera_params()
        logger.info("Параметры сброшены")

    def run_display_thread(self):
        """Поток для отображения видео"""
        while self.is_running:
            if self.frame_ready.wait(timeout=0.1):
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()

                        # Добавляем информацию на кадр
                        info_text = f"FPS: {self.fps:.1f} | Yaw: {self.yaw:.1f} | Pitch: {self.pitch:.1f} | FOV: {self.fov:.1f}"
                        cv2.putText(frame, info_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Показываем кадр
                        cv2.imshow(self.window_name, frame)

                        # Обновляем панель управления
                        control_panel = self.draw_control_panel()
                        cv2.imshow(self.control_window, control_panel)

                        self.frame_ready.clear()

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    if self.loop:
                        self.loop.quit()
                elif key == ord(' '):
                    # Пауза/воспроизведение
                    state = self.pipeline.get_state(0)[1]
                    if state == Gst.State.PLAYING:
                        self.pipeline.set_state(Gst.State.PAUSED)
                        logger.info("Пауза")
                    else:
                        self.pipeline.set_state(Gst.State.PLAYING)
                        logger.info("Воспроизведение")
                elif key == ord('r'):
                    self.reset_params()
                elif key == ord('s'):
                    # Сохранить кадр
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"vcam_frame_{timestamp}.jpg"
                    with self.frame_lock:
                        if self.current_frame is not None:
                            cv2.imwrite(filename, self.current_frame)
                            logger.info(f"Кадр сохранён: {filename}")

    def run(self):
        """Запуск приложения"""
        try:
            # Создаём pipeline
            self.create_pipeline()

            # Запускаем pipeline сначала
            logger.info("Запуск pipeline...")
            self.pipeline.set_state(Gst.State.PLAYING)

            # Небольшая задержка для инициализации pipeline
            time.sleep(0.5)

            # Создаём UI после запуска pipeline
            self.create_control_ui()

            # Запускаем поток отображения
            self.is_running = True
            display_thread = threading.Thread(target=self.run_display_thread)
            display_thread.daemon = True
            display_thread.start()

            # Запускаем главный цикл GStreamer
            self.loop = GLib.MainLoop()

            logger.info("Виртуальная камера запущена")
            logger.info("Используйте ползунки для управления камерой")

            self.loop.run()

        except Exception as e:
            logger.error(f"Ошибка: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Завершение работы...")

        self.is_running = False

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        cv2.destroyAllWindows()

        logger.info("Ресурсы освобождены")


def main():
    """Точка входа"""
    if len(sys.argv) != 3:
        print("Использование: python3 test_virtual_camera_interactive.py <левое_видео> <правое_видео>")
        print("Пример: python3 test_virtual_camera_interactive.py left.mp4 right.mp4")
        sys.exit(1)

    left_video = sys.argv[1]
    right_video = sys.argv[2]

    # Проверяем существование файлов
    if not Path(left_video).exists():
        logger.error(f"Левое видео не найдено: {left_video}")
        sys.exit(1)

    if not Path(right_video).exists():
        logger.error(f"Правое видео не найдено: {right_video}")
        sys.exit(1)

    # Устанавливаем путь к плагинам
    plugin_path = "/home/nvidia/deep_cv_football/my_virt_cam/src"
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

    logger.info(f"Левое видео: {left_video}")
    logger.info(f"Правое видео: {right_video}")
    logger.info(f"GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH')}")

    try:
        # Создаём и запускаем контроллер
        controller = VirtualCameraController(left_video, right_video)
        controller.run()

    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()