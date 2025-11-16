#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест виртуальной камеры с визуальными ползунками на экране
Использует nvdsstitch для склейки и nvdsvirtualcam для виртуальной камеры
Отрисовка ползунков через nvdsosd
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
from pathlib import Path
import logging
import threading
import termios
import tty
import select
import math

try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    PYDS_AVAILABLE = False
    print("ПРЕДУПРЕЖДЕНИЕ: pyds не установлен, ползунки не будут отображаться")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("virtual_camera_sliders")

# Инициализация GStreamer
Gst.init(None)

# Константы размеров панорамы
PANORAMA_WIDTH = 6528
PANORAMA_HEIGHT = 1800

# Размер выходного изображения виртуальной камеры
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

class VirtualCameraController:
    """Контроллер виртуальной камеры с визуальными ползунками"""

    def __init__(self, left_file, right_file):
        self.left_file = left_file
        self.right_file = right_file

        # Параметры виртуальной камеры (согласно ограничениям плагина)
        self.yaw = 0.0      # Горизонтальный поворот (-90 до 90)
        self.pitch = 0.0    # Вертикальный поворот (-32 до 22)
        self.roll = 0.0     # Крен (-23 до 23)
        self.fov = 50.0     # Поле зрения (40 до 75)

        # Шаги изменения параметров
        self.yaw_step = 5.0
        self.pitch_step = 2.0
        self.fov_step = 5.0

        # GStreamer компоненты
        self.pipeline = None
        self.loop = None
        self.vcam_element = None
        self.is_running = False

        # Счётчики для статистики
        self.frame_count = 0
        self.start_time = None

        # Текущий активный ползунок (0-3: yaw, pitch, fov, roll)
        self.active_slider = 0

        # Параметры ползунков
        self.slider_params = [
            {'name': 'Yaw', 'value': self.yaw, 'min': -90, 'max': 90},
            {'name': 'Pitch', 'value': self.pitch, 'min': -32, 'max': 22},
            {'name': 'FOV', 'value': self.fov, 'min': 40, 'max': 75},
            {'name': 'Roll', 'value': self.roll, 'min': -23, 'max': 23}
        ]

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
            video/x-raw(memory:NVMM),format=RGBA !
            nvdsosd name=osd !
            nvegltransform !
            nveglglessink sync=false async=false
        """

        logger.info("Создание pipeline...")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Получаем элементы для управления
        self.vcam_element = self.pipeline.get_by_name("vcam")
        self.osd_element = self.pipeline.get_by_name("osd")

        # Добавляем probe для отображения информации и ползунков
        if self.osd_element:
            sink_pad = self.osd_element.get_static_pad("sink")
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.osd_probe, None)

        # Настраиваем обработку сообщений
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        logger.info("Pipeline создан успешно")

    def draw_slider(self, display_meta, index, y_pos):
        """Рисует один ползунок"""
        if not PYDS_AVAILABLE:
            return

        # ВАЖНО: Берём актуальные значения, а не из массива!
        if index == 0:
            current_value = self.yaw
            param = {'name': 'Yaw', 'value': current_value, 'min': -90, 'max': 90}
        elif index == 1:
            current_value = self.pitch
            param = {'name': 'Pitch', 'value': current_value, 'min': -32, 'max': 22}
        elif index == 2:
            current_value = self.fov
            param = {'name': 'FOV', 'value': current_value, 'min': 40, 'max': 75}
        else:
            current_value = self.roll
            param = {'name': 'Roll', 'value': current_value, 'min': -23, 'max': 23}

        # Размеры ползунка
        slider_x = 50
        slider_width = OUTPUT_WIDTH - 100
        slider_height = 30

        # Вычисляем позицию бегунка
        value_normalized = (param['value'] - param['min']) / (param['max'] - param['min'])
        handle_x = slider_x + int(value_normalized * slider_width)

        # Фон ползунка (прямоугольник)
        bg_rect_idx = index * 3  # По 3 прямоугольника на ползунок
        bg_rect = display_meta.rect_params[bg_rect_idx]
        bg_rect.left = slider_x
        bg_rect.top = y_pos
        bg_rect.width = slider_width
        bg_rect.height = slider_height
        bg_rect.has_bg_color = 1
        bg_rect.bg_color.set(0.2, 0.2, 0.2, 0.7)
        bg_rect.border_width = 2

        # Цвет рамки - подсвечиваем активный ползунок
        if index == self.active_slider:
            bg_rect.border_color.set(0.0, 1.0, 0.0, 1.0)  # Зелёный для активного
        else:
            bg_rect.border_color.set(0.5, 0.5, 0.5, 1.0)  # Серый для неактивного

        # Заполненная часть ползунка
        fill_rect_idx = bg_rect_idx + 1
        fill_rect = display_meta.rect_params[fill_rect_idx]
        fill_rect.left = slider_x
        fill_rect.top = y_pos + 5
        fill_rect.width = max(1, handle_x - slider_x)
        fill_rect.height = slider_height - 10
        fill_rect.has_bg_color = 1

        # Цвет заполнения в зависимости от параметра
        if index == 0:  # Yaw - синий
            fill_rect.bg_color.set(0.0, 0.5, 1.0, 0.8)
        elif index == 1:  # Pitch - зелёный
            fill_rect.bg_color.set(0.0, 1.0, 0.5, 0.8)
        elif index == 2:  # FOV - жёлтый
            fill_rect.bg_color.set(1.0, 1.0, 0.0, 0.8)
        else:  # Roll - фиолетовый
            fill_rect.bg_color.set(1.0, 0.0, 1.0, 0.8)

        fill_rect.border_width = 0

        # Бегунок
        handle_rect_idx = bg_rect_idx + 2
        handle_rect = display_meta.rect_params[handle_rect_idx]
        handle_rect.left = max(slider_x, min(slider_x + slider_width - 10, handle_x - 5))
        handle_rect.top = y_pos + 2
        handle_rect.width = 10
        handle_rect.height = slider_height - 4
        handle_rect.has_bg_color = 1
        handle_rect.bg_color.set(1.0, 1.0, 1.0, 1.0)
        handle_rect.border_width = 1
        handle_rect.border_color.set(0.0, 0.0, 0.0, 1.0)

        # Текст с названием и значением
        text_idx = index
        text_params = display_meta.text_params[text_idx]
        text_params.display_text = f"{param['name']}: {param['value']:.1f}°"
        text_params.x_offset = slider_x + slider_width + 10
        text_params.y_offset = y_pos + 8
        text_params.font_params.font_name = "Serif"
        text_params.font_params.font_size = 14

        if index == self.active_slider:
            text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
        else:
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

    def osd_probe(self, pad, info, user_data):
        """Probe для добавления ползунков и информации на видео"""
        try:
            if not PYDS_AVAILABLE:
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    logger.info(f"Frame: {self.frame_count} | Yaw: {self.yaw:.1f}° | "
                              f"Pitch: {self.pitch:.1f}° | FOV: {self.fov:.1f}°")
                return Gst.PadProbeReturn.OK

            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK

            # Проходим по всем фреймам в батче
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

                    # Получаем display meta
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

                    # Настраиваем количество элементов
                    display_meta.num_rects = 12  # 3 прямоугольника на каждый из 4 ползунков
                    display_meta.num_labels = 6  # 4 подписи ползунков + 2 для информации

                    # Рисуем ползунки
                    y_start = OUTPUT_HEIGHT - 200  # Начальная позиция Y для ползунков
                    for i in range(4):
                        self.draw_slider(display_meta, i, y_start + i * 45)

                    # Добавляем информацию о режиме
                    info_text_idx = 4
                    info_params = display_meta.text_params[info_text_idx]
                    info_params.display_text = f"Virtual Camera Control | Frame: {self.frame_count}"
                    info_params.x_offset = 10
                    info_params.y_offset = 30
                    info_params.font_params.font_name = "Serif"
                    info_params.font_params.font_size = 16
                    info_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                    info_params.set_bg_clr = 1
                    info_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

                    # Инструкции управления
                    help_text_idx = 5
                    help_params = display_meta.text_params[help_text_idx]
                    help_text = ("Tab - выбор ползунка | ← → - изменить значение | "
                               "R - сброс | Space - пауза | Esc - выход")
                    help_params.display_text = help_text
                    help_params.x_offset = 10
                    help_params.y_offset = OUTPUT_HEIGHT - 230
                    help_params.font_params.font_name = "Serif"
                    help_params.font_params.font_size = 12
                    help_params.font_params.font_color.set(1.0, 1.0, 1.0, 0.8)
                    help_params.set_bg_clr = 1
                    help_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)

                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                except StopIteration:
                    break

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

            self.frame_count += 1

        except Exception as e:
            logger.error(f"Ошибка в osd_probe: {e}")
            import traceback
            traceback.print_exc()

        return Gst.PadProbeReturn.OK

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
        time.sleep(0.1)
        self.pipeline.set_state(Gst.State.PLAYING)

    def update_camera_params(self):
        """Обновляет параметры виртуальной камеры"""
        if self.vcam_element:
            logger.info(f"Обновляю параметры виртуальной камеры...")

            # Устанавливаем новые значения
            self.vcam_element.set_property("yaw", self.yaw)
            self.vcam_element.set_property("pitch", self.pitch)
            self.vcam_element.set_property("roll", self.roll)
            self.vcam_element.set_property("fov", self.fov)

            # Проверяем, что значения установились
            actual_yaw = self.vcam_element.get_property("yaw")
            actual_pitch = self.vcam_element.get_property("pitch")
            actual_fov = self.vcam_element.get_property("fov")
            actual_roll = self.vcam_element.get_property("roll")

            logger.info(f"Установлено в vcam: yaw={actual_yaw:.1f}, pitch={actual_pitch:.1f}, "
                       f"fov={actual_fov:.1f}, roll={actual_roll:.1f}")

            # Обновляем значения в массиве ползунков
            self.slider_params[0]['value'] = self.yaw
            self.slider_params[1]['value'] = self.pitch
            self.slider_params[2]['value'] = self.fov
            self.slider_params[3]['value'] = self.roll

            logger.info(f"Обновлены значения ползунков: Yaw={self.yaw:.1f}° Pitch={self.pitch:.1f}° "
                       f"FOV={self.fov:.1f}° Roll={self.roll:.1f}°")
        else:
            logger.error("vcam_element не найден! Невозможно обновить параметры")

    def reset_params(self):
        """Сбрасывает параметры камеры"""
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.fov = 50.0
        self.update_camera_params()
        logger.info("Параметры сброшены")

    def handle_keyboard(self):
        """Обработка клавиатурного ввода"""
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            logger.info("\n" + "="*60)
            logger.info("УПРАВЛЕНИЕ ВИРТУАЛЬНОЙ КАМЕРОЙ С ПОЛЗУНКАМИ:")
            logger.info("  Tab     - Выбор следующего ползунка")
            logger.info("  ← / A   - Уменьшить значение")
            logger.info("  → / D   - Увеличить значение")
            logger.info("  ↑ / W   - Большой шаг вправо")
            logger.info("  ↓ / S   - Большой шаг влево")
            logger.info("  R       - Сброс всех параметров")
            logger.info("  Space   - Пауза/Воспроизведение")
            logger.info("  ESC     - Выход")
            logger.info("="*60 + "\n")

            while self.is_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)

                    # Отладка - показываем что прочитали
                    logger.debug(f"Прочитан символ: {repr(char)} (код: {ord(char) if char and len(char)==1 else 'N/A'})")

                    # Для стрелок нужно читать последовательность
                    if char == '\x1b':
                        # Проверяем, есть ли ещё символы
                        if select.select([sys.stdin], [], [], 0)[0]:
                            next_char = sys.stdin.read(1)
                            if next_char == '[':
                                # Читаем код стрелки
                                arrow_code = sys.stdin.read(1)
                                logger.debug(f"Escape последовательность: ESC[{arrow_code}")

                                if arrow_code == 'A':  # Стрелка вверх
                                    char = 'UP'
                                elif arrow_code == 'B':  # Стрелка вниз
                                    char = 'DOWN'
                                elif arrow_code == 'C':  # Стрелка вправо
                                    char = 'RIGHT'
                                elif arrow_code == 'D':  # Стрелка влево
                                    char = 'LEFT'
                            else:
                                # Не escape-последовательность, вернём символ обратно
                                char = '\x1b' + next_char
                        else:
                            # Просто ESC без последовательности
                            logger.info("ESC нажат - выход...")
                            self.is_running = False
                            if self.loop:
                                self.loop.quit()
                            break

                    logger.info(f"Обрабатываю команду: {repr(char)}")

                    # Обработка команд
                    if char == '\t':  # Tab
                        self.active_slider = (self.active_slider + 1) % 4
                        logger.info(f"Переключение на ползунок: {self.slider_params[self.active_slider]['name']}")

                    elif char in ['LEFT', 'a', 'A']:  # Уменьшить
                        param = self.slider_params[self.active_slider]
                        step = self.get_step_for_slider(self.active_slider)

                        logger.info(f"Уменьшаю {param['name']}: текущее={param['value']}, шаг={step}")

                        old_value = None
                        new_value = None

                        if self.active_slider == 0:  # Yaw
                            old_value = self.yaw
                            self.yaw = max(param['min'], self.yaw - step)
                            new_value = self.yaw
                        elif self.active_slider == 1:  # Pitch
                            old_value = self.pitch
                            self.pitch = max(param['min'], self.pitch - step)
                            new_value = self.pitch
                        elif self.active_slider == 2:  # FOV
                            old_value = self.fov
                            self.fov = max(param['min'], self.fov - step)
                            new_value = self.fov
                        elif self.active_slider == 3:  # Roll
                            old_value = self.roll
                            self.roll = max(param['min'], self.roll - step)
                            new_value = self.roll

                        logger.info(f"Изменение: {old_value} -> {new_value}")
                        self.update_camera_params()

                    elif char in ['RIGHT', 'd', 'D']:  # Увеличить
                        param = self.slider_params[self.active_slider]
                        step = self.get_step_for_slider(self.active_slider)

                        logger.info(f"Увеличиваю {param['name']}: текущее={param['value']}, шаг={step}")

                        old_value = None
                        new_value = None

                        if self.active_slider == 0:  # Yaw
                            old_value = self.yaw
                            self.yaw = min(param['max'], self.yaw + step)
                            new_value = self.yaw
                        elif self.active_slider == 1:  # Pitch
                            old_value = self.pitch
                            self.pitch = min(param['max'], self.pitch + step)
                            new_value = self.pitch
                        elif self.active_slider == 2:  # FOV
                            old_value = self.fov
                            self.fov = min(param['max'], self.fov + step)
                            new_value = self.fov
                        elif self.active_slider == 3:  # Roll
                            old_value = self.roll
                            self.roll = min(param['max'], self.roll + step)
                            new_value = self.roll

                        logger.info(f"Изменение: {old_value} -> {new_value}")
                        self.update_camera_params()

                    elif char in ['UP', 'w', 'W']:  # Большой шаг вправо
                        param = self.slider_params[self.active_slider]
                        step = self.get_step_for_slider(self.active_slider) * 3

                        if self.active_slider == 0:
                            self.yaw = min(param['max'], self.yaw + step)
                        elif self.active_slider == 1:
                            self.pitch = min(param['max'], self.pitch + step)
                        elif self.active_slider == 2:
                            self.fov = min(param['max'], self.fov + step)
                        elif self.active_slider == 3:
                            self.roll = min(param['max'], self.roll + step)

                        self.update_camera_params()

                    elif char in ['DOWN', 's', 'S']:  # Большой шаг влево
                        param = self.slider_params[self.active_slider]
                        step = self.get_step_for_slider(self.active_slider) * 3

                        if self.active_slider == 0:
                            self.yaw = max(param['min'], self.yaw - step)
                        elif self.active_slider == 1:
                            self.pitch = max(param['min'], self.pitch - step)
                        elif self.active_slider == 2:
                            self.fov = max(param['min'], self.fov - step)
                        elif self.active_slider == 3:
                            self.roll = max(param['min'], self.roll - step)

                        self.update_camera_params()

                    elif char in ['r', 'R']:  # Reset
                        self.reset_params()

                    elif char == ' ':  # Пауза
                        state = self.pipeline.get_state(0)[1]
                        if state == Gst.State.PLAYING:
                            self.pipeline.set_state(Gst.State.PAUSED)
                            logger.info("ПАУЗА")
                        else:
                            self.pipeline.set_state(Gst.State.PLAYING)
                            logger.info("ВОСПРОИЗВЕДЕНИЕ")

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_step_for_slider(self, index):
        """Возвращает шаг изменения для конкретного ползунка"""
        if index == 0:  # Yaw
            return 5.0
        elif index == 1:  # Pitch
            return 2.0
        elif index == 2:  # FOV
            return 2.0
        elif index == 3:  # Roll
            return 2.0
        return 1.0

    def run(self):
        """Запуск приложения"""
        try:
            # Создаём pipeline
            self.create_pipeline()

            # Запускаем поток для обработки клавиатуры
            self.is_running = True
            keyboard_thread = threading.Thread(target=self.handle_keyboard)
            keyboard_thread.daemon = True
            keyboard_thread.start()

            # Запускаем pipeline
            logger.info("Запуск pipeline...")
            self.pipeline.set_state(Gst.State.PLAYING)

            # Запускаем главный цикл GStreamer
            self.loop = GLib.MainLoop()
            self.start_time = time.time()

            logger.info("Виртуальная камера с ползунками запущена")
            logger.info("Используйте клавиши для управления")

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

        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Статистика: {self.frame_count} кадров за {elapsed:.1f}с (FPS: {fps:.1f})")

        logger.info("Ресурсы освобождены")


def main():
    """Точка входа"""
    if len(sys.argv) != 3:
        print("Использование: python3 test_virtual_camera_sliders.py <левое_видео> <правое_видео>")
        print("Пример: python3 test_virtual_camera_sliders.py left.mp4 right.mp4")
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