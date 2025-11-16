#!/usr/bin/env python3
"""
Захват калибровочных фото с одной камеры
Для полной калибровки искажений по всему полю зрения камеры
Доска: 11x8 клеток (10x7 углов), размер клетки 23.5мм
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import sys
import threading
import time
from datetime import datetime
import json
import cv2
import numpy as np
import gc

Gst.init(None)

class SingleCameraCapture:
    def __init__(self, cam_id=0, output_dir="single_camera_data",
                 interval=3.0, max_captures=25, board_size=(10, 7), square_size=23.5):
        """
        Инициализация системы захвата для одной камеры

        Args:
            cam_id: ID камеры
            output_dir: Директория для сохранения
            interval: Интервал между захватами (сек)
            max_captures: Максимальное количество изображений
            board_size: Размер доски (внутренние углы)
            square_size: Размер клетки в мм
        """
        self.cam_id = cam_id
        self.output_dir = output_dir
        self.interval = interval
        self.max_captures = max_captures
        self.board_size = board_size
        self.square_size = square_size

        # Создаем директорию
        os.makedirs(output_dir, exist_ok=True)

        # Переменные состояния
        self.capture_count = 0
        self.latest_sample = None
        self.sample_lock = threading.Lock()
        self.last_capture_time = 0
        self.running = False

        # GStreamer элементы
        self.pipeline = None
        self.loop = None

        # Сохраняем информацию о настройке
        self.save_setup_info()

    def save_setup_info(self):
        """Сохраняет информацию о калибровочной настройке"""
        setup_info = {
            "board_size": self.board_size,
            "square_size_mm": self.square_size,
            "cells": (self.board_size[0]+1, self.board_size[1]+1),
            "capture_date": datetime.now().isoformat(),
            "camera_setup": {
                "cam_id": self.cam_id,
                "resolution": {"width": 3840, "height": 2160}
            },
            "calibration_type": "monocular",
            "purpose": "full_field_distortion_calibration"
        }

        with open(os.path.join(self.output_dir, "capture_info.json"), "w") as f:
            json.dump(setup_info, f, indent=2)

    def create_pipeline(self):
        """Создает GStreamer pipeline с превью и appsink для захвата"""
        cam_width = 3840
        cam_height = 2160
        preview_width = 1920
        preview_height = 1080

        # Pipeline с минимальным буфером для экономии памяти
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={self.cam_id} sensor-mode=0 !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},format=NV12,framerate=30/1 !
            tee name=t

            t. !
            queue max-size-buffers=2 leaky=downstream !
            nvvideoconvert !
            video/x-raw(memory:NVMM),width={preview_width},height={preview_height},format=RGBA !
            nvvideoconvert !
            nvegltransform !
            nveglglessink sync=false

            t. !
            queue max-size-buffers=2 leaky=downstream !
            nvvideoconvert !
            video/x-raw,width={cam_width},height={cam_height},format=RGBA !
            appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false
        """

        return Gst.parse_launch(pipeline_str)

    def on_new_sample(self, sink):
        """Обработчик новых кадров с камеры"""
        sample = sink.emit('pull-sample')
        if sample:
            with self.sample_lock:
                # Удаляем старый sample перед сохранением нового
                if self.latest_sample is not None:
                    del self.latest_sample
                self.latest_sample = sample
                if self.capture_count == 0:
                    print("[DEBUG] Получен первый кадр")
        return Gst.FlowReturn.OK

    def save_sample_as_jpeg(self, sample, filename):
        """Сохраняет GStreamer sample как JPEG с максимальным качеством"""
        try:
            # Получаем буфер и метаданные
            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Получаем размеры из caps
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')

            # Извлекаем данные из буфера
            result, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not result:
                return False

            # Конвертируем в numpy array
            data = np.frombuffer(mapinfo.data, dtype=np.uint8)

            # RGBA формат - 4 канала
            image = data.reshape((height, width, 4))

            # Конвертируем RGBA в BGR для OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            # Сохраняем с максимальным качеством
            cv2.imwrite(filename, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # ВАЖНО: освобождаем буфер
            buffer.unmap(mapinfo)

            # Очищаем данные
            del image_bgr
            del image
            del data

            return True

        except Exception as e:
            print(f"[ERROR] Ошибка сохранения {filename}: {e}")
            return False

    def capture_image(self):
        """Захватывает изображение"""
        with self.sample_lock:
            if self.latest_sample is None:
                print("[WARNING] Камера не готова")
                return False

            sample = self.latest_sample
            # КРИТИЧНО: обнуляем latest_sample СРАЗУ чтобы освободить ссылку
            self.latest_sample = None

        # Имя файла
        filename = os.path.join(self.output_dir, f"cam{self.cam_id}_{self.capture_count:05d}.jpg")

        # Сохраняем изображение
        success = self.save_sample_as_jpeg(sample, filename)

        # КРИТИЧНО: удаляем ссылку на sample сразу после использования
        del sample

        if success:
            self.capture_count += 1
            print(f"💾 Изображение #{self.capture_count}/{self.max_captures} сохранено: {os.path.basename(filename)}")

            # Периодическая очистка памяти каждые 10 кадров
            if self.capture_count % 10 == 0:
                gc.collect()
                print(f"   🔄 Очистка памяти")

            return True
        else:
            print("[ERROR] Ошибка сохранения")
            return False

    def auto_capture_thread(self):
        """Поток для автоматического захвата"""
        countdown_active = False

        while self.running:
            current_time = time.time()

            if self.capture_count >= self.max_captures:
                print(f"\n✅ Захвачено {self.max_captures} изображений!")
                self.running = False
                if self.loop:
                    self.loop.quit()
                break

            time_since_capture = current_time - self.last_capture_time

            if time_since_capture >= self.interval:
                # Делаем захват
                self.capture_image()
                self.last_capture_time = current_time
                countdown_active = False

                if self.capture_count < self.max_captures:
                    print(f"⏱️  Следующий захват через {self.interval:.1f} сек. Переместите доску!\n")
            elif time_since_capture >= self.interval - 3 and not countdown_active:
                # Начинаем обратный отсчет за 3 секунды
                countdown_active = True
                remaining = int(self.interval - time_since_capture)
                if remaining > 0:
                    print(f"⏰ Захват через {remaining} сек... НЕ ДВИГАЙТЕ доску!", flush=True)

            time.sleep(0.1)  # Проверяем каждые 100мс

    def start(self):
        """Запускает систему захвата"""
        print("📸 ЗАХВАТ КАЛИБРОВОЧНЫХ ИЗОБРАЖЕНИЙ (ОДНА КАМЕРА)")
        print("=" * 60)
        print(f"🎯 Калибровочная доска: {self.board_size[0]}x{self.board_size[1]} углов")
        print(f"📏 Размер клетки: {self.square_size} мм")
        print(f"📷 Камера ID: {self.cam_id}")
        print(f"🔢 Количество изображений: {self.max_captures}")
        print(f"⏱️  Интервал: {self.interval} сек")
        print(f"📁 Сохранение в: {self.output_dir}/")
        print("=" * 60)
        print("\n⚠️  ВАЖНО - Стратегия калибровки:")
        print("   • Покройте ВСЕ области кадра:")
        print("     - Углы (4 позиции)")
        print("     - Края (верх, низ, слева, справа)")
        print("     - Центр")
        print("   • Разные расстояния (близко/далеко)")
        print("   • Разные наклоны доски")
        print("   • НЕ двигайте доску во время обратного отсчета!")
        print("=" * 60)

        # Создаем и настраиваем pipeline
        print("\n🚀 Запуск pipeline...")
        self.pipeline = self.create_pipeline()

        # Подключаем обработчик
        sink = self.pipeline.get_by_name('sink')

        if not sink:
            print("[ERROR] Не найден appsink элемент")
            return False

        print("[DEBUG] Подключаем обработчик кадров...")
        sink.connect('new-sample', self.on_new_sample)

        # Устанавливаем свойства appsink
        sink.set_property('emit-signals', True)
        sink.set_property('max-buffers', 1)
        sink.set_property('drop', True)
        sink.set_property('sync', False)

        # Запускаем pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Не удалось запустить pipeline")
            return False

        # Ждем инициализации камеры
        print("⏳ Инициализация камеры...")
        time.sleep(3)

        print("\n▶️  Начинаем захват!")
        print("⏸️  Ctrl+C для остановки\n")

        # Запускаем поток автозахвата
        self.running = True
        self.last_capture_time = time.time() - self.interval + 3  # Первый кадр через 3 сек

        capture_thread = threading.Thread(target=self.auto_capture_thread, daemon=True)
        capture_thread.start()

        # Запускаем главный цикл
        self.loop = GLib.MainLoop()

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏸️  Остановка...")

        # Очистка
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)

        print(f"\n✅ Всего сохранено: {self.capture_count} изображений")
        print(f"📁 Директория: {self.output_dir}/")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Захват калибровочных изображений с одной камеры')
    parser.add_argument('--cam-id', type=int, default=0, help='ID камеры')
    parser.add_argument('--output', '-o', default='single_camera_data', help='Директория вывода')
    parser.add_argument('--count', '-n', type=int, default=25, help='Количество изображений')
    parser.add_argument('--interval', '-i', type=float, default=3.0, help='Интервал (сек)')

    args = parser.parse_args()

    capture = SingleCameraCapture(
        cam_id=args.cam_id,
        output_dir=args.output,
        interval=args.interval,
        max_captures=args.count
    )

    capture.start()

if __name__ == "__main__":
    main()
