#!/usr/bin/env python3
"""
Синхронный захват калибровочных фото с двух камер для стерео калибровки
Доска: 11x8 клеток (10x7 углов), размер клетки 23.5мм
Использует appsink для точной синхронизации
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

class StereoCalibrationCapture:
    def __init__(self, cam0_id=0, cam1_id=1, output_dir="stereo_calibration_data",
                 interval=5.0, max_captures=30, cam0_max=None, cam1_max=None,
                 board_size=(10, 7), square_size=23.5):
        """
        Инициализация системы синхронного захвата для калибровки

        Args:
            cam0_id: ID первой камеры (левая)
            cam1_id: ID второй камеры (правая)
            output_dir: Директория для сохранения
            interval: Интервал между захватами (сек)
            max_captures: Максимальное количество пар (по умолчанию)
            cam0_max: Максимум кадров для камеры 0 (если None, то max_captures)
            cam1_max: Максимум кадров для камеры 1 (если None, то max_captures)
            board_size: Размер доски (внутренние углы)
            square_size: Размер клетки в мм
        """
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
        self.output_dir = output_dir
        self.interval = interval
        self.max_captures = max_captures
        self.cam0_max = cam0_max if cam0_max is not None else max_captures
        self.cam1_max = cam1_max if cam1_max is not None else max_captures
        self.board_size = board_size
        self.square_size = square_size
        
        # Создаем директории
        self.cam0_dir = os.path.join(output_dir, "cam0")
        self.cam1_dir = os.path.join(output_dir, "cam1")
        self.preview_dir = os.path.join(output_dir, "preview")
        
        for d in [self.cam0_dir, self.cam1_dir, self.preview_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Переменные состояния
        self.capture_count = 0
        self.cam0_count = 0  # Счетчик для камеры 0
        self.cam1_count = 0  # Счетчик для камеры 1
        self.latest_sample0 = None
        self.latest_sample1 = None
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
                "angle_between_cameras": 85,
                "tilt_angle": -15,
                "resolution": {"width": 3840, "height": 2160},
                "cam0_id": self.cam0_id,
                "cam1_id": self.cam1_id
            },
            "calibration_type": "stereo",
            "sync_method": "appsink"
        }
        
        with open(os.path.join(self.output_dir, "stereo_setup_info.json"), "w") as f:
            json.dump(setup_info, f, indent=2)
    
    def create_pipeline(self):
        """Создает GStreamer pipeline с appsink для синхронного захвата"""
        cam_width = 3840
        cam_height = 2160
        preview_width = 640
        preview_height = 360

        # Pipeline с preview + захват по требованию
        # Preview: маленькое разрешение, непрерывно
        # Appsink: полное разрешение, захватываем только когда нужно (раз в N секунд)
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={self.cam0_id} sensor-mode=0 !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},format=NV12,framerate=30/1 !
            tee name=t0

            t0. ! queue max-size-buffers=1 leaky=downstream !
            nvvideoconvert !
            video/x-raw(memory:NVMM),width={preview_width},height={preview_height},format=RGBA !
            nvvideoconvert ! nvegltransform !
            nveglglessink sync=false window-x=0 window-y=0

            t0. ! queue max-size-buffers=1 leaky=downstream !
            nvvideoconvert !
            video/x-raw,format=RGBA !
            appsink name=sink0 emit-signals=true max-buffers=1 drop=true sync=false

            nvarguscamerasrc sensor-id={self.cam1_id} sensor-mode=0 !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},format=NV12,framerate=30/1 !
            tee name=t1

            t1. ! queue max-size-buffers=1 leaky=downstream !
            nvvideoconvert !
            video/x-raw(memory:NVMM),width={preview_width},height={preview_height},format=RGBA !
            nvvideoconvert ! nvegltransform !
            nveglglessink sync=false window-x={preview_width} window-y=0

            t1. ! queue max-size-buffers=1 leaky=downstream !
            nvvideoconvert !
            video/x-raw,format=RGBA !
            appsink name=sink1 emit-signals=true max-buffers=1 drop=true sync=false
        """

        return Gst.parse_launch(pipeline_str)
    
    def on_new_sample_cam0(self, sink):
        """Обработчик новых кадров с камеры 0"""
        sample = sink.emit('pull-sample')
        if sample:
            with self.sample_lock:
                # Освобождаем старый sample
                if self.latest_sample0 is not None:
                    del self.latest_sample0
                self.latest_sample0 = sample
                # Отладка
                if self.capture_count == 0:
                    print("[DEBUG] Камера 0: получен первый кадр")
        return Gst.FlowReturn.OK

    def on_new_sample_cam1(self, sink):
        """Обработчик новых кадров с камеры 1"""
        sample = sink.emit('pull-sample')
        if sample:
            with self.sample_lock:
                # Освобождаем старый sample
                if self.latest_sample1 is not None:
                    del self.latest_sample1
                self.latest_sample1 = sample
                # Отладка
                if self.capture_count == 0:
                    print("[DEBUG] Камера 1: получен первый кадр")
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

            buffer.unmap(mapinfo)

            # Очищаем данные
            del image_bgr
            del image
            del data

            return True
            
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения {filename}: {e}")
            return False
    
    def capture_stereo_pair(self):
        """Захватывает синхронную пару изображений"""
        with self.sample_lock:
            if self.latest_sample0 is None or self.latest_sample1 is None:
                print("[WARNING] Камеры не готовы")
                return False

            sample0 = self.latest_sample0
            sample1 = self.latest_sample1
            # Обнуляем сразу чтобы освободить ссылки
            self.latest_sample0 = None
            self.latest_sample1 = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Имена файлов
        filename0 = os.path.join(self.cam0_dir, f"cam0_{self.capture_count:05d}.jpg")
        filename1 = os.path.join(self.cam1_dir, f"cam1_{self.capture_count:05d}.jpg")

        # Сохраняем оба изображения
        success0 = self.save_sample_as_jpeg(sample0, filename0)
        success1 = self.save_sample_as_jpeg(sample1, filename1)

        # Освобождаем samples
        del sample0
        del sample1

        # КРИТИЧНО: Агрессивная очистка памяти ПОСЛЕ КАЖДОЙ пары (две камеры!)
        gc.collect()

        if success0 and success1:
            self.capture_count += 1
            print(f"💾 Пара #{self.capture_count}/{self.max_captures} сохранена")
            return True
        else:
            print("[ERROR] Ошибка сохранения пары")
            return False
    
    def create_preview(self, file0, file1, index):
        """Создает композитное превью для проверки"""
        img0 = cv2.imread(file0)
        img1 = cv2.imread(file1)
        
        if img0 is None or img1 is None:
            return
        
        # Уменьшаем для превью
        h, w = img0.shape[:2]
        scale = 0.25  # 1/4 от оригинала
        new_w, new_h = int(w * scale), int(h * scale)
        
        img0_small = cv2.resize(img0, (new_w, new_h))
        img1_small = cv2.resize(img1, (new_w, new_h))
        
        # Объединяем горизонтально
        preview = np.hstack([img0_small, img1_small])
        
        # Добавляем текст
        cv2.putText(preview, f"Pair {index:03d}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        preview_file = os.path.join(self.preview_dir, f"preview_{index:05d}.jpg")
        cv2.imwrite(preview_file, preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    def auto_capture_thread(self):
        """Поток для автоматического захвата"""
        countdown_shown = False

        while self.running:
            current_time = time.time()
            time_since_capture = current_time - self.last_capture_time
            remaining = self.interval - time_since_capture

            if time_since_capture >= self.interval:
                if self.capture_count < self.max_captures:
                    self.capture_stereo_pair()
                    self.last_capture_time = current_time
                    countdown_shown = False

                    if self.capture_count < self.max_captures:
                        print(f"⏱️  Следующая пара через {self.interval:.0f} сек. Переместите доску!\n")
                else:
                    print(f"\n✅ Захвачено {self.max_captures} пар!")
                    self.running = False
                    if self.loop:
                        self.loop.quit()
                    break
            elif remaining <= 3 and remaining > 0 and not countdown_shown:
                # Показываем предупреждение за 3 секунды
                print(f"⏰ Захват через {int(remaining)} сек... НЕ ДВИГАЙТЕ доску!", flush=True)
                countdown_shown = True

            time.sleep(0.5)  # Проверяем каждые 500мс
    
    def start(self):
        """Запускает систему захвата"""
        print("📸 СИНХРОННЫЙ ЗАХВАТ ДЛЯ СТЕРЕО КАЛИБРОВКИ")
        print("=" * 60)
        print(f"🎯 Калибровочная доска: {self.board_size[0]}x{self.board_size[1]} углов")
        print(f"📏 Размер клетки: {self.square_size} мм")
        print(f"📷 Камеры: ID {self.cam0_id} (левая) и ID {self.cam1_id} (правая)")
        print(f"🔢 Левая: {self.cam0_max} кадров, Правая: {self.cam1_max} кадров")
        print(f"⏱️  Интервал: {self.interval} сек")
        print(f"📁 Сохранение в: {self.output_dir}/")
        print("=" * 60)
        print("\n⚠️  ВАЖНО:")
        print("   • Preview: 640x360 (низкое разрешение для экономии памяти)")
        print("   • Захват: 3840x2160 (полное разрешение раз в N сек)")
        print("   • Доска должна быть ПОЛНОСТЬЮ видна обеими камерами")
        print("   • Держите доску в зоне перекрытия")
        print("   • НЕ двигайте доску во время захвата")
        print("=" * 60)
        
        # Создаем и настраиваем pipeline
        print("\n🚀 Запуск pipeline...")
        self.pipeline = self.create_pipeline()
        
        # Подключаем обработчики
        sink0 = self.pipeline.get_by_name('sink0')
        sink1 = self.pipeline.get_by_name('sink1')
        
        if not sink0 or not sink1:
            print("[ERROR] Не найдены appsink элементы")
            return False
        
        print("[DEBUG] Подключаем обработчики кадров...")
        sink0.connect('new-sample', self.on_new_sample_cam0)
        sink1.connect('new-sample', self.on_new_sample_cam1)

        # Устанавливаем свойства appsink - МИНИМАЛЬНЫЕ буферы!
        sink0.set_property('emit-signals', True)
        sink0.set_property('max-buffers', 1)  # Минимум для экономии памяти
        sink0.set_property('drop', True)
        sink0.set_property('sync', False)

        sink1.set_property('emit-signals', True)
        sink1.set_property('max-buffers', 1)  # Минимум для экономии памяти
        sink1.set_property('drop', True)
        sink1.set_property('sync', False)
        
        # Запускаем pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Не удалось запустить pipeline")
            return False
        
        # Ждем инициализации камер
        print("⏳ Инициализация камер...")
        time.sleep(2)
        
        print("\n▶️  Начинаем захват!")
        print("⏸️  Ctrl+C для остановки\n")
        
        # Запускаем поток автозахвата
        self.running = True
        self.last_capture_time = time.time() - self.interval + 2  # Первый кадр через 2 сек
        
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
        
        # Показываем результаты
        self.show_results()
        
        return True
    
    def show_results(self):
        """Показывает результаты захвата"""
        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print("=" * 60)

        cam0_files = sorted([f for f in os.listdir(self.cam0_dir) if f.endswith('.jpg')])
        cam1_files = sorted([f for f in os.listdir(self.cam1_dir) if f.endswith('.jpg')])

        print(f"📷 Левая камера: {len(cam0_files)} файлов")
        print(f"📷 Правая камера: {len(cam1_files)} файлов")
        print(f"🖼️  Превью: {len(os.listdir(self.preview_dir))} файлов")

        # Создаем файл со списком пар
        with open(os.path.join(self.output_dir, "image_pairs.txt"), "w") as f:
            for i in range(min(len(cam0_files), len(cam1_files))):
                f.write(f"cam0/{cam0_files[i]} cam1/{cam1_files[i]}\n")

        print(f"\n📝 Список пар: {self.output_dir}/image_pairs.txt")
        print(f"📁 Все файлы: {self.output_dir}/")

        print("\n💡 Следующий шаг:")
        print("   python3 calibrate_fisheye_cameras.py")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Синхронный захват для стерео калибровки')
    parser.add_argument('--cam0', type=int, default=0, help='ID первой камеры')
    parser.add_argument('--cam1', type=int, default=1, help='ID второй камеры')
    parser.add_argument('--output', '-o', default='stereo_calibration_data', help='Директория вывода')
    parser.add_argument('--count', '-n', type=int, default=30, help='Количество пар')
    parser.add_argument('--interval', '-i', type=float, default=5.0, help='Интервал (сек)')
    
    args = parser.parse_args()
    
    if args.cam0 == args.cam1:
        print("[ERROR] ID камер должны быть разными!")
        return
    
    capture = StereoCalibrationCapture(
        cam0_id=args.cam0,
        cam1_id=args.cam1,
        output_dir=args.output,
        interval=args.interval,
        max_captures=args.count
    )
    
    capture.start()

if __name__ == "__main__":
    main()