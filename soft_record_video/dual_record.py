#!/usr/bin/env python3
"""
Простая раздельная запись двух 4K камер
Каждая камера в свой файл, без автосклейки
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import signal
import sys
import time
import threading
from datetime import datetime
import os

class DualCameraRecorder:
    def __init__(self):
        """Инициализация двойного записывателя"""
        Gst.init(None)
        
        # Параметры 4K записи
        self.width = 3840
        self.height = 2160
        self.fps = 30
        self.bitrate = 25000000  # 25 Мбит/с на камеру (можно повысить)
        
        # Пайплайны
        self.pipeline_left = None
        self.pipeline_right = None
        self.running = True
        
        # Файлы
        self.left_file = None
        self.right_file = None
        
        # Статистика
        self.start_time = None
        
    def create_camera_pipeline(self, sensor_id, output_file, codec='h264'):
        """Создает простой пайплайн для одной камеры"""
        
        if codec == 'h264':
            encoder = f"nvv4l2h264enc bitrate={self.bitrate} maxperf-enable=1 iframeinterval=30"
            parser = "h264parse"
        else:
            encoder = f"nvv4l2h265enc bitrate={self.bitrate-5000000} maxperf-enable=1 iframeinterval=30"
            parser = "h265parse"
        
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={sensor_id} !
            video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
            format=NV12, framerate={self.fps}/1 !
            nvvideoconvert ! 
            video/x-raw(memory:NVMM), format=I420 !
            {encoder} !
            {parser} ! 
            mp4mux ! 
            filesink location={output_file}
        """
        
        camera_name = "Левая" if sensor_id == 0 else "Правая"
        print(f"[INFO] 📹 {camera_name} камера -> {output_file}")
        
        pipeline = Gst.parse_launch(pipeline_str)
        
        # Подключаем обработчик сообщений
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda bus, msg: self.on_message(bus, msg, sensor_id))
        
        return pipeline
    
    def on_message(self, bus, message, sensor_id):
        """Обработчик сообщений GStreamer"""
        msg_type = message.type
        camera_name = "Левая" if sensor_id == 0 else "Правая"
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] ❌ {camera_name}: {err}")
            self.stop()
            
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] ⚠️ {camera_name}: {warn}")
            
        elif msg_type == Gst.MessageType.EOS:
            print(f"[INFO] ✅ {camera_name}: Запись завершена корректно")
            
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            element_name = message.src.get_name()
            if "nvv4l2" in element_name and new_state == Gst.State.PLAYING:
                print(f"[INFO] 🎬 {camera_name}: Запись началась!")
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        print(f"\n[INFO] 🛑 Получен сигнал {signum}, останавливаем запись...")
        self.stop()
    
    def stop(self):
        """Корректная остановка"""
        print("[INFO] ⏹️ Останавливаем записи...")
        self.running = False
        
        # Останавливаем пайплайны
        for pipeline, name in [(self.pipeline_left, "Левая"), (self.pipeline_right, "Правая")]:
            if pipeline:
                print(f"[INFO] 📹 Останавливаем {name.lower()} камеру...")
                pipeline.send_event(Gst.Event.new_eos())
                
        # Ждем немного для обработки EOS
        time.sleep(2)
        
        # Устанавливаем NULL состояние
        if self.pipeline_left:
            self.pipeline_left.set_state(Gst.State.NULL)
        if self.pipeline_right:
            self.pipeline_right.set_state(Gst.State.NULL)
        
        print("[INFO] ✅ Все записи остановлены")
    
    def run_dual_recording(self, duration_sec, codec='h264'):
        """Запуск двойной записи"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.left_file = f"camera_left_{timestamp}.mp4"
        self.right_file = f"camera_right_{timestamp}.mp4"
        
        print("🎬 ДВОЙНАЯ ЗАПИСЬ 4K КАМЕР")
        print("=" * 40)
        print(f"[INFO] 🚀 Запись двух 4K камер на {duration_sec} секунд")
        print(f"[INFO] 🎯 Кодек: {codec.upper()}")
        print(f"[INFO] 📊 Разрешение: {self.width}x{self.height} @ {self.fps}fps")
        print(f"[INFO] 💾 Битрейт: {self.bitrate//1000000} Мбит/с на камеру")
        print(f"[INFO] 📁 Левая камера: {self.left_file}")
        print(f"[INFO] 📁 Правая камера: {self.right_file}")
        print(f"[INFO] 🛑 Для остановки нажмите Ctrl+C")
        print()
        
        # Устанавливаем обработчик сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Создаем пайплайны
        try:
            print("[INFO] 🔧 Создаем пайплайны...")
            self.pipeline_left = self.create_camera_pipeline(0, self.left_file, codec)
            self.pipeline_right = self.create_camera_pipeline(1, self.right_file, codec)
        except Exception as e:
            print(f"[ERROR] ❌ Не удалось создать пайплайны: {e}")
            return False
        
        # Запускаем левую камеру
        print("[INFO] ▶️ Запускаем левую камеру...")
        ret1 = self.pipeline_left.set_state(Gst.State.PLAYING)
        if ret1 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось запустить левую камеру")
            return False
        
        # Небольшая пауза для синхронизации
        time.sleep(0.5)
        
        # Запускаем правую камеру
        print("[INFO] ▶️ Запускаем правую камеру...")
        ret2 = self.pipeline_right.set_state(Gst.State.PLAYING)
        if ret2 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось запустить правую камеру")
            self.pipeline_left.set_state(Gst.State.NULL)
            return False
        
        self.start_time = time.time()
        print()
        print("[INFO] ✅ Обе камеры запущены!")
        print("[INFO] 📽️ Запись в процессе...")
        
        # Создаем главный цикл
        loop = GLib.MainLoop()
        
        # Статистика каждые 5 секунд
        def show_progress():
            if self.running and self.start_time:
                elapsed = time.time() - self.start_time
                remaining = max(0, duration_sec - elapsed)
                print(f"[INFO] ⏱️ Прошло: {elapsed:.1f}с, осталось: {remaining:.1f}с")
            return self.running
        
        GLib.timeout_add_seconds(5, show_progress)
        
        # Таймер автостопа
        def stop_recording():
            print(f"\n[INFO] ⏰ {duration_sec} секунд записано! Останавливаем...")
            self.stop()
            loop.quit()
            return False
        
        GLib.timeout_add_seconds(duration_sec, stop_recording)
        
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] 🛑 Прерывание пользователем...")
            self.stop()
        
        # Проверяем результаты
        print("\n[INFO] 📊 Анализируем результаты...")
        return self.analyze_results(duration_sec)
    
    def analyze_results(self, expected_duration):
        """Анализирует результаты записи"""
        success = True
        total_size = 0
        
        print("[INFO] 📋 Результаты записи:")
        
        # Анализируем левый файл
        if os.path.exists(self.left_file):
            size_left = os.path.getsize(self.left_file)
            size_mb = size_left // 1024 // 1024
            total_size += size_left
            
            print(f"[INFO] 📁 Левая камера: {size_mb} МБ")
            
            if size_left < 5 * 1024 * 1024:  # меньше 5 МБ
                print("[WARNING] ⚠️ Левый файл слишком маленький")
                success = False
            else:
                # Примерный bitrate
                actual_bitrate = (size_left * 8) / expected_duration / 1000000
                print(f"[INFO] 📊 Левая: ~{actual_bitrate:.1f} Мбит/с")
        else:
            print("[ERROR] ❌ Левый файл не создан")
            success = False
        
        # Анализируем правый файл
        if os.path.exists(self.right_file):
            size_right = os.path.getsize(self.right_file)
            size_mb = size_right // 1024 // 1024
            total_size += size_right
            
            print(f"[INFO] 📁 Правая камера: {size_mb} МБ")
            
            if size_right < 5 * 1024 * 1024:  # меньше 5 МБ
                print("[WARNING] ⚠️ Правый файл слишком маленький")
                success = False
            else:
                # Примерный bitrate
                actual_bitrate = (size_right * 8) / expected_duration / 1000000
                print(f"[INFO] 📊 Правая: ~{actual_bitrate:.1f} Мбит/с")
        else:
            print("[ERROR] ❌ Правый файл не создан")
            success = False
        
        # Общая статистика
        if success:
            total_mb = total_size // 1024 // 1024
            print(f"[INFO] 📊 Общий размер: {total_mb} МБ")
            print(f"[INFO] ⏱️ Длительность: {expected_duration} секунд")
            print("[INFO] ✅ Запись завершена успешно!")
            
            # Советы по использованию
            print(f"\n[INFO] 💡 Советы по использованию:")
            print(f"• Просмотр: vlc {self.left_file} (или правый файл)")
            print(f"• Склейка: ffmpeg -i {self.left_file} -i {self.right_file} -filter_complex hstack output.mp4")
            print(f"• Конвертация: ffmpeg -i {self.left_file} -c:v libx264 -crf 18 output_compressed.mp4")
        
        return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Двойная запись 4K камер')
    parser.add_argument('--duration', type=int, default=30,
                       help='Длительность записи в секундах (по умолчанию: 30)')
    parser.add_argument('--codec', choices=['h264', 'h265'], default='h264',
                       help='Кодек для записи (по умолчанию: h264)')
    parser.add_argument('--bitrate', type=int, default=25,
                       help='Битрейт в Мбит/с (по умолчанию: 25)')
    
    args = parser.parse_args()
    
    recorder = DualCameraRecorder()
    
    # Настраиваем битрейт если указан
    if args.bitrate != 25:
        recorder.bitrate = args.bitrate * 1000000
        print(f"[INFO] 🎯 Пользовательский битрейт: {args.bitrate} Мбит/с")
    
    success = recorder.run_dual_recording(args.duration, args.codec)
    
    if not success:
        print("\n[ERROR] ❌ Запись не удалась")
        sys.exit(1)
    else:
        print("\n[INFO] 🎉 Успех! Два 4K файла готовы!")


if __name__ == "__main__":
    main()