#!/usr/bin/env python3
"""
Раздельная запись двух 4K камер с ФИКСИРОВАННОЙ экспозицией
Отключена автоматическая подстройка яркости
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

class FixedExposureDualRecorder:
    def __init__(self):
        """Инициализация двойного записывателя с фиксированной экспозицией"""
        Gst.init(None)
        
        # Параметры 4K записи
        self.width = 3840
        self.height = 2160
        self.fps = 30
        self.bitrate = 25000000  # 25 Мбит/с на камеру
        
        # ФИКСИРОВАННЫЕ параметры экспозиции
        # Время экспозиции в наносекундах (1/60 сек = 16666666 нс)
        self.exposure_time = 16666666  # ~1/60 секунды
        self.gain = 8.0  # Усиление (1-16)
        self.isp_digital_gain = 2.0  # Цифровое усиление (1-256)
        
        # Можно настроить разные параметры для каждой камеры
        self.camera_settings = {
            0: {  # Левая камера
                "exposure": 16666666,  # 1/60 сек
                "gain": 8.0,
                "digital_gain": 2.0,
                "name": "Левая"
            },
            1: {  # Правая камера
                "exposure": 16666666,  # 1/60 сек
                "gain": 8.0,
                "digital_gain": 2.0,
                "name": "Правая"
            }
        }
        
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
        """Создает пайплайн с ФИКСИРОВАННОЙ экспозицией"""
        
        settings = self.camera_settings[sensor_id]
        
        if codec == 'h264':
            encoder = f"nvv4l2h264enc bitrate={self.bitrate} maxperf-enable=1 iframeinterval=30"
            parser = "h264parse"
        else:
            encoder = f"nvv4l2h265enc bitrate={self.bitrate-5000000} maxperf-enable=1 iframeinterval=30"
            parser = "h265parse"
        
        # ВАЖНО: aelock=true отключает автоэкспозицию!
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={sensor_id}
                aelock=true
                awblock=true
                exposuretimerange="{settings['exposure']} {settings['exposure']}"
                gainrange="{settings['gain']} {settings['gain']}"
                ispdigitalgainrange="{settings['digital_gain']} {settings['digital_gain']}"
                saturation=1.2 !
            video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
            format=NV12, framerate={self.fps}/1 !
            nvvideoconvert ! 
            video/x-raw(memory:NVMM), format=I420 !
            {encoder} !
            {parser} ! 
            mp4mux ! 
            filesink location={output_file}
        """
        
        camera_name = settings['name']
        print(f"[INFO] 📹 {camera_name} камера -> {output_file}")
        print(f"[INFO] 🔒 Фиксированная экспозиция: {settings['exposure']/1000000:.1f}мс, "
              f"Gain: {settings['gain']}, Digital: {settings['digital_gain']}")
        
        pipeline = Gst.parse_launch(pipeline_str)
        
        # Подключаем обработчик сообщений
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda bus, msg: self.on_message(bus, msg, sensor_id))
        
        return pipeline
    
    def adjust_camera_settings(self, sensor_id, exposure_ms=None, gain=None, digital_gain=None):
        """Изменить настройки конкретной камеры"""
        settings = self.camera_settings[sensor_id]
        
        if exposure_ms is not None:
            settings['exposure'] = int(exposure_ms * 1000000)  # мс в наносекунды
            print(f"[INFO] 📸 {settings['name']}: Экспозиция = {exposure_ms}мс")
        
        if gain is not None:
            settings['gain'] = max(1.0, min(16.0, gain))
            print(f"[INFO] 📸 {settings['name']}: Gain = {settings['gain']}")
        
        if digital_gain is not None:
            settings['digital_gain'] = max(1.0, min(256.0, digital_gain))
            print(f"[INFO] 📸 {settings['name']}: Digital Gain = {settings['digital_gain']}")
    
    def set_preset(self, preset):
        """Применить предустановленные настройки"""
        presets = {
            "daylight": {  # Яркий день
                "exposure_ms": 8.3,   # 1/120 сек
                "gain": 1.0,
                "digital_gain": 1.0
            },
            "cloudy": {    # Облачно
                "exposure_ms": 16.6,  # 1/60 сек
                "gain": 4.0,
                "digital_gain": 1.5
            },
            "indoor": {    # В помещении
                "exposure_ms": 33.3,  # 1/30 сек
                "gain": 8.0,
                "digital_gain": 2.0
            },
            "lowlight": {  # Сумерки
                "exposure_ms": 33.3,  # 1/30 сек
                "gain": 12.0,
                "digital_gain": 4.0
            }
        }
        
        if preset in presets:
            p = presets[preset]
            print(f"\n[INFO] 🎨 Применяем пресет: {preset}")
            
            # Применяем ко всем камерам
            for sensor_id in [0, 1]:
                self.adjust_camera_settings(
                    sensor_id,
                    exposure_ms=p["exposure_ms"],
                    gain=p["gain"],
                    digital_gain=p["digital_gain"]
                )
        else:
            print(f"[ERROR] ❌ Неизвестный пресет: {preset}")
            print(f"[INFO] Доступные пресеты: {', '.join(presets.keys())}")
    
    def on_message(self, bus, message, sensor_id):
        """Обработчик сообщений GStreamer"""
        msg_type = message.type
        camera_name = self.camera_settings[sensor_id]['name']
        
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
                print(f"[INFO] 🎬 {camera_name}: Запись началась (фиксированная экспозиция)!")
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        print(f"\n[INFO] 🛑 Получен сигнал {signum}, останавливаем запись...")
        self.stop()
    
    def stop(self):
        """Корректная остановка"""
        print("[INFO] ⏹️ Останавливаем записи...")
        self.running = False
        
        # Останавливаем пайплайны
        for pipeline, sensor_id in [(self.pipeline_left, 0), (self.pipeline_right, 1)]:
            if pipeline:
                name = self.camera_settings[sensor_id]['name']
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
        
        # Вызываем callback для остановки loop
        if hasattr(self, 'stop_callback') and self.stop_callback:
            GLib.idle_add(self.stop_callback)
    
    def run_continuous_recording(self, codec='h264', preset=None):
        """Запуск непрерывной записи с фиксированной экспозицией"""
        
        # Применяем пресет если указан
        if preset:
            self.set_preset(preset)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.left_file = f"camera_left_fixed_{timestamp}.mp4"
        self.right_file = f"camera_right_fixed_{timestamp}.mp4"
        
        print("\n🎬 ЗАПИСЬ С ФИКСИРОВАННОЙ ЭКСПОЗИЦИЕЙ")
        print("=" * 50)
        print(f"[INFO] 🚀 Запись двух 4K камер БЕЗ автоэкспозиции")
        print(f"[INFO] 🎯 Кодек: {codec.upper()}")
        print(f"[INFO] 📊 Разрешение: {self.width}x{self.height} @ {self.fps}fps")
        print(f"[INFO] 💾 Битрейт: {self.bitrate//1000000} Мбит/с на камеру")
        print(f"[INFO] 🔒 Автоэкспозиция: ОТКЛЮЧЕНА")
        print(f"[INFO] 📁 Левая камера: {self.left_file}")
        print(f"[INFO] 📁 Правая камера: {self.right_file}")
        print(f"[INFO] 🛑 Для остановки нажмите Ctrl+C")
        print()
        
        # Устанавливаем обработчик сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Создаем пайплайны
        try:
            print("[INFO] 🔧 Создаем пайплайны с фиксированной экспозицией...")
            self.pipeline_left = self.create_camera_pipeline(0, self.left_file, codec)
            self.pipeline_right = self.create_camera_pipeline(1, self.right_file, codec)
        except Exception as e:
            print(f"[ERROR] ❌ Не удалось создать пайплайны: {e}")
            return False
        
        # Запускаем камеры
        print("\n[INFO] ▶️ Запускаем камеры...")
        ret1 = self.pipeline_left.set_state(Gst.State.PLAYING)
        if ret1 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось запустить левую камеру")
            return False
        
        time.sleep(0.5)
        
        ret2 = self.pipeline_right.set_state(Gst.State.PLAYING)
        if ret2 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось запустить правую камеру")
            self.pipeline_left.set_state(Gst.State.NULL)
            return False
        
        self.start_time = time.time()
        print("\n[INFO] ✅ Обе камеры запущены с ФИКСИРОВАННОЙ экспозицией!")
        print("[INFO] 💡 Яркость НЕ будет меняться автоматически")
        print("[INFO] 📽️ Запись в процессе...")
        
        # Создаем главный цикл
        loop = GLib.MainLoop()
        
        # Статистика каждые 10 секунд
        def show_progress():
            if self.running and self.start_time:
                elapsed = time.time() - self.start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                
                # Проверяем размеры файлов
                size_info = ""
                if os.path.exists(self.left_file):
                    size_left_mb = os.path.getsize(self.left_file) // 1024 // 1024
                    size_info += f"Л: {size_left_mb}МБ"
                
                if os.path.exists(self.right_file):
                    size_right_mb = os.path.getsize(self.right_file) // 1024 // 1024
                    if size_info:
                        size_info += ", "
                    size_info += f"П: {size_right_mb}МБ"
                
                print(f"[INFO] ⏱️ Время: {minutes:02d}:{seconds:02d} | {size_info} | 🔒 Fixed exposure")
            return self.running
        
        GLib.timeout_add_seconds(10, show_progress)
        
        # Обработчик для корректной остановки loop
        def stop_loop():
            loop.quit()
            return False
        
        self.stop_callback = stop_loop
        
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] 🛑 Прерывание пользователем...")
            self.stop()
        
        # Анализируем результаты
        print("\n[INFO] 📊 Анализируем результаты...")
        return self.analyze_results()
    
    def analyze_results(self):
        """Анализирует результаты записи"""
        success = True
        total_size = 0
        
        # Вычисляем длительность
        if self.start_time:
            actual_duration = time.time() - self.start_time
        else:
            actual_duration = 0
        
        minutes = int(actual_duration // 60)
        seconds = int(actual_duration % 60)
        
        print("[INFO] 📋 Результаты записи с фиксированной экспозицией:")
        print(f"[INFO] ⏱️ Общая длительность: {minutes:02d}:{seconds:02d}")
        
        # Показываем использованные настройки
        print("\n[INFO] 📸 Использованные настройки:")
        for sensor_id, settings in self.camera_settings.items():
            print(f"[INFO] {settings['name']}: "
                  f"Экспозиция={settings['exposure']/1000000:.1f}мс, "
                  f"Gain={settings['gain']}, "
                  f"Digital={settings['digital_gain']}")
        
        # Анализируем файлы
        for file_path, name in [(self.left_file, "Левая"), (self.right_file, "Правая")]:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_mb = size // 1024 // 1024
                total_size += size
                
                print(f"\n[INFO] 📁 {name} камера: {size_mb} МБ")
                
                if size < 5 * 1024 * 1024:
                    print(f"[WARNING] ⚠️ {name} файл слишком маленький")
                    success = False
                elif actual_duration > 0:
                    actual_bitrate = (size * 8) / actual_duration / 1000000
                    print(f"[INFO] 📊 Битрейт: ~{actual_bitrate:.1f} Мбит/с")
            else:
                print(f"[ERROR] ❌ {name} файл не создан")
                success = False
        
        if success:
            total_mb = total_size // 1024 // 1024
            total_gb = total_mb / 1024
            print(f"\n[INFO] 📊 Общий размер: {total_mb} МБ ({total_gb:.2f} ГБ)")
            print("[INFO] ✅ Запись с фиксированной экспозицией завершена успешно!")
        
        return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Запись 4K камер с ФИКСИРОВАННОЙ экспозицией',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Запись с пресетом для яркого дня
  %(prog)s --preset daylight
  
  # Запись с пресетом для помещения
  %(prog)s --preset indoor
  
  # Ручная настройка параметров
  %(prog)s --exposure 16.6 --gain 4 --digital 2
  
  # Разные настройки для каждой камеры
  %(prog)s --left-exposure 8.3 --left-gain 2 --right-exposure 33.3 --right-gain 8
        """
    )
    
    parser.add_argument('--codec', choices=['h264', 'h265'], default='h264',
                       help='Кодек для записи (по умолчанию: h264)')
    parser.add_argument('--bitrate', type=int, default=25,
                       help='Битрейт в Мбит/с (по умолчанию: 25)')
    
    # Пресеты
    parser.add_argument('--preset', choices=['daylight', 'cloudy', 'indoor', 'lowlight'],
                       help='Использовать пресет настроек')
    
    # Общие настройки для обеих камер
    parser.add_argument('--exposure', type=float,
                       help='Время экспозиции в мс (например: 16.6 для 1/60)')
    parser.add_argument('--gain', type=float,
                       help='Аналоговое усиление (1-16)')
    parser.add_argument('--digital', type=float,
                       help='Цифровое усиление (1-256)')
    
    # Индивидуальные настройки
    parser.add_argument('--left-exposure', type=float,
                       help='Экспозиция левой камеры (мс)')
    parser.add_argument('--left-gain', type=float,
                       help='Gain левой камеры')
    parser.add_argument('--right-exposure', type=float,
                       help='Экспозиция правой камеры (мс)')
    parser.add_argument('--right-gain', type=float,
                       help='Gain правой камеры')
    
    args = parser.parse_args()
    
    recorder = FixedExposureDualRecorder()
    
    # Настраиваем битрейт
    if args.bitrate != 25:
        recorder.bitrate = args.bitrate * 1000000
        print(f"[INFO] 🎯 Битрейт: {args.bitrate} Мбит/с")
    
    # Применяем общие настройки
    if args.exposure or args.gain or args.digital:
        print("[INFO] 🔧 Применяем пользовательские настройки...")
        for sensor_id in [0, 1]:
            recorder.adjust_camera_settings(
                sensor_id,
                exposure_ms=args.exposure,
                gain=args.gain,
                digital_gain=args.digital
            )
    
    # Применяем индивидуальные настройки
    if args.left_exposure or args.left_gain:
        recorder.adjust_camera_settings(
            0,
            exposure_ms=args.left_exposure,
            gain=args.left_gain
        )
    
    if args.right_exposure or args.right_gain:
        recorder.adjust_camera_settings(
            1,
            exposure_ms=args.right_exposure,
            gain=args.right_gain
        )
    
    # Запускаем запись
    success = recorder.run_continuous_recording(args.codec, args.preset)
    
    if not success:
        print("\n[ERROR] ❌ Запись не удалась")
        sys.exit(1)
    else:
        print("\n[INFO] 🎉 Успех! Записи с фиксированной экспозицией готовы!")


if __name__ == "__main__":
    main()