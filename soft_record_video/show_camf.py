#!/usr/bin/env python3
"""
Просмотр камеры с возможностью кропа центра для настройки фокуса
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import signal
import sys

class CameraFocusViewer:
    def __init__(self):
        """Инициализация просмотрщика для настройки фокуса"""
        Gst.init(None)
        
        # Максимальное разрешение для 4K камеры
        self.width = 3840
        self.height = 2160
        self.fps = 30
        
        self.pipeline = None
        self.loop = None
        
    def create_pipeline(self, sensor_id=0, crop_zoom=1.0):
        """
        Создает pipeline с возможностью кропа центра
        crop_zoom: коэффициент увеличения (1.0 = без кропа, 2.0 = 2x zoom, и т.д.)
        """
        
        if crop_zoom > 1.0:
            # Вычисляем размеры кропа
            crop_width = int(self.width / crop_zoom)
            crop_height = int(self.height / crop_zoom)
            
            # Центрируем кроп
            crop_left = (self.width - crop_width) // 2
            crop_top = (self.height - crop_height) // 2
            
            # Pipeline с кропом через nvvideoconvert
            pipeline_str = f"""
                nvarguscamerasrc sensor-id={sensor_id} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
                format=NV12, framerate={self.fps}/1 !
                nvvideoconvert 
                    src-crop={crop_left}:{crop_top}:{crop_width}:{crop_height} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12 !
                nvegltransform !
                nveglglessink sync=false
            """
            
            print(f"[INFO] 🔍 Режим КРОПА: {crop_zoom}x увеличение")
            print(f"[INFO] 📐 Кроп: {crop_width}x{crop_height} из центра")
            print(f"[INFO] 📍 Позиция кропа: left={crop_left}, top={crop_top}")
            
        else:
            # Обычный pipeline без кропа
            pipeline_str = f"""
                nvarguscamerasrc sensor-id={sensor_id} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
                format=NV12, framerate={self.fps}/1 !
                nvvideoconvert !
                nvegltransform !
                nveglglessink sync=false
            """
            
            print(f"[INFO] 📹 Полный кадр (без кропа)")
        
        print(f"[INFO] 📹 Камера #{sensor_id}")
        print(f"[INFO] 📊 Исходное разрешение: {self.width}x{self.height} @ {self.fps}fps")
        
        return Gst.parse_launch(pipeline_str)
    
    def create_grid_overlay_pipeline(self, sensor_id=0, crop_zoom=1.0):
        """Pipeline с альтернативным кропом для Jetson Orin"""
        
        if crop_zoom > 1.0:
            crop_width = int(self.width / crop_zoom)
            crop_height = int(self.height / crop_zoom)
            crop_left = (self.width - crop_width) // 2
            crop_top = (self.height - crop_height) // 2
            
            # Используем nvvideoconvert с параметрами кропа
            pipeline_str = f"""
                nvarguscamerasrc sensor-id={sensor_id} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
                format=NV12, framerate={self.fps}/1 !
                nvvideoconvert 
                    src-crop={crop_left}:{crop_top}:{crop_width}:{crop_height} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12 !
                nvegltransform !
                nveglglessink sync=false
            """
        else:
            pipeline_str = f"""
                nvarguscamerasrc sensor-id={sensor_id} !
                video/x-raw(memory:NVMM), width={self.width}, height={self.height}, 
                format=NV12, framerate={self.fps}/1 !
                nvvideoconvert !
                nvegltransform !
                nveglglessink sync=false
            """
        
        return Gst.parse_launch(pipeline_str)
    
    def on_message(self, bus, message):
        """Обработчик сообщений GStreamer"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] ❌ {err}")
            print(f"[DEBUG] {debug}")
            self.stop()
            
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] ⚠️ {warn}")
            
        elif msg_type == Gst.MessageType.EOS:
            print("[INFO] ✅ Поток завершён")
            self.stop()
            
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    print("[INFO] ▶️ Трансляция началась!")
                    print("[INFO] 🛑 Нажмите Ctrl+C для остановки\n")
        
        return True
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        print("\n[INFO] 🛑 Получен сигнал, останавливаем...")
        self.stop()
    
    def stop(self):
        """Корректная остановка"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        print("[INFO] ✅ Трансляция остановлена")
    
    def run(self, sensor_id=0, crop_zoom=1.0, show_grid=False):
        """Запуск трансляции"""
        print("🎥 НАСТРОЙКА ФОКУСА КАМЕРЫ")
        print("=" * 60)
        
        if crop_zoom > 1.0:
            print(f"[INFO] 🔍 РЕЖИМ УВЕЛИЧЕНИЯ ЦЕНТРА: {crop_zoom}x")
            print("[INFO] 💡 Это поможет точнее настроить фокус объектива")
        
        # Устанавливаем обработчик сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Создаем pipeline
        try:
            if show_grid:
                self.pipeline = self.create_grid_overlay_pipeline(sensor_id, crop_zoom)
            else:
                self.pipeline = self.create_pipeline(sensor_id, crop_zoom)
        except Exception as e:
            print(f"[ERROR] ❌ Не удалось создать pipeline: {e}")
            return False
        
        # Настраиваем bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        
        # Создаем главный цикл
        self.loop = GLib.MainLoop()
        
        # Если нужна сетка, добавляем её
        if show_grid and self.pipeline.get_by_name("overlay"):
            overlay = self.pipeline.get_by_name("overlay")
            # Создаем простую сетку из символов
            grid_text = "+" + "-"*20 + "+" + "-"*20 + "+"
            overlay.set_property("text", grid_text)
        
        # Запускаем pipeline
        print("[INFO] 🚀 Запускаем трансляцию...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось запустить pipeline")
            return False
        
        print("\n[INFO] 📌 СОВЕТЫ ПО НАСТРОЙКЕ ФОКУСА:")
        if crop_zoom >= 4.0:
            print("   🎯 РЕЖИМ ДЛЯ ДАЛЬНЕЙ ФОКУСИРОВКИ (10-50м):")
            print("   • Найдите контрастный объект на нужном расстоянии")
            print("   • Используйте края зданий, вывески, антенны")
            print("   • В солнечную погоду детали видны лучше")
            print("   • Фокус на бесконечность обычно НЕ самый дальний!")
        else:
            print("   • Для дальних объектов используйте --zoom 4 или больше")
            print("   • Для близких: текст, мелкие детали")
        print("   • ОЧЕНЬ медленно вращайте кольцо фокусировки")
        print("   • Можете пройти точку фокуса и вернуться")
        if crop_zoom > 1.0:
            print(f"   • Увеличение {crop_zoom}x покажет мелкие детали")
        print()
        
        # Запускаем цикл
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] 🛑 Прерывание пользователем...")
            self.stop()
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Настройка фокуса камеры с увеличением центра')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID камеры (по умолчанию: 0)')
    parser.add_argument('--zoom', type=float, default=1.0,
                       help='Коэффициент увеличения центра (1.0 = без увеличения, 2.0 = 2x, 4.0 = 4x)')
    parser.add_argument('--grid', action='store_true',
                       help='Показать сетку для выравнивания')
    
    args = parser.parse_args()
    
    # Проверяем zoom
    if args.zoom < 1.0:
        print("[ERROR] ❌ Zoom должен быть >= 1.0")
        sys.exit(1)
    
    if args.zoom > 10.0:
        print("[WARNING] ⚠️ Очень большой zoom может снизить качество")
    
    viewer = CameraFocusViewer()
    
    # Запускаем трансляцию
    success = viewer.run(args.camera, args.zoom, args.grid)
    
    if not success:
        print("\n[ERROR] ❌ Не удалось запустить трансляцию")
        sys.exit(1)


if __name__ == "__main__":
    print("\n📌 УТИЛИТА ДЛЯ НАСТРОЙКИ ФОКУСА")
    print("=" * 60)
    print("Использование:")
    print("   python3 camera_focus.py [--camera ID] [--zoom FACTOR] [--grid]")
    print("\nПримеры:")
    print("   python3 camera_focus.py                    # Полный кадр с камеры 0")
    print("   python3 camera_focus.py --zoom 2           # 2x увеличение центра")
    print("   python3 camera_focus.py --zoom 4           # 4x увеличение центра")
    print("   python3 camera_focus.py --camera 1 --zoom 3  # Камера 1, 3x увеличение")
    print("   python3 camera_focus.py --zoom 2 --grid    # С сеткой для выравнивания")
    print("\n💡 Рекомендуется начать с --zoom 2 или --zoom 4 для точной настройки")
    print()
    
    main()