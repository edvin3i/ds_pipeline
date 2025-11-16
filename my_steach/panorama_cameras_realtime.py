#!/usr/bin/env python3
"""
Панорама с реальных камер - объединение panorama_stream.py и camera_real_red_line.py
Используем проверенный рабочий код без лишних усложнений
"""
import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# Константы размеров панорамы
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

class PanoramaCameras:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        
    def create_pipeline(self, left_cam=0, right_cam=1, display_mode="egl"):
        """Pipeline для панорамы с камер - берем рабочий код из camera_real_red_line.py"""
        
        # Базовый pipeline с камерами как в camera_real_red_line.py
        camera_pipeline = f"""
            nvarguscamerasrc sensor-id={left_cam} !
            video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
            nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=2 !
            mux.sink_0
            
            nvarguscamerasrc sensor-id={right_cam} !
            video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
            nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=2 !
            mux.sink_1
            
            nvstreammux name=mux 
                batch-size=2 
                width=3840 
                height=2160 
                live-source=1 
                batched-push-timeout=33333 !
            
            nvdsstitch
                name=stitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !
        """
        
        # Display как в panorama_stream.py
        if display_mode == "egl":
            display_pipeline = """
                nvegltransform ! 
                nveglglessink sync=false async=false
            """
        elif display_mode == "x11":
            # Масштабируем панораму для удобного просмотра
            display_pipeline = """
                nvvideoconvert ! 
                capsfilter caps="video/x-raw(memory:NVMM),format=RGBA,width=1920,height=960" !
                nvvideoconvert ! 
                capsfilter caps="video/x-raw,format=RGBA" !
                ximagesink sync=false async=false
            """
        else:  # scale/auto
            display_pipeline = """
                nvvideoconvert ! 
                capsfilter caps="video/x-raw(memory:NVMM),format=RGBA,width=1920,height=960" !
                nvvideoconvert ! 
                capsfilter caps="video/x-raw,format=RGBA" !
                autovideosink sync=false async=false
            """
        
        full_pipeline = camera_pipeline + display_pipeline
        
        print("🎥 Создание pipeline для панорамы с камер")
        print(f"  • Левая камера: {left_cam}")
        print(f"  • Правая камера: {right_cam}")
        print(f"  • Режим отображения: {display_mode}")
        print(f"  • Выходное разрешение: 4096x2048")
        
        return Gst.parse_launch(full_pipeline)
    
    def run(self, left_cam=0, right_cam=1, display_mode="egl"):
        """Запуск pipeline - копируем логику из обоих файлов"""
        
        print("🚀 ПАНОРАМА С КАМЕР")
        print("=" * 60)
        print(f"📹 Источники:")
        print(f"   • Левая камера: {left_cam}")
        print(f"   • Правая камера: {right_cam}")
        print(f"🎯 Параметры:")
        print(f"   • Режим отображения: {display_mode}")
        print(f"   • Угол обзора: ~180-220°")
        print("-" * 60)
        
        # Проверяем элементы как в panorama_stream.py
        required_elements = ["nvarguscamerasrc", "nvstreammux", "nvdsstitch", "nvvideoconvert"]
        missing = []
        for elem in required_elements:
            if not Gst.ElementFactory.find(elem):
                missing.append(elem)
        
        if missing:
            print(f"❌ Отсутствуют элементы: {', '.join(missing)}")
            print("   Проверьте GST_PLUGIN_PATH и наличие библиотек")
            return False
        
        # Создаем pipeline
        try:
            self.pipeline = self.create_pipeline(left_cam, right_cam, display_mode)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            return False
        
        if not self.pipeline:
            print("❌ Не удалось создать pipeline")
            return False
        
        # Настройка bus как в обоих файлах
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        self.loop = GLib.MainLoop()
        
        def on_message(bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                print("\n🏁 Конец потока")
                self.loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"\n❌ Ошибка от {message.src.get_name()}: {err}")
                if debug:
                    print(f"   Debug: {debug}")
                self.loop.quit()
            elif t == Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                print(f"\n⚠️  Предупреждение: {warn}")
            elif t == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old, new, pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        print("▶️  Pipeline запущен!")
                        print("📐 Панорама отображается")
                        print("⌨️  Нажмите Ctrl+C для остановки\n")
            return True
        
        bus.connect("message", on_message)
        
        # Запуск
        print("\n🔄 Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"   Текущее состояние: {state.value_name}")
            return False
        
        # Основной цикл
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏹️  Остановлено пользователем")
        
        # Очистка
        print("\n🧹 Завершение работы...")
        self.pipeline.set_state(Gst.State.NULL)
        
        print("\n✅ Готово")
        return True


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Использование: python panorama_cameras.py [left_cam] [right_cam] [режим]")
        print("\nПараметры:")
        print("  left_cam  - ID левой камеры (по умолчанию: 0)")
        print("  right_cam - ID правой камеры (по умолчанию: 1)")
        print("  режим     - режим отображения:")
        print("            egl   - EGL отображение (по умолчанию)")
        print("            x11   - X11 отображение")
        print("            scale - Автомасштабирование")
        print("\nПримеры:")
        print("  python panorama_cameras.py")
        print("  python panorama_cameras.py 0 1")
        print("  python panorama_cameras.py 0 1 x11")
        sys.exit(0)
    
    left_cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    right_cam = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    display_mode = sys.argv[3] if len(sys.argv) > 3 else "egl"
    
    # Проверка режима
    if display_mode not in ["egl", "x11", "scale"]:
        print(f"❌ Неизвестный режим: {display_mode}")
        print("   Используйте: egl, x11 или scale")
        sys.exit(1)
    
    # Настройка окружения как в обоих файлах
    plugin_path = os.getcwd()
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"
    
    # Отладка
    os.environ['GST_DEBUG'] = os.environ.get('GST_DEBUG', '') + ',nvarguscamerasrc:3,nvdsstitch:3'
    
    print(f"📁 GST_PLUGIN_PATH: {os.environ['GST_PLUGIN_PATH']}")
    
    # Создаем и запускаем
    app = PanoramaCameras()
    
    try:
        success = app.run(left_cam, right_cam, display_mode)
        if success:
            print("\n✅ Программа завершена успешно")
            sys.exit(0)
        else:
            print("\n❌ Программа завершена с ошибками")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()