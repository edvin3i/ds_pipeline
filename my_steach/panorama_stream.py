#!/usr/bin/env python3
"""
Панорама 360° из двух видеофайлов
БЕЗ nvmultiurisrcbin - простой и надёжный pipeline
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

class PanoramaStream:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        
    def create_simple_pipeline(self, left_file, right_file, display_mode="egl"):
        """Создает простой pipeline без nvmultiurisrcbin"""
        
        # Pipeline строка - простая и понятная
        pipeline_str = f"""
            filesrc location={left_file} ! 
            qtdemux ! h264parse ! nvv4l2decoder name=dec0 ! 
            nvvideoconvert ! 
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_0
            
            filesrc location={right_file} ! 
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
            
            nvdsstitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !
            
            queue max-size-buffers=3 !
        """
        
        # Добавляем вывод в зависимости от режима
        if display_mode == "egl":
            pipeline_str += """
                nvegltransform ! 
                nveglglessink sync=false async=false
            """
        elif display_mode == "x11":
            # Масштабируем для удобного просмотра
            pipeline_str += """
                nvvideoconvert ! 
                video/x-raw(memory:NVMM),format=RGBA,width=1920,height=960 !
                nvvideoconvert ! 
                video/x-raw,format=RGBA !
                ximagesink sync=false
            """
        elif display_mode == "file":
            # Сохранение в файл
            pipeline_str += """
                nvvideoconvert ! 
                nvv4l2h264enc bitrate=20000000 !
                h264parse !
                qtmux !
                filesink location=panorama_output.mp4
            """
        else:  # auto
            pipeline_str += """
                nvvideoconvert ! 
                video/x-raw(memory:NVMM),format=RGBA,width=1920,height=960 !
                nvvideoconvert ! 
                video/x-raw,format=RGBA !
                autovideosink sync=false
            """
        
        print("📋 Pipeline создан (простая версия без nvmultiurisrcbin)")
        return Gst.parse_launch(pipeline_str)
    
    def create_advanced_pipeline(self, left_file, right_file, display_mode="egl", loop=True):
        """Создает продвинутый pipeline с программным управлением"""
        
        # Создаём pipeline вручную для полного контроля
        pipeline = Gst.Pipeline.new("panorama-pipeline")
        
        # === ЛЕВЫЙ ИСТОЧНИК ===
        src_left = Gst.ElementFactory.make("filesrc", "src_left")
        src_left.set_property("location", left_file)
        
        demux_left = Gst.ElementFactory.make("qtdemux", "demux_left")
        parse_left = Gst.ElementFactory.make("h264parse", "parse_left")
        parse_left.set_property("config-interval", -1)  # Форсировать SPS/PPS
        
        decode_left = Gst.ElementFactory.make("nvv4l2decoder", "decode_left")
        decode_left.set_property("enable-max-performance", True)
        
        convert_left = Gst.ElementFactory.make("nvvideoconvert", "convert_left")
        
        queue_left = Gst.ElementFactory.make("queue", "queue_left")
        queue_left.set_property("max-size-buffers", 5)
        queue_left.set_property("max-size-time", 0)
        queue_left.set_property("max-size-bytes", 0)
        
        # === ПРАВЫЙ ИСТОЧНИК ===
        src_right = Gst.ElementFactory.make("filesrc", "src_right")
        src_right.set_property("location", right_file)
        
        demux_right = Gst.ElementFactory.make("qtdemux", "demux_right")
        parse_right = Gst.ElementFactory.make("h264parse", "parse_right")
        parse_right.set_property("config-interval", -1)
        
        decode_right = Gst.ElementFactory.make("nvv4l2decoder", "decode_right")
        decode_right.set_property("enable-max-performance", True)
        
        convert_right = Gst.ElementFactory.make("nvvideoconvert", "convert_right")
        
        queue_right = Gst.ElementFactory.make("queue", "queue_right")
        queue_right.set_property("max-size-buffers", 5)
        queue_right.set_property("max-size-time", 0)
        queue_right.set_property("max-size-bytes", 0)
        
        # === STREAMMUX ===
        streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        streammux.set_property("batch-size", 2)
        streammux.set_property("width", 3840)
        streammux.set_property("height", 2160)
        streammux.set_property("batched-push-timeout", 40000)
        streammux.set_property("live-source", 0)
        
        # === STITCH ===
        stitch = Gst.ElementFactory.make("nvdsstitch", "stitch")
        stitch.set_property("left-source-id", 0)
        stitch.set_property("right-source-id", 1)
        stitch.set_property("gpu-id", 0)
        stitch.set_property("panorama-width", PANORAMA_WIDTH)
        stitch.set_property("panorama-height", PANORAMA_HEIGHT)
        
        # Queue после stitch
        queue_out = Gst.ElementFactory.make("queue", "queue_out")
        queue_out.set_property("max-size-buffers", 3)
        
        # === DISPLAY ===
        if display_mode == "egl":
            transform = Gst.ElementFactory.make("nvegltransform", "transform")
            sink = Gst.ElementFactory.make("nveglglessink", "sink")
        elif display_mode == "x11":
            converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
            scaler = Gst.ElementFactory.make("nvvideoconvert", "scaler")
            caps_nvmm = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA,width=1920,height=960")
            filter_nvmm = Gst.ElementFactory.make("capsfilter", "filter_nvmm")
            filter_nvmm.set_property("caps", caps_nvmm)
            
            caps_raw = Gst.Caps.from_string("video/x-raw,format=RGBA")
            filter_raw = Gst.ElementFactory.make("capsfilter", "filter_raw")
            filter_raw.set_property("caps", caps_raw)
            
            sink = Gst.ElementFactory.make("ximagesink", "sink")
        else:
            converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
            caps = Gst.Caps.from_string("video/x-raw,format=RGBA")
            filter = Gst.ElementFactory.make("capsfilter", "filter")
            filter.set_property("caps", caps)
            sink = Gst.ElementFactory.make("autovideosink", "sink")
        
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Добавляем все элементы в pipeline
        elements = [
            src_left, demux_left, parse_left, decode_left, convert_left, queue_left,
            src_right, demux_right, parse_right, decode_right, convert_right, queue_right,
            streammux, stitch, queue_out
        ]
        
        if display_mode == "egl":
            elements.extend([transform, sink])
        elif display_mode == "x11":
            elements.extend([converter, filter_nvmm, scaler, filter_raw, sink])
        else:
            elements.extend([converter, filter, sink])
        
        for element in elements:
            if not element:
                print(f"❌ Не удалось создать элемент")
                return None
            pipeline.add(element)
        
        # === СВЯЗЫВАНИЕ ===
        # Левая ветка
        src_left.link(demux_left)
        # demux_left динамически подключится к parse_left
        parse_left.link(decode_left)
        decode_left.link(convert_left)
        convert_left.link(queue_left)
        
        # Правая ветка
        src_right.link(demux_right)
        # demux_right динамически подключится к parse_right
        parse_right.link(decode_right)
        decode_right.link(convert_right)
        convert_right.link(queue_right)
        
        # Обработчики для динамических пэдов demux
        def on_pad_added_left(demux, pad):
            if pad.get_current_caps().to_string().startswith("video/x-h264"):
                parse_pad = parse_left.get_static_pad("sink")
                if not parse_pad.is_linked():
                    pad.link(parse_pad)
                    print("  ✓ Левый видеопоток подключен")
        
        def on_pad_added_right(demux, pad):
            if pad.get_current_caps().to_string().startswith("video/x-h264"):
                parse_pad = parse_right.get_static_pad("sink")
                if not parse_pad.is_linked():
                    pad.link(parse_pad)
                    print("  ✓ Правый видеопоток подключен")
        
        demux_left.connect("pad-added", on_pad_added_left)
        demux_right.connect("pad-added", on_pad_added_right)
        
        # Подключаем к streammux
        sinkpad0 = streammux.get_request_pad("sink_0")
        srcpad_left = queue_left.get_static_pad("src")
        srcpad_left.link(sinkpad0)
        
        sinkpad1 = streammux.get_request_pad("sink_1")
        srcpad_right = queue_right.get_static_pad("src")
        srcpad_right.link(sinkpad1)
        
        # Остальная цепочка
        streammux.link(stitch)
        stitch.link(queue_out)
        
        if display_mode == "egl":
            queue_out.link(transform)
            transform.link(sink)
        elif display_mode == "x11":
            queue_out.link(converter)
            converter.link(filter_nvmm)
            filter_nvmm.link(scaler)
            scaler.link(filter_raw)
            filter_raw.link(sink)
        else:
            queue_out.link(converter)
            converter.link(filter)
            filter.link(sink)
        
        # Для зацикливания
        if loop:
            def on_eos():
                print("🔄 Перезапуск для зацикливания...")
                pipeline.seek_simple(
                    Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                    0
                )
                return True
            
            # Сохраняем обработчик
            pipeline.on_eos = on_eos
        
        return pipeline
    
    def run(self, left_file, right_file, display_mode="egl", use_advanced=False, loop=True):
        """Запуск pipeline"""
        
        # Проверка файлов
        for f in [left_file, right_file]:
            if not os.path.exists(f):
                print(f"❌ Файл не найден: {f}")
                return False
        
        print("🎬 ПАНОРАМА 360°")
        print("=" * 60)
        print(f"📹 Источники:")
        print(f"   • Левый: {left_file}")
        print(f"   • Правый: {right_file}")
        print(f"🎯 Параметры:")
        print(f"   • Режим: {display_mode}")
        print(f"   • Pipeline: {'продвинутый' if use_advanced else 'простой'}")
        print(f"   • Зацикливание: {'да' if loop else 'нет'}")
        print(f"   • Выход: 4096x2048 (эквиректангулярная проекция)")
        print("-" * 60)
        
        # Создаем pipeline
        try:
            if use_advanced:
                print("🔧 Используется продвинутый pipeline с полным контролем")
                self.pipeline = self.create_advanced_pipeline(left_file, right_file, display_mode, loop)
            else:
                print("🚀 Используется простой pipeline")
                self.pipeline = self.create_simple_pipeline(left_file, right_file, display_mode)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if not self.pipeline:
            print("❌ Не удалось создать pipeline")
            return False
        
        # Настройка bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        self.loop = GLib.MainLoop()
        
        def on_message(bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                print("\n🏁 Конец потока")
                if use_advanced and loop and hasattr(self.pipeline, 'on_eos'):
                    self.pipeline.on_eos()
                else:
                    self.loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"\n❌ Ошибка: {err}")
                if debug:
                    print(f"   Debug: {debug}")
                self.loop.quit()
            elif t == Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                # Игнорируем предупреждения о segment
                if "segment event" not in str(warn):
                    print(f"\n⚠️  {warn}")
            elif t == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old, new, pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        print("\n✅ Pipeline запущен!")
                        print("📺 Отображение панорамы...")
                        print("🛑 Нажмите Ctrl+C для остановки\n")
            elif t == Gst.MessageType.STREAM_STATUS:
                status_type, owner = message.parse_stream_status()
                if status_type == Gst.StreamStatusType.CREATE:
                    self.frame_count += 1
                    if self.frame_count % 300 == 0:
                        print(f"📊 Обработано кадров: {self.frame_count}")
            return True
        
        bus.connect("message", on_message)
        
        # Запуск
        print("\n⏳ Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            return False
        
        # Главный цикл
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏹️  Остановлено пользователем")
        
        # Очистка
        print("🧹 Завершение...")
        self.pipeline.set_state(Gst.State.NULL)
        
        print(f"\n✅ Готово! Обработано кадров: {self.frame_count}")
        return True


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 panorama_stream.py left.mp4 right.mp4 [опции]")
        print("\nОпции:")
        print("  egl      - вывод через EGL (по умолчанию)")
        print("  x11      - вывод через X11")
        print("  file     - сохранить в файл")
        print("  auto     - автоматический выбор")
        print("  --adv    - продвинутый pipeline")
        print("  --noloop - без зацикливания")
        print("\nПримеры:")
        print("  python3 panorama_stream.py left.mp4 right.mp4")
        print("  python3 panorama_stream.py left.mp4 right.mp4 x11")
        print("  python3 panorama_stream.py left.mp4 right.mp4 --adv")
        sys.exit(1)
    
    left_file = sys.argv[1]
    right_file = sys.argv[2]
    
    # Парсинг аргументов
    display_mode = "egl"
    use_advanced = False
    loop = True
    
    for arg in sys.argv[3:]:
        if arg in ["egl", "x11", "file", "auto"]:
            display_mode = arg
        elif arg == "--adv":
            use_advanced = True
        elif arg == "--noloop":
            loop = False
    
    # Настройка путей
    plugin_path = os.getcwd()
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"
    
    # Минимальная отладка
    os.environ['GST_DEBUG'] = 'nvdsstitch:3,GST_STATES:2'
    
    print(f"📁 Plugin path: {plugin_path}")
    
    # Запуск
    app = PanoramaStream()
    
    try:
        if app.run(left_file, right_file, display_mode, use_advanced, loop):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()