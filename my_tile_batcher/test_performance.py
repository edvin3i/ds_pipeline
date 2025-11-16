#!/usr/bin/env python3
"""
Тест производительности nvtilebatcher
Проверяет FPS после исправлений
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time
import os

Gst.init(None)

def test_performance():
    """Тест производительности плагина"""
    print("\n" + "="*70)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ nvtilebatcher")
    print("="*70)

    # Пути к плагинам
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    stitch_path = "/home/nvidia/deep_cv_football/my_steach"

    print(f"📁 TileBatcher plugin: {plugin_path}")
    print(f"📁 Stitch plugin: {stitch_path}")

    # Добавляем пути к плагинам
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{stitch_path}"

    # Pipeline: videotestsrc -> nvvideoconvert -> создаём панораму -> tilebatcher -> fakesink
    pipeline_str = f"""
        videotestsrc pattern=smpte num-buffers=300 !
        video/x-raw,format=I420,width=6528,height=1632,framerate=30/1 !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 !
        nvtilebatcher name=batcher silent=false !
        fakesink name=sink sync=false
    """

    print(f"\n🎬 Creating pipeline...")
    print(f"Pipeline: videotestsrc -> nvvideoconvert -> nvtilebatcher -> fakesink")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return False

    # Статистика
    stats = {
        'buffers': 0,
        'start_time': None,
        'first_buffer_time': None,
        'last_buffer_time': None,
        'errors': 0
    }

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ ERROR: {err.message}")
            stats['errors'] += 1
            loop.quit()
        elif t == Gst.MessageType.EOS:
            stats['last_buffer_time'] = time.time()
            print(f"\n✅ EOS received")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"⚠️  WARNING: {warn.message}")
        return True

    def probe_callback(pad, info):
        """Probe для подсчёта FPS"""
        buffer = info.get_buffer()
        if buffer:
            current_time = time.time()

            if stats['first_buffer_time'] is None:
                stats['first_buffer_time'] = current_time

            stats['buffers'] += 1
            stats['last_buffer_time'] = current_time

            # Логируем каждые 50 буферов
            if stats['buffers'] % 50 == 0:
                elapsed = current_time - stats['first_buffer_time']
                fps = stats['buffers'] / elapsed if elapsed > 0 else 0
                print(f"  📊 Processed {stats['buffers']} buffers, FPS: {fps:.2f}")

        return Gst.PadProbeReturn.OK

    # Добавляем probe
    batcher = pipeline.get_by_name('batcher')
    if batcher:
        src_pad = batcher.get_static_pad("src")
        if src_pad:
            src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback)

    # Запускаем
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    print(f"\n▶️  Starting pipeline...")
    stats['start_time'] = time.time()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print(f"❌ Unable to set the pipeline to PLAYING state")
        return False

    # Main loop
    loop = GLib.MainLoop()

    # Timeout после 30 секунд
    def timeout_callback():
        print(f"\n⏰ Timeout after 30s")
        stats['last_buffer_time'] = time.time()
        loop.quit()
        return False

    GLib.timeout_add_seconds(30, timeout_callback)

    try:
        loop.run()
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        stats['last_buffer_time'] = time.time()

    # Останавливаем
    pipeline.set_state(Gst.State.NULL)

    # Результаты
    print(f"\n" + "="*70)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТА ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"="*70)

    if stats['first_buffer_time'] and stats['last_buffer_time']:
        total_time = stats['last_buffer_time'] - stats['first_buffer_time']
        fps = stats['buffers'] / total_time if total_time > 0 else 0

        print(f"  Всего буферов:     {stats['buffers']}")
        print(f"  Время обработки:   {total_time:.2f} сек")
        print(f"  Средний FPS:       {fps:.2f}")
        print(f"  Ошибок:            {stats['errors']}")

        # Оценка производительности
        print(f"\n🎯 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        if fps >= 40:
            print(f"  ✅ ОТЛИЧНО: {fps:.2f} FPS - высокая производительность!")
        elif fps >= 30:
            print(f"  ✅ ХОРОШО: {fps:.2f} FPS - приемлемая производительность")
        elif fps >= 20:
            print(f"  ⚠️  СРЕДНЕ: {fps:.2f} FPS - производительность снижена")
        else:
            print(f"  ❌ ПЛОХО: {fps:.2f} FPS - низкая производительность")

        print(f"\n📝 Примечание:")
        print(f"  - Тест использует videotestsrc (CPU генерация)")
        print(f"  - Реальная производительность с GPU может быть выше")
        print(f"  - Исправления добавили проверки, но минимальный overhead")

        return fps >= 30
    else:
        print(f"  ❌ Недостаточно данных для оценки производительности")
        return False

if __name__ == '__main__':
    success = test_performance()
    sys.exit(0 if success else 1)
