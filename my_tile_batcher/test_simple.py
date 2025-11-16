#!/usr/bin/env python3
"""
Простой тест плагина nvtilebatcher
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time

Gst.init(None)

def test_basic():
    """Базовый тест плагина"""
    print("\n" + "="*70)
    print("ТЕСТ: Базовая работа nvtilebatcher")
    print("="*70)

    # Устанавливаем путь к плагину
    import os
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Plugin path: {plugin_path}")

    pipeline_str = f"""
        filesrc location={plugin_path}/panorama.jpg num-buffers=5 !
        jpegdec !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 !
        nvtilebatcher name=batcher panorama-width=6528 panorama-height=1632 silent=false !
        fakesink name=sink
    """

    print(f"\n🎬 Creating pipeline...")
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return False

    # Счётчики
    stats = {
        'buffers': 0,
        'errors': 0
    }

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ ERROR: {err.message}")
            print(f"   Debug: {debug}")
            stats['errors'] += 1
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print(f"\n✅ EOS received")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"⚠️  WARNING: {warn.message}")
        elif t == Gst.MessageType.INFO:
            info, debug = message.parse_info()
            print(f"ℹ️  INFO: {info.message}")
        return True

    def probe_callback(pad, info):
        """Probe для подсчёта буферов"""
        buffer = info.get_buffer()
        if buffer:
            stats['buffers'] += 1
            if stats['buffers'] <= 5:
                print(f"  📦 Buffer #{stats['buffers']}: {buffer.get_size()} bytes, "
                      f"PTS={buffer.pts/Gst.SECOND:.3f}s")
        return Gst.PadProbeReturn.OK

    # Добавляем probe
    batcher = pipeline.get_by_name('batcher')
    if batcher:
        src_pad = batcher.get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback)

    # Запускаем
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    print(f"▶️  Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print(f"❌ Unable to set the pipeline to PLAYING state")
        return False

    # Main loop
    loop = GLib.MainLoop()

    # Timeout после 10 секунд
    def timeout_callback():
        print(f"\n⏰ Timeout after 10s")
        loop.quit()
        return False

    GLib.timeout_add_seconds(10, timeout_callback)

    try:
        loop.run()
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")

    # Останавливаем
    pipeline.set_state(Gst.State.NULL)

    # Результаты
    print(f"\n" + "="*70)
    print(f"📊 РЕЗУЛЬТАТЫ:")
    print(f"   Буферов обработано: {stats['buffers']}")
    print(f"   Ошибок: {stats['errors']}")

    if stats['errors'] > 0:
        print(f"❌ ТЕСТ ПРОВАЛЕН: есть ошибки")
        return False
    elif stats['buffers'] >= 5:
        print(f"✅ ТЕСТ ПРОЙДЕН: плагин работает корректно")
        return True
    else:
        print(f"⚠️  ТЕСТ ЧАСТИЧНО ПРОЙДЕН: мало буферов")
        return False

if __name__ == '__main__':
    success = test_basic()
    sys.exit(0 if success else 1)
