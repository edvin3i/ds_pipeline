#!/usr/bin/env python3
"""
Простой тест ring buffer БЕЗ tee
Проверяем работает ли ring buffer сам по себе
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Путь к плагину
ringbuffer_path = "/home/nvidia/deep_cv_football/my_ring_buffer"
os.environ['GST_PLUGIN_PATH'] = f"{ringbuffer_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

Gst.init(None)

# Параметры для 1 секунду задержки (маленькая задержка для быстрого теста)
WIDTH = 640
HEIGHT = 480
BUFFER_DURATION = 1.0
FRAMERATE = 30

buffer_slots = int(BUFFER_DURATION * FRAMERATE)  # 30 кадров
frame_size = WIDTH * HEIGHT * 4  # RGBA
ring_bytes = buffer_slots * frame_size

print("="*60)
print("🧪 ТЕСТ: Ring Buffer standalone (без tee)")
print("="*60)
print(f"Параметры: {WIDTH}x{HEIGHT}, задержка {BUFFER_DURATION}с, {buffer_slots} кадров")
print()

# Простой pipeline: videotestsrc → nvvideoconvert → ring buffer → fakesink
pipeline_str = f"""
    videotestsrc num-buffers=120 pattern=ball !
    video/x-raw,width={WIDTH},height={HEIGHT},framerate=30/1 !
    nvvideoconvert compute-hw=1 !
    video/x-raw(memory:NVMM),format=RGBA,width={WIDTH},height={HEIGHT} !
    nvdsringbuf
        ring-bytes={ring_bytes}
        min-slots={buffer_slots}
        chunk=1
        preregister-cuda=false !
    fakesink sync=false
"""

print("📦 Pipeline:")
print(pipeline_str)
print()

try:
    pipeline = Gst.parse_launch(pipeline_str)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

def on_message(bus, message):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n✅ EOS - pipeline завершён успешно!")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n❌ Ошибка: {err}")
        print(f"   Debug: {debug}")
        loop.quit()
    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

print("▶️  Запуск...")
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("❌ Не удалось запустить")
    sys.exit(1)

loop = GLib.MainLoop()

try:
    loop.run()
except KeyboardInterrupt:
    print("\n⏸️  Ctrl+C")

pipeline.set_state(Gst.State.NULL)
print("\n✅ Готово")
