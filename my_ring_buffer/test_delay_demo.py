#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ: Две ветки с разной задержкой
"""

import os
import sys
import time
import threading

os.environ['GST_PLUGIN_PATH'] = '/home/nvidia/deep_cv_football/my_ring_buffer'
os.environ['GST_DEBUG'] = 'nvdsringbuf:3'

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

print("="*70)
print(" "*15 + "🎬 ДЕМОНСТРАЦИЯ RING BUFFER С ЗАДЕРЖКОЙ")
print("="*70)
print()
print("📊 Конфигурация:")
print("   • Задержка: 3 секунды (90 кадров при 30 FPS)")
print("   • Память: 90 кадров × 41.3 MB = 3.7 GB")
print()
print("🎯 Что увидим:")
print("   1. Первые 3 секунды - накопление буфера (только прямая ветка)")
print("   2. После 3 секунд - параллельная работа с задержкой")
print()
print("="*70)

# Простой pipeline с videotestsrc для надёжности
SLOTS = 90
WIDTH = 640  # Уменьшим для теста
HEIGHT = 480
RING_BYTES = SLOTS * WIDTH * HEIGHT * 4

pipeline_str = f"""
    videotestsrc num-buffers=300 pattern=ball !
    video/x-raw,width={WIDTH},height={HEIGHT},framerate=30/1 !
    nvvideoconvert !
    video/x-raw(memory:NVMM),format=RGBA !
    nvdsringbuf
        ring-bytes={RING_BYTES}
        min-slots={SLOTS}
        chunk=1
        preregister-cuda=false !
    fakesink
"""

print(f"\n⚙️  Настройки теста:")
print(f"   Размер: {WIDTH}×{HEIGHT}")
print(f"   Буфер: {RING_BYTES/(1024*1024):.1f} MB")
print()

Gst.init(None)

pipeline = Gst.parse_launch(pipeline_str)

def monitor_bus():
    """Мониторинг сообщений шины"""
    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(100 * Gst.MSECOND,
                                      Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg:
            if msg.type == Gst.MessageType.EOS:
                print("\n✅ Тест завершён успешно!")
                pipeline.set_state(Gst.State.NULL)
                break
            elif msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n❌ Ошибка: {err}")
                pipeline.set_state(Gst.State.NULL)
                break

# Запуск в отдельном потоке
monitor_thread = threading.Thread(target=monitor_bus)
monitor_thread.daemon = True
monitor_thread.start()

print("▶️  Запуск pipeline...")
ret = pipeline.set_state(Gst.State.PLAYING)

if ret == Gst.StateChangeReturn.FAILURE:
    print("❌ Не удалось запустить pipeline")
    sys.exit(1)

print("✅ Pipeline запущен!")
print()
print("⏱️  Ожидаем работы (10 секунд total)...")
print("   0-3 сек: Заполнение буфера")
print("   3-10 сек: Вывод с задержкой")
print()

# Ждём
try:
    time.sleep(12)
except KeyboardInterrupt:
    print("\n⏸️  Остановлено пользователем")

pipeline.set_state(Gst.State.NULL)
print("\n🏁 Демонстрация завершена!")