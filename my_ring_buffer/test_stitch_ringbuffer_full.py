#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНЫЙ ТЕСТ: Stitch → Tee → Ring Buffer

Архитектура:
1. Два видео → nvdsstitch → панорама 5700x1900
2. Tee разделяет поток на две ветки:
   - Прямая ветка (без задержки)
   - Ring buffer ветка (с задержкой)
3. Замеряем FPS и временные метки на обеих ветках

Цель: убедиться что ring buffer даёт точную задержку
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import threading

# Устанавливаем пути к плагинам
ringbuffer_path = "/home/nvidia/deep_cv_football/my_ring_buffer"
stitch_path = "/home/nvidia/deep_cv_football/my_steach"
os.environ['GST_PLUGIN_PATH'] = f"{ringbuffer_path}:{stitch_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

# Инициализация GStreamer
Gst.init(None)

# Статистика
stats = {
    'direct': {'frames': 0, 'first_pts': None, 'last_pts': None, 'start_time': None},
    'buffered': {'frames': 0, 'first_pts': None, 'last_pts': None, 'start_time': None}
}
stats_lock = threading.Lock()


def probe_callback(pad, info, branch_name):
    """Probe для замера метрик на каждой ветке."""
    global stats

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    pts = gst_buffer.pts
    if pts == Gst.CLOCK_TIME_NONE:
        return Gst.PadProbeReturn.OK

    pts_sec = pts / 1e9
    wall_time = time.time()

    with stats_lock:
        branch = stats[branch_name]

        if branch['first_pts'] is None:
            branch['first_pts'] = pts_sec
            branch['start_time'] = wall_time

        branch['last_pts'] = pts_sec
        branch['frames'] += 1

        # Логируем каждые 30 кадров (раз в секунду)
        if branch['frames'] % 30 == 0:
            elapsed = wall_time - branch['start_time']
            fps = branch['frames'] / elapsed if elapsed > 0 else 0

            if branch_name == 'direct':
                print(f"⚡ Прямая:  кадр {branch['frames']:4d} | PTS: {pts_sec:6.2f}s | Прошло: {elapsed:6.2f}s | FPS: {fps:5.1f}")
            else:
                print(f"🔄 Буфер:   кадр {branch['frames']:4d} | PTS: {pts_sec:6.2f}s | Прошло: {elapsed:6.2f}s | FPS: {fps:5.1f}")

    return Gst.PadProbeReturn.OK


def on_message(bus, message):
    """Обработчик сообщений bus."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n📭 Конец потока")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n❌ Ошибка: {err}")
        print(f"   Debug: {debug}")
        loop.quit()
    return True


def print_final_stats():
    """Выводим финальную статистику."""
    print("\n" + "="*80)
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    print("="*80)

    with stats_lock:
        for branch_name, data in stats.items():
            if data['frames'] == 0:
                continue

            elapsed = time.time() - data['start_time']
            fps = data['frames'] / elapsed if elapsed > 0 else 0
            duration = data['last_pts'] - data['first_pts']

            icon = "⚡" if branch_name == 'direct' else "🔄"
            name = "Прямая ветка" if branch_name == 'direct' else "Буферная ветка"

            print(f"\n{icon} {name}:")
            print(f"   Кадров обработано: {data['frames']}")
            print(f"   FPS (среднее):     {fps:.2f}")
            print(f"   Первый PTS:        {data['first_pts']:.3f}s")
            print(f"   Последний PTS:     {data['last_pts']:.3f}s")
            print(f"   Длительность:      {duration:.3f}s")
            print(f"   Wall time:         {elapsed:.3f}s")

    # Вычисляем задержку между ветками
    with stats_lock:
        direct = stats['direct']
        buffered = stats['buffered']

        if direct['frames'] > 0 and buffered['frames'] > 0:
            # Сравниваем одинаковые номера кадров
            min_frames = min(direct['frames'], buffered['frames'])

            # Примерная задержка = разница в wall time при одинаковом количестве кадров
            time_diff = buffered['start_time'] - direct['start_time']

            print(f"\n⏱️  ЗАДЕРЖКА:")
            print(f"   Ожидаемая:  3.0 секунды")
            print(f"   Фактическая: ~{abs(time_diff):.2f} секунды")

            # Более точный расчёт через PTS
            if buffered['first_pts'] and direct['first_pts']:
                pts_delay = buffered['first_pts'] - direct['first_pts']
                print(f"   По PTS:      {pts_delay:.2f} секунды")

    print("\n" + "="*80)


# Создаём pipeline
print("="*80)
print("🚀 ТЕСТ: Stitch → Tee → Direct / Ring Buffer")
print("="*80)
print()

# Параметры ring buffer для 3 секунд
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900
BUFFER_DURATION = 3.0
FRAMERATE = 30

buffer_slots = int(BUFFER_DURATION * FRAMERATE)  # 90 кадров
frame_size = PANORAMA_WIDTH * PANORAMA_HEIGHT * 4  # RGBA
ring_bytes = buffer_slots * frame_size

print(f"📐 Параметры:")
print(f"   Панорама:    {PANORAMA_WIDTH}x{PANORAMA_HEIGHT}")
print(f"   Задержка:    {BUFFER_DURATION} сек")
print(f"   Слоты:       {buffer_slots} кадров")
print(f"   Размер:      {ring_bytes / (1024**3):.2f} GB")
print()

pipeline_str = f"""
    filesrc location=/home/nvidia/deep_cv_football/new_week/left.mp4 !
    qtdemux ! h264parse ! nvv4l2decoder !
    nvvideoconvert compute-hw=1 !
    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
    queue max-size-buffers=2 !
    mux.sink_0

    filesrc location=/home/nvidia/deep_cv_football/new_week/right.mp4 !
    qtdemux ! h264parse ! nvv4l2decoder !
    nvvideoconvert compute-hw=1 !
    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
    queue max-size-buffers=2 !
    mux.sink_1

    nvstreammux name=mux
        batch-size=2
        width=3840
        height=2160
        live-source=0
        batched-push-timeout=40000 !

    nvdsstitch
        left-source-id=0
        right-source-id=1
        gpu-id=0
        panorama-width={PANORAMA_WIDTH}
        panorama-height={PANORAMA_HEIGHT} !

    queue max-size-buffers=2 !
    video/x-raw(memory:NVMM),format=RGBA,width={PANORAMA_WIDTH},height={PANORAMA_HEIGHT} !

    tee name=splitter

    splitter. !
    queue name=direct_queue max-size-buffers=2 !
    identity name=direct_identity !
    fakesink name=direct_sink sync=false

    splitter. !
    queue name=buffer_queue max-size-buffers=5 !
    nvdsringbuf
        ring-bytes={ring_bytes}
        min-slots={buffer_slots}
        chunk=1
        preregister-cuda=false !
    identity name=buffered_identity !
    fakesink name=buffered_sink sync=false
"""

print("🔧 Создание pipeline...")
try:
    pipeline = Gst.parse_launch(pipeline_str)
except Exception as e:
    print(f"❌ Ошибка создания pipeline: {e}")
    sys.exit(1)

# Добавляем probe'ы для замера метрик
direct_identity = pipeline.get_by_name("direct_identity")
if direct_identity:
    pad = direct_identity.get_static_pad("src")
    pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, "direct")
    print("✅ Probe добавлен на прямую ветку")

buffered_identity = pipeline.get_by_name("buffered_identity")
if buffered_identity:
    pad = buffered_identity.get_static_pad("src")
    pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, "buffered")
    print("✅ Probe добавлен на буферную ветку")

# Настраиваем bus
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Запускаем pipeline
print("\n▶️  Запуск pipeline...")
print("="*80)
print()

ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("❌ Не удалось запустить pipeline")
    sys.exit(1)

# Главный цикл
loop = GLib.MainLoop()

try:
    loop.run()
except KeyboardInterrupt:
    print("\n⏸️  Остановка по Ctrl+C")

# Останавливаем pipeline
pipeline.set_state(Gst.State.NULL)

# Выводим статистику
print_final_stats()

print("\n✅ Тест завершён")
