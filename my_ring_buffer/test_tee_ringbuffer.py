#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УПРОЩЁННЫЙ ТЕСТ: Video → Tee → Direct / Ring Buffer

Проверяем работу ring buffer с разделением потока через tee
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import threading

# Устанавливаем путь к плагину
ringbuffer_path = "/home/nvidia/deep_cv_football/my_ring_buffer"
os.environ['GST_PLUGIN_PATH'] = f"{ringbuffer_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

Gst.init(None)

# Статистика
stats = {
    'direct': {'frames': 0, 'pts_list': [], 'wall_times': []},
    'buffered': {'frames': 0, 'pts_list': [], 'wall_times': []}
}
stats_lock = threading.Lock()
start_time = time.time()


def probe_callback(pad, info, branch_name):
    """Probe для замера метрик."""
    global stats, start_time

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    pts = gst_buffer.pts
    if pts == Gst.CLOCK_TIME_NONE:
        return Gst.PadProbeReturn.OK

    pts_sec = pts / 1e9
    wall_time = time.time() - start_time

    with stats_lock:
        branch = stats[branch_name]
        branch['frames'] += 1
        branch['pts_list'].append(pts_sec)
        branch['wall_times'].append(wall_time)

        # Логируем каждые 15 кадров
        if branch['frames'] % 15 == 0:
            icon = "⚡" if branch_name == 'direct' else "🔄"
            print(f"{icon} {branch_name:8s}: кадр {branch['frames']:4d} | PTS: {pts_sec:6.2f}s | Wall: {wall_time:6.2f}s")

    return Gst.PadProbeReturn.OK


def on_message(bus, message):
    """Обработчик сообщений."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n📭 EOS")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n❌ Ошибка: {err}")
        loop.quit()
    return True


def print_stats():
    """Выводим статистику."""
    print("\n" + "="*80)
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    print("="*80)

    with stats_lock:
        direct = stats['direct']
        buffered = stats['buffered']

        print(f"\n⚡ Прямая ветка:")
        print(f"   Кадров: {direct['frames']}")
        if direct['frames'] > 0:
            fps = direct['frames'] / direct['wall_times'][-1] if direct['wall_times'] else 0
            print(f"   FPS: {fps:.2f}")
            print(f"   Первый PTS: {direct['pts_list'][0]:.3f}s")
            print(f"   Последний PTS: {direct['pts_list'][-1]:.3f}s")

        print(f"\n🔄 Буферная ветка:")
        print(f"   Кадров: {buffered['frames']}")
        if buffered['frames'] > 0:
            fps = buffered['frames'] / buffered['wall_times'][-1] if buffered['wall_times'] else 0
            print(f"   FPS: {fps:.2f}")
            print(f"   Первый PTS: {buffered['pts_list'][0]:.3f}s")
            print(f"   Последний PTS: {buffered['pts_list'][-1]:.3f}s")

        # Рассчитываем задержку
        if direct['frames'] > 0 and buffered['frames'] > 0:
            print(f"\n⏱️  ЗАДЕРЖКА:")
            # Берём первые кадры обеих веток
            pts_delay = buffered['pts_list'][0] - direct['pts_list'][0]
            wall_delay = buffered['wall_times'][0] - direct['wall_times'][0]

            print(f"   Ожидаемая: 3.0 сек")
            print(f"   По PTS: {pts_delay:.2f} сек")
            print(f"   По wall time: {wall_delay:.2f} сек")

        # АНАЛИЗ ПОСТОЯНСТВА РАЗРЫВА
        print(f"\n📈 АНАЛИЗ ПОСТОЯНСТВА РАЗРЫВА PTS:")
        if direct['frames'] >= 120 and buffered['frames'] >= 120:
            # Берём каждый 30-й кадр, начиная с кадра 90 (после заполнения буфера)
            gaps = []
            for i in range(90, min(direct['frames'], buffered['frames']), 30):
                d_pts = direct['pts_list'][i]
                b_pts = buffered['pts_list'][i]
                gap = abs(d_pts - b_pts)
                gaps.append(gap)
                print(f"   Кадр {i:3d}: разрыв = {gap:.3f}s")

            if len(gaps) >= 3:
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                min_gap = min(gaps)
                variance = max_gap - min_gap

                print(f"\n   Средний разрыв:  {avg_gap:.3f}s")
                print(f"   Минимум:         {min_gap:.3f}s")
                print(f"   Максимум:        {max_gap:.3f}s")
                print(f"   Вариация:        {variance:.3f}s")

                if variance < 0.1:
                    print(f"\n   ✅ РАЗРЫВ СТАБИЛЬНЫЙ! (вариация {variance:.3f}s < 0.1s)")
                elif variance < 0.5:
                    print(f"\n   ⚠️  Разрыв умеренно стабильный (вариация {variance:.3f}s)")
                else:
                    print(f"\n   ❌ Разрыв нестабильный! (вариация {variance:.3f}s > 0.5s)")

    print("\n" + "="*80)


# Параметры
WIDTH = 1920
HEIGHT = 1080
BUFFER_DURATION = 3.0
FRAMERATE = 30

buffer_slots = int(BUFFER_DURATION * FRAMERATE)
frame_size = WIDTH * HEIGHT * 4  # RGBA
ring_bytes = buffer_slots * frame_size

print("="*80)
print("🚀 ТЕСТ: Tee → Direct / Ring Buffer")
print("="*80)
print(f"\n📐 Параметры:")
print(f"   Разрешение:  {WIDTH}x{HEIGHT}")
print(f"   Задержка:    {BUFFER_DURATION} сек")
print(f"   Слоты:       {buffer_slots} кадров")
print(f"   Размер:      {ring_bytes / (1024**2):.1f} MB")
print()

pipeline_str = f"""
    filesrc location=/home/nvidia/deep_cv_football/new_week/left.mp4 !
    qtdemux ! h264parse ! nvv4l2decoder !
    nvvideoconvert compute-hw=1 !
    video/x-raw(memory:NVMM),format=RGBA,width={WIDTH},height={HEIGHT} !
    queue max-size-buffers=2 !

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
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

# Добавляем probe'ы
direct_identity = pipeline.get_by_name("direct_identity")
if direct_identity:
    pad = direct_identity.get_static_pad("src")
    pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, "direct")
    print("✅ Probe на прямую ветку")

buffered_identity = pipeline.get_by_name("buffered_identity")
if buffered_identity:
    pad = buffered_identity.get_static_pad("src")
    pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, "buffered")
    print("✅ Probe на буферную ветку")

# Bus
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Запуск
print("\n▶️  Запуск...")
print("="*80)
print()

start_time = time.time()  # Сбрасываем start_time перед запуском
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
print_stats()
print("\n✅ Готово")
