#!/usr/bin/env python3
"""
ТЕСТ ЗАДЕРЖКИ: Stitch → Tee → Direct / Ring Buffer

Анализ:
1. FPS на обеих ветках
2. PTS временные метки
3. ПОСТОЯНСТВО разрыва между ветками
"""

import sys, os, gi, time, threading
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Пути к плагинам
ringbuffer_path = "/home/nvidia/deep_cv_football/my_ring_buffer"
stitch_path = "/home/nvidia/deep_cv_football/my_steach"
os.environ['GST_PLUGIN_PATH'] = f"{ringbuffer_path}:{stitch_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

Gst.init(None)

# Статистика
stats = {
    'direct': [],      # [(pts, wall_time), ...]
    'buffered': []     # [(pts, wall_time), ...]
}
stats_lock = threading.Lock()
start_time = None

def probe_callback(pad, info, branch_name):
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
        stats[branch_name].append((pts_sec, wall_time))

        # Логируем каждые 30 кадров
        if len(stats[branch_name]) % 30 == 0:
            icon = "⚡" if branch_name == 'direct' else "🔄"
            count = len(stats[branch_name])

            # Вычисляем FPS
            if count >= 30:
                recent = stats[branch_name][-30:]
                time_span = recent[-1][1] - recent[0][1]
                fps = 30 / time_span if time_span > 0 else 0
            else:
                fps = 0

            print(f"{icon} {branch_name:8s}: кадр {count:4d} | PTS: {pts_sec:7.2f}s | Wall: {wall_time:6.2f}s | FPS: {fps:5.1f}")

            # Анализ разрыва между ветками
            if count >= 30 and len(stats['direct']) >= 30 and len(stats['buffered']) >= 30:
                # Сравниваем последние кадры
                direct_pts = stats['direct'][-1][0]
                buffered_pts = stats['buffered'][-1][0]
                gap = abs(direct_pts - buffered_pts)

                if branch_name == 'buffered':
                    print(f"   📊 Разрыв PTS: {gap:.3f}s")

    return Gst.PadProbeReturn.OK

def on_message(bus, message):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n📭 EOS")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n❌ Ошибка: {err}")
        loop.quit()
    return True

def print_final_analysis():
    print("\n" + "="*80)
    print("📊 ФИНАЛЬНЫЙ АНАЛИЗ ЗАДЕРЖКИ")
    print("="*80)

    with stats_lock:
        direct = stats['direct']
        buffered = stats['buffered']

        if len(direct) < 120 or len(buffered) < 120:
            print("⚠️  Недостаточно данных для анализа")
            return

        print(f"\n⚡ Прямая ветка:   {len(direct)} кадров")
        print(f"🔄 Буферная ветка: {len(buffered)} кадров")

        # Анализ разрыва PTS через весь тест
        print("\n📈 Анализ постоянства разрыва:")

        # Берём каждый 30-й кадр
        samples = []
        for i in range(90, min(len(direct), len(buffered)), 30):
            if i < len(direct) and i < len(buffered):
                d_pts = direct[i][0]
                b_pts = buffered[i][0]
                gap = abs(d_pts - b_pts)
                samples.append(gap)
                print(f"   Кадр {i:3d}: разрыв = {gap:.3f}s")

        if len(samples) >= 3:
            avg_gap = sum(samples) / len(samples)
            max_gap = max(samples)
            min_gap = min(samples)
            variance = max_gap - min_gap

            print(f"\n📊 Статистика разрыва:")
            print(f"   Средний:    {avg_gap:.3f}s")
            print(f"   Минимум:    {min_gap:.3f}s")
            print(f"   Максимум:    {max_gap:.3f}s")
            print(f"   Вариация:   {variance:.3f}s")

            if variance < 0.1:
                print(f"\n✅ РАЗРЫВ СТАБИЛЬНЫЙ! (вариация {variance:.3f}s < 0.1s)")
            elif variance < 0.5:
                print(f"\n⚠️  Разрыв умеренно стабильный (вариация {variance:.3f}s)")
            else:
                print(f"\n❌ Разрыв нестабильный! (вариация {variance:.3f}s > 0.5s)")

        # FPS анализ
        if len(direct) >= 60:
            recent_direct = direct[-60:]
            time_span = recent_direct[-1][1] - recent_direct[0][1]
            fps_direct = 60 / time_span if time_span > 0 else 0

            recent_buffered = buffered[-60:]
            time_span = recent_buffered[-1][1] - recent_buffered[0][1]
            fps_buffered = 60 / time_span if time_span > 0 else 0

            print(f"\n📊 Финальный FPS:")
            print(f"   ⚡ Прямая:  {fps_direct:.1f} FPS")
            print(f"   🔄 Буфер:   {fps_buffered:.1f} FPS")

# Параметры
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900
BUFFER_DURATION = 5.0
FRAMERATE = 30

buffer_slots = int(BUFFER_DURATION * FRAMERATE)
frame_size = PANORAMA_WIDTH * PANORAMA_HEIGHT * 4
ring_bytes = buffer_slots * frame_size

print("="*80)
print("🚀 АНАЛИЗ ЗАДЕРЖКИ: Stitch → Tee → Direct / Ring Buffer")
print("="*80)
print(f"\n📐 Конфигурация:")
print(f"   Панорама:     {PANORAMA_WIDTH}x{PANORAMA_HEIGHT}")
print(f"   Задержка:     {BUFFER_DURATION}с ({buffer_slots} кадров)")
print(f"   Ring buffer:  {ring_bytes / (1024**3):.2f} GB")
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
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

# Probe'ы
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

start_time = time.time()
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
print_final_analysis()
print("\n✅ Анализ завершён")
