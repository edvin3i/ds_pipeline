#!/usr/bin/env python3
"""
Простой бенчмарк производительности без stitcher
"""

import os
import sys
import time
import psutil

os.environ['GST_PLUGIN_PATH'] = '/home/nvidia/deep_cv_football/my_ring_buffer'

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class SimpleBenchmark:
    def __init__(self):
        self.direct_frames = 0
        self.buffered_frames = 0
        self.start_time = None
        self.pipeline = None

    def run(self):
        print("="*60)
        print("⚡ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ RING BUFFER")
        print("="*60)

        # 3 секунды = 90 слотов
        slots = 90
        width = 1920
        height = 1080
        ring_bytes = slots * width * height * 4

        print(f"\n📊 Параметры:")
        print(f"   Разрешение: {width}×{height}")
        print(f"   Буфер: {slots} слотов ({slots/30:.1f} сек)")
        print(f"   Память: {ring_bytes/(1024**3):.2f} GB")
        print("="*60)

        pipeline_str = f"""
            videotestsrc num-buffers=600 pattern=ball !
            video/x-raw,width={width},height={height},framerate=30/1 !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            tee name=t

            t. ! queue !
                 identity name=direct !
                 fpsdisplaysink text-overlay=false video-sink=fakesink sync=false

            t. ! queue !
                 nvdsringbuf
                    ring-bytes={ring_bytes}
                    min-slots={slots}
                    chunk=1 !
                 identity name=buffered !
                 fpsdisplaysink text-overlay=false video-sink=fakesink sync=false
        """

        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Пробы для подсчёта
        direct = self.pipeline.get_by_name("direct")
        direct.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            lambda pad, info: self.count_direct()
        )

        buffered = self.pipeline.get_by_name("buffered")
        buffered.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            lambda pad, info: self.count_buffered()
        )

        # Обработчик сообщений
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        # Запуск
        print("\n▶️ Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить")
            return

        self.start_time = time.time()

        # Мониторинг
        def print_stats():
            if not self.start_time:
                return True

            elapsed = time.time() - self.start_time
            if elapsed < 0.1:
                return True

            # FPS
            direct_fps = self.direct_frames / elapsed
            buffered_fps = self.buffered_frames / elapsed

            # Ресурсы
            process = psutil.Process()
            cpu = process.cpu_percent(interval=0)
            mem_gb = process.memory_info().rss / (1024**3)

            # Задержка
            delay_frames = self.direct_frames - self.buffered_frames
            delay_sec = delay_frames / 30.0 if delay_frames > 0 else 0

            print(f"\r📊 Direct: {direct_fps:.1f} FPS | "
                  f"Buffered: {buffered_fps:.1f} FPS | "
                  f"Delay: {delay_sec:.1f}s | "
                  f"CPU: {cpu:.1f}% | RAM: {mem_gb:.2f}GB     ",
                  end='', flush=True)

            return True

        GLib.timeout_add(500, print_stats)

        # Главный цикл
        mainloop = GLib.MainLoop()
        try:
            mainloop.run()
        except KeyboardInterrupt:
            print("\n\n⏸️ Остановлено")

        self.print_final()

    def count_direct(self):
        self.direct_frames += 1
        return Gst.PadProbeReturn.OK

    def count_buffered(self):
        self.buffered_frames += 1
        return Gst.PadProbeReturn.OK

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            print("\n\n✅ Тест завершён")
            self.print_final()
            GLib.MainLoop().quit()
        elif message.type == Gst.MessageType.ERROR:
            err, _ = message.parse_error()
            print(f"\n❌ Ошибка: {err}")
            GLib.MainLoop().quit()

    def print_final(self):
        if not self.start_time:
            return

        elapsed = time.time() - self.start_time

        print("\n" + "="*60)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*60)

        avg_direct = self.direct_frames / elapsed if elapsed > 0 else 0
        avg_buffered = self.buffered_frames / elapsed if elapsed > 0 else 0

        print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:")
        print(f"   Время работы: {elapsed:.1f} сек")
        print(f"   Прямая ветка:  {avg_direct:.1f} FPS ({self.direct_frames} кадров)")
        print(f"   Буфер ветка:   {avg_buffered:.1f} FPS ({self.buffered_frames} кадров)")

        delay_frames = self.direct_frames - self.buffered_frames
        delay_sec = delay_frames / 30.0 if delay_frames > 0 else 0
        print(f"\n⏱️ ЗАДЕРЖКА:")
        print(f"   {delay_frames} кадров = {delay_sec:.1f} секунд")

        print(f"\n✨ ОЦЕНКА:")
        if avg_direct >= 29 and avg_buffered >= 29:
            print("   ✅ ОТЛИЧНО! Реальное время на обеих ветках.")
        elif avg_direct >= 25 and avg_buffered >= 25:
            print("   ✅ ХОРОШО. Близко к реальному времени.")
        else:
            print(f"   ⚠️ Производительность ниже ожидаемой.")

        # Нагрузка на копирование
        overhead_percent = ((avg_direct - avg_buffered) / avg_direct * 100) if avg_direct > 0 else 0
        print(f"\n📈 Накладные расходы буфера: {overhead_percent:.1f}%")

        if overhead_percent < 5:
            print("   ✅ Минимальные накладные расходы")
        elif overhead_percent < 10:
            print("   ✅ Приемлемые накладные расходы")
        else:
            print("   ⚠️ Высокие накладные расходы")

        print("="*60)

if __name__ == "__main__":
    benchmark = SimpleBenchmark()
    benchmark.run()