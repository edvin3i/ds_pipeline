#!/usr/bin/env python3
"""
Простой тест FPS для плагина nvdsstitch
Измеряет производительность обработки панорамы
"""
import sys
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# Константы размеров панорамы (можно изменить для тестирования)
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

class FPSTester:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.start_time = None
        self.frame_count = 0
        self.last_report_time = None
        self.last_report_frames = 0

    def create_pipeline(self, left_file, right_file):
        """Создает pipeline для тестирования"""

        pipeline_str = f"""
            filesrc location={left_file} !
            qtdemux ! h264parse ! nvv4l2decoder !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_0

            filesrc location={right_file} !
            qtdemux ! h264parse ! nvv4l2decoder !
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
            fakesink sync=false
        """

        return Gst.parse_launch(pipeline_str)

    def on_buffer_probe(self, pad, info):
        """Probe для подсчёта кадров"""
        self.frame_count += 1

        current_time = time.time()

        # Первый кадр
        if self.start_time is None:
            self.start_time = current_time
            self.last_report_time = current_time
            self.last_report_frames = 0
            print(f"✅ Первый кадр получен в {current_time:.2f}")
            return Gst.PadProbeReturn.OK

        # Отчёты каждые 5 секунд
        time_since_report = current_time - self.last_report_time
        if time_since_report >= 5.0:
            frames_processed = self.frame_count - self.last_report_frames
            instant_fps = frames_processed / time_since_report

            total_time = current_time - self.start_time
            avg_fps = self.frame_count / total_time

            print(f"📊 [{total_time:6.1f}s] Кадры: {self.frame_count:5d} | "
                  f"Мгновенный FPS: {instant_fps:5.2f} | Средний FPS: {avg_fps:5.2f}")

            self.last_report_time = current_time
            self.last_report_frames = self.frame_count

        return Gst.PadProbeReturn.OK

    def run(self, left_file, right_file, duration=30):
        """Запуск теста"""

        # Проверка файлов
        for f in [left_file, right_file]:
            if not os.path.exists(f):
                print(f"❌ Файл не найден: {f}")
                return False

        print("=" * 70)
        print("🎯 ТЕСТ FPS ПЛАГИНА nvdsstitch")
        print("=" * 70)
        print(f"📹 Левый источник:  {left_file}")
        print(f"📹 Правый источник: {right_file}")
        print(f"⏱️  Длительность:     {duration} секунд")
        print(f"🖥️  Вход:            2 × 3840×2160 (4K)")
        print(f"📺 Выход:           {PANORAMA_WIDTH}×{PANORAMA_HEIGHT} (Panorama)")
        print("-" * 70)

        # Создаём pipeline
        try:
            self.pipeline = self.create_pipeline(left_file, right_file)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            return False

        if not self.pipeline:
            print("❌ Не удалось создать pipeline")
            return False

        # Добавляем probe на выход nvdsstitch
        stitch = self.pipeline.get_by_name("nvdsstitch0")
        if not stitch:
            print("❌ Не удалось найти элемент nvdsstitch")
            return False

        srcpad = stitch.get_static_pad("src")
        if not srcpad:
            print("❌ Не удалось получить src pad у nvdsstitch")
            return False

        srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer_probe)

        # Настройка bus
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
                print(f"\n❌ Ошибка: {err}")
                if debug:
                    print(f"   Debug: {debug}")
                self.loop.quit()
            return True

        bus.connect("message", on_message)

        # Таймер для автоматической остановки
        def timeout_handler():
            print(f"\n⏰ Время вышло ({duration}s)")
            self.loop.quit()
            return False

        GLib.timeout_add_seconds(duration, timeout_handler)

        # Запуск
        print("⏳ Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            return False

        print("✅ Pipeline запущен, начинается обработка...\n")

        # Главный цикл
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏹️  Остановлено пользователем")

        # Финальная статистика
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        self.pipeline.set_state(Gst.State.NULL)

        print("\n" + "=" * 70)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 70)
        print(f"⏱️  Время обработки:  {total_time:.2f} секунд")
        print(f"🎞️  Всего кадров:      {self.frame_count}")

        if total_time > 0:
            avg_fps = self.frame_count / total_time
            avg_latency = (total_time / self.frame_count) * 1000 if self.frame_count > 0 else 0

            print(f"⚡ Средний FPS:      {avg_fps:.2f}")
            print(f"⏲️  Средняя latency:  {avg_latency:.2f} ms")

            # Оценка производительности
            print("\n📈 Оценка производительности:")
            if avg_fps >= 45:
                print("   🟢 ОТЛИЧНО - плавная обработка 4K панорамы")
            elif avg_fps >= 40:
                print("   🟢 ХОРОШО - стабильная работа")
            elif avg_fps >= 30:
                print("   🟡 УДОВЛЕТВОРИТЕЛЬНО - приемлемая производительность")
            else:
                print("   🔴 НИЗКАЯ - требуется оптимизация")
        else:
            print("❌ Недостаточно данных для статистики")

        print("=" * 70)

        return True


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 test_fps.py left.mp4 right.mp4 [duration]")
        print("\nПараметры:")
        print("  left.mp4  - левый видеофайл (4K)")
        print("  right.mp4 - правый видеофайл (4K)")
        print("  duration  - длительность теста в секундах (по умолчанию: 30)")
        print("\nПримеры:")
        print("  python3 test_fps.py left.mp4 right.mp4")
        print("  python3 test_fps.py left.mp4 right.mp4 60")
        sys.exit(1)

    left_file = sys.argv[1]
    right_file = sys.argv[2]
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    # Настройка окружения
    plugin_path = os.getcwd()
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

    # Минимальные логи
    os.environ['GST_DEBUG'] = 'nvdsstitch:3'

    # Запуск теста
    tester = FPSTester()

    try:
        if tester.run(left_file, right_file, duration):
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
