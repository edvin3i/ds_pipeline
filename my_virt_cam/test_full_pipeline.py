#!/usr/bin/env python3
"""
Тест ПОЛНОГО PIPELINE: Стичинг + Виртуальная камера
2 × 4K видео → Панорама 6528×1800 → Виртуальная камера 1920×1080
"""
import sys
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# Константы размеров панорамы
PANORAMA_WIDTH = 6528
PANORAMA_HEIGHT = 1800

class FullPipelineFPSTester:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.start_time = None
        self.frame_count = 0
        self.last_report_time = None
        self.last_report_frames = 0

        # Счетчики для каждого этапа
        self.stitch_frames = 0
        self.vcam_frames = 0

    def create_full_pipeline(self, left_file, right_file, yaw=0, pitch=0, fov=50):
        """
        Полный pipeline: 2×4K → Stitch → Panorama → VirtualCam → 1920×1080
        """

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

            nvdsstitch name=stitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=3 !

            nvdsvirtualcam name=vcam
                yaw={yaw}
                pitch={pitch}
                roll=0
                fov={fov}
                output-width=1920
                output-height=1080
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !

            queue max-size-buffers=3 !
            fakesink sync=false
        """

        return Gst.parse_launch(pipeline_str)

    def on_stitch_probe(self, pad, info):
        """Probe на выходе стичинга (панорама)"""
        self.stitch_frames += 1
        return Gst.PadProbeReturn.OK

    def on_vcam_probe(self, pad, info):
        """Probe на выходе виртуальной камеры (финальное изображение)"""
        self.vcam_frames += 1
        self.frame_count += 1

        current_time = time.time()

        # Первый кадр
        if self.start_time is None:
            self.start_time = current_time
            self.last_report_time = current_time
            self.last_report_frames = 0
            print(f"✅ Первый кадр прошёл весь pipeline в {current_time:.2f}")
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
            print(f"   └─ Stitch: {self.stitch_frames:5d} | VCam: {self.vcam_frames:5d}")

            self.last_report_time = current_time
            self.last_report_frames = self.frame_count

        return Gst.PadProbeReturn.OK

    def run(self, left_file, right_file, yaw=0, pitch=0, fov=50, duration=30):
        """Запуск теста полного pipeline"""

        # Проверка файлов
        for f in [left_file, right_file]:
            if not os.path.exists(f):
                print(f"❌ Файл не найден: {f}")
                return False

        print("=" * 70)
        print("🎯 ТЕСТ ПОЛНОГО PIPELINE: STITCH + VIRTUALCAM")
        print("=" * 70)
        print(f"📹 Левый источник:   {left_file}")
        print(f"📹 Правый источник:  {right_file}")
        print(f"⏱️  Длительность:     {duration} секунд")
        print("")
        print("📊 Этапы обработки:")
        print("   1️⃣  Декодирование:   2 × H.264 → 2 × 3840×2160 RGBA")
        print(f"   2️⃣  Стичинг:         2 × 3840×2160 → {PANORAMA_WIDTH}×{PANORAMA_HEIGHT} (Panorama)")
        print(f"   3️⃣  Виртуальная камера: {PANORAMA_WIDTH}×{PANORAMA_HEIGHT} → 1920×1080 (Virtual)")
        print("")
        print(f"🎥 Параметры виртуальной камеры:")
        print(f"   • Yaw (поворот):   {yaw}°")
        print(f"   • Pitch (наклон):  {pitch}°")
        print(f"   • FOV (обзор):     {fov}°")
        print("-" * 70)

        # Создаём pipeline
        try:
            self.pipeline = self.create_full_pipeline(left_file, right_file, yaw, pitch, fov)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

        if not self.pipeline:
            print("❌ Не удалось создать pipeline")
            return False

        # Добавляем probe на выход стичинга
        stitch = self.pipeline.get_by_name("stitch")
        if stitch:
            srcpad = stitch.get_static_pad("src")
            if srcpad:
                srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_stitch_probe)

        # Добавляем probe на выход виртуальной камеры
        vcam = self.pipeline.get_by_name("vcam")
        if not vcam:
            print("❌ Не удалось найти элемент nvdsvirtualcam")
            return False

        srcpad = vcam.get_static_pad("src")
        if not srcpad:
            print("❌ Не удалось получить src pad у nvdsvirtualcam")
            return False

        srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_vcam_probe)

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
            elif t == Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                # Игнорируем некоторые предупреждения
                warn_str = str(warn).lower()
                if "segment" not in warn_str and "timestamp" not in warn_str:
                    print(f"\n⚠️  {warn}")
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
        print("📊 ИТОГОВАЯ СТАТИСТИКА ПОЛНОГО PIPELINE")
        print("=" * 70)
        print(f"⏱️  Время обработки:   {total_time:.2f} секунд")
        print(f"🎞️  Финальных кадров:  {self.frame_count}")
        print(f"   ├─ Стичинг кадров: {self.stitch_frames}")
        print(f"   └─ VCam кадров:    {self.vcam_frames}")

        if total_time > 0 and self.frame_count > 0:
            avg_fps = self.frame_count / total_time
            avg_latency = (total_time / self.frame_count) * 1000

            print(f"\n⚡ Средний FPS:       {avg_fps:.2f}")
            print(f"⏲️  Средняя latency:   {avg_latency:.2f} ms")

            # Расчёт пропускной способности
            input_mpx = 2 * 3840 * 2160 * avg_fps / 1e6  # Вход: 2×4K
            stitch_mpx = PANORAMA_WIDTH * PANORAMA_HEIGHT * avg_fps / 1e6  # Панорама
            output_mpx = 1920 * 1080 * avg_fps / 1e6     # Выход

            print(f"\n📊 Пропускная способность:")
            print(f"   • Вход (2×4K):     {input_mpx:6.1f} Mpx/s")
            print(f"   • Панорама:        {stitch_mpx:6.1f} Mpx/s")
            print(f"   • Выход (FullHD):  {output_mpx:6.1f} Mpx/s")

            # Оценка производительности
            print("\n📈 Оценка производительности ПОЛНОГО PIPELINE:")
            if avg_fps >= 30:
                print("   🟢 ОТЛИЧНО - плавная обработка всей цепочки!")
            elif avg_fps >= 25:
                print("   🟢 ХОРОШО - приемлемая производительность")
            elif avg_fps >= 20:
                print("   🟡 УДОВЛЕТВОРИТЕЛЬНО - работает, но можно оптимизировать")
            else:
                print("   🔴 НИЗКАЯ - требуется серьезная оптимизация")

            # Детальный breakdown
            print(f"\n🔬 Анализ производительности:")
            print(f"   • GPU время на кадр:      ~{avg_latency:.2f} ms")
            print(f"   • Теоретический макс FPS: ~{1000.0/avg_latency:.2f}")

            if self.stitch_frames > 0 and self.vcam_frames > 0:
                stitch_fps = self.stitch_frames / total_time
                vcam_fps = self.vcam_frames / total_time
                print(f"\n   📍 FPS по этапам:")
                print(f"      • Стичинг:         {stitch_fps:.2f} FPS")
                print(f"      • Виртуальная кам: {vcam_fps:.2f} FPS")
        else:
            print("❌ Недостаточно данных для статистики")

        print("=" * 70)

        return True


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 test_full_pipeline.py left.mp4 right.mp4 [options]")
        print("\nПараметры:")
        print("  left.mp4  - левый видеофайл (4K)")
        print("  right.mp4 - правый видеофайл (4K)")
        print("\nОпции:")
        print("  --yaw=N     - угол поворота камеры (-90 до +90, по умолчанию: 0)")
        print("  --pitch=N   - угол наклона камеры (-32 до +22, по умолчанию: 0)")
        print("  --fov=N     - угол обзора камеры (40-75, по умолчанию: 50)")
        print("  --duration=N- длительность теста в секундах (по умолчанию: 30)")
        print("\nПримеры:")
        print("  python3 test_full_pipeline.py left.mp4 right.mp4")
        print("  python3 test_full_pipeline.py left.mp4 right.mp4 --yaw=30 --pitch=10")
        print("  python3 test_full_pipeline.py left.mp4 right.mp4 --fov=55 --duration=60")
        sys.exit(1)

    left_file = sys.argv[1]
    right_file = sys.argv[2]

    # Парсинг аргументов
    yaw = 0
    pitch = 0
    fov = 50
    duration = 30

    for arg in sys.argv[3:]:
        if arg.startswith('--yaw='):
            yaw = float(arg.split('=')[1])
        elif arg.startswith('--pitch='):
            pitch = float(arg.split('=')[1])
        elif arg.startswith('--fov='):
            fov = float(arg.split('=')[1])
        elif arg.startswith('--duration='):
            duration = int(arg.split('=')[1])

    # Настройка окружения - ВАЖНО: Оба плагина в PATH
    current_dir = os.getcwd()
    stitch_dir = os.path.join(os.path.dirname(current_dir), 'my_steach')

    plugin_path = f"{current_dir}:{stitch_dir}:{os.environ.get('GST_PLUGIN_PATH', '')}"
    os.environ['GST_PLUGIN_PATH'] = plugin_path

    # Минимальные логи
    os.environ['GST_DEBUG'] = 'nvdsstitch:3,nvdsvirtualcam:3'

    print(f"📁 Plugin paths:")
    print(f"   • Stitch:  {stitch_dir}")
    print(f"   • VCam:    {current_dir}")
    print()

    # Запуск теста
    tester = FullPipelineFPSTester()

    try:
        if tester.run(left_file, right_file, yaw, pitch, fov, duration):
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
