#!/usr/bin/env python3
"""
Синхронизированная запись двух 4K камер с hardware sync
Поддерживает master/slave синхронизацию для IMX678 и IMX477
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import signal
import sys
import time
import threading
from datetime import datetime
import os
import subprocess

class SyncedDualCameraRecorder:
    def __init__(self, master_id=0, slave_id=1):
        """
        Инициализация синхронизированного записывателя

        Args:
            master_id: ID мастер-камеры (обычно 0)
            slave_id: ID слейв-камеры (обычно 1)
        """
        Gst.init(None)

        # Параметры 4K записи
        self.width = 3840
        self.height = 2160
        self.fps = 30
        self.bitrate = 25000000  # 25 Мбит/с на камеру
        self.sensor_mode = 0  # Режим сенсора (0 = без HDR для IMX678)

        # ID камер
        self.master_id = master_id
        self.slave_id = slave_id

        # Пайплайны
        self.pipeline_master = None
        self.pipeline_slave = None
        self.running = True

        # Общие часы для синхронизации
        self.base_clock = None

        # Файлы
        self.master_file = None
        self.slave_file = None

        # Статистика
        self.start_time = None
        self.master_started = False
        self.slave_started = False

    def setup_hardware_sync(self, enable=True):
        """
        Настраивает hardware sync через V4L2 controls

        Для IMX678/IMX477:
        - operation_mode=0: Master mode
        - operation_mode=1: Slave mode
        - synchronizing_function=1: Master sync output
        - synchronizing_function=2: Slave sync input

        Args:
            enable: True для включения hardware sync
        """
        if not enable:
            print("[INFO] ⚠️ Hardware sync отключен")
            return True

        print("\n[INFO] 🔧 Настройка hardware sync через V4L2...")

        # Определяем video devices для каждой камеры
        # Обычно sensor-id=0 -> /dev/video0, sensor-id=1 -> /dev/video1
        # Но может быть и наоборот, поэтому проверим
        master_video_dev = f"/dev/video{self.master_id}"
        slave_video_dev = f"/dev/video{self.slave_id}"

        try:
            # Настраиваем мастер-камеру (operation_mode=0, synchronizing_function=1)
            print(f"[INFO] 🎯 Настраиваем мастер: {master_video_dev}")
            cmd_master = [
                'v4l2-ctl', '-d', master_video_dev,
                '-c', 'operation_mode=0',
                '-c', 'synchronizing_function=1'
            ]
            result = subprocess.run(cmd_master, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                print(f"[WARNING] ⚠️ Не удалось настроить мастер: {result.stderr}")
                print(f"[INFO] 💡 Возможно камеры не поддерживают эти controls или используют другие /dev/video*")
                return False
            else:
                print(f"[INFO] ✅ Мастер настроен: operation_mode=0, synchronizing_function=1")

            # Настраиваем слейв-камеру (operation_mode=1, synchronizing_function=2)
            print(f"[INFO] 🎯 Настраиваем слейв: {slave_video_dev}")
            cmd_slave = [
                'v4l2-ctl', '-d', slave_video_dev,
                '-c', 'operation_mode=1',
                '-c', 'synchronizing_function=2'
            ]
            result = subprocess.run(cmd_slave, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                print(f"[WARNING] ⚠️ Не удалось настроить слейв: {result.stderr}")
                return False
            else:
                print(f"[INFO] ✅ Слейв настроен: operation_mode=1, synchronizing_function=2")

            print("[INFO] ✅ Hardware sync успешно настроен!")
            print("[INFO] 💡 Убедитесь что sync провод физически соединяет камеры")
            return True

        except subprocess.TimeoutExpired:
            print("[ERROR] ❌ Timeout при настройке V4L2 controls")
            return False
        except FileNotFoundError:
            print("[ERROR] ❌ v4l2-ctl не найден. Установите: sudo apt install v4l-utils")
            return False
        except Exception as e:
            print(f"[ERROR] ❌ Ошибка настройки hardware sync: {e}")
            return False

    def create_camera_pipeline(self, sensor_id, output_file, is_master=True, codec='h264'):
        """
        Создает пайплайн для одной камеры с поддержкой синхронизации

        Args:
            sensor_id: ID сенсора камеры
            output_file: Имя выходного файла
            is_master: True если это мастер-камера
            codec: Кодек для записи (h264 или h265)
        """

        if codec == 'h264':
            encoder = f"nvv4l2h264enc bitrate={self.bitrate} maxperf-enable=1 iframeinterval=30"
            parser = "h264parse"
        else:
            encoder = f"nvv4l2h265enc bitrate={self.bitrate-5000000} maxperf-enable=1 iframeinterval=30"
            parser = "h265parse"

        # Для синхронизации важно:
        # 1. do-timestamp=true - применять stream time к буферам
        # 2. Одинаковые параметры для обеих камер
        # 3. sensor-mode=0 для IMX678 без HDR
        # 4. mp4mux faststart=true для корректного воспроизведения
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={sensor_id}
            sensor-mode={self.sensor_mode}
            do-timestamp=true
            ! video/x-raw(memory:NVMM), width={self.width}, height={self.height},
            format=NV12, framerate={self.fps}/1 !
            nvvideoconvert !
            video/x-raw(memory:NVMM), format=I420 !
            {encoder} !
            {parser} !
            mp4mux faststart=true fragment-duration=1000 !
            filesink location={output_file} sync=false
        """

        camera_type = "Мастер" if is_master else "Слейв"
        print(f"[INFO] 📹 {camera_type} камера (ID {sensor_id}) -> {output_file}")

        pipeline = Gst.parse_launch(pipeline_str)

        # Подключаем обработчик сообщений
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda bus, msg: self.on_message(bus, msg, sensor_id, is_master))

        return pipeline

    def on_message(self, bus, message, sensor_id, is_master):
        """Обработчик сообщений GStreamer"""
        msg_type = message.type
        camera_type = "Мастер" if is_master else "Слейв"

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] ❌ {camera_type}: {err}")
            print(f"[DEBUG] {debug}")
            self.stop()

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] ⚠️ {camera_type}: {warn}")

        elif msg_type == Gst.MessageType.EOS:
            print(f"[INFO] ✅ {camera_type}: Запись завершена корректно")

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == (self.pipeline_master if is_master else self.pipeline_slave):
                old_state, new_state, pending_state = message.parse_state_changed()

                if new_state == Gst.State.PLAYING:
                    timestamp = time.time()
                    if is_master:
                        self.master_started = timestamp
                        print(f"[INFO] 🎬 {camera_type}: Запись началась! (t={timestamp:.6f})")
                    else:
                        self.slave_started = timestamp
                        print(f"[INFO] 🎬 {camera_type}: Запись началась! (t={timestamp:.6f})")

                    # Вычисляем разницу во времени старта
                    if self.master_started and self.slave_started:
                        diff_ms = abs(self.master_started - self.slave_started) * 1000
                        diff_frames = diff_ms / (1000.0 / self.fps)
                        print(f"[INFO] 📊 Разница старта камер: {diff_ms:.2f}мс (~{diff_frames:.1f} кадров)")

    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        print(f"\n[INFO] 🛑 Получен сигнал {signum}, останавливаем запись...")
        self.stop()

    def stop(self):
        """Корректная остановка с синхронизацией"""
        print("[INFO] ⏹️ Останавливаем записи...")
        self.running = False

        # Останавливаем пайплайны ОДНОВРЕМЕННО
        print("[INFO] 📹 Отправляем EOS обеим камерам...")
        if self.pipeline_master:
            self.pipeline_master.send_event(Gst.Event.new_eos())
        if self.pipeline_slave:
            self.pipeline_slave.send_event(Gst.Event.new_eos())

        # Ждем обработки EOS
        time.sleep(2)

        # Устанавливаем NULL состояние
        if self.pipeline_master:
            self.pipeline_master.set_state(Gst.State.NULL)
        if self.pipeline_slave:
            self.pipeline_slave.set_state(Gst.State.NULL)

        print("[INFO] ✅ Все записи остановлены")

        # Вызываем callback для остановки loop
        if hasattr(self, 'stop_callback') and self.stop_callback:
            GLib.idle_add(self.stop_callback)

    def run_synced_recording(self, codec='h264', use_shared_clock=True, hardware_sync=True):
        """
        Запуск синхронизированной записи

        Args:
            codec: Кодек (h264 или h265)
            use_shared_clock: Использовать общие часы для пайплайнов
            hardware_sync: Использовать аппаратную синхронизацию через V4L2
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_file = f"camera_master_{timestamp}.mp4"
        self.slave_file = f"camera_slave_{timestamp}.mp4"

        print("🎬 СИНХРОНИЗИРОВАННАЯ ЗАПИСЬ 4K КАМЕР")
        print("=" * 50)
        print(f"[INFO] 🚀 Режим: Hardware sync (Master-Slave)")
        print(f"[INFO] 🎯 Кодек: {codec.upper()}")
        print(f"[INFO] 📊 Разрешение: {self.width}x{self.height} @ {self.fps}fps")
        print(f"[INFO] 💾 Битрейт: {self.bitrate//1000000} Мбит/с на камеру")
        print(f"[INFO] 📷 Sensor mode: {self.sensor_mode} {'(без HDR)' if self.sensor_mode == 0 else '(HDR/другой)'}")
        print(f"[INFO] 🔄 Общие часы: {'Да' if use_shared_clock else 'Нет'}")
        print(f"[INFO] 🔌 Hardware sync: {'Да' if hardware_sync else 'Нет'}")
        print(f"[INFO] 📁 Мастер камера (ID {self.master_id}): {self.master_file}")
        print(f"[INFO] 📁 Слейв камера (ID {self.slave_id}): {self.slave_file}")
        print(f"[INFO] 🛑 Для остановки нажмите Ctrl+C")
        print()

        # Настраиваем hardware sync через V4L2 controls ПЕРЕД созданием пайплайнов
        if hardware_sync:
            if not self.setup_hardware_sync(enable=True):
                print("[WARNING] ⚠️ Hardware sync не удалось настроить, продолжаем без него")
                print("[INFO] 💡 Запись будет использовать только программную синхронизацию")

        # Устанавливаем обработчик сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Создаем пайплайны
        try:
            print("[INFO] 🔧 Создаем пайплайны...")
            self.pipeline_master = self.create_camera_pipeline(
                self.master_id, self.master_file, is_master=True, codec=codec
            )
            self.pipeline_slave = self.create_camera_pipeline(
                self.slave_id, self.slave_file, is_master=False, codec=codec
            )
        except Exception as e:
            print(f"[ERROR] ❌ Не удалось создать пайплайны: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Настраиваем общие часы для синхронизации
        if use_shared_clock:
            print("[INFO] 🕐 Настраиваем общие часы для синхронизации...")
            self.base_clock = Gst.SystemClock.obtain()
            self.pipeline_master.use_clock(self.base_clock)
            self.pipeline_slave.use_clock(self.base_clock)

            # Устанавливаем одинаковое base time
            base_time = self.base_clock.get_time()
            self.pipeline_master.set_base_time(base_time)
            self.pipeline_slave.set_base_time(base_time)
            print(f"[INFO] ⏰ Base time: {base_time}")

        # МЕТОД 1: Синхронизация через PAUSED -> PLAYING
        print("\n[INFO] 🔄 Метод синхронизации: PAUSED -> PLAYING")
        print("[INFO] 🔧 Переводим камеры в PAUSED для инициализации...")

        # Переводим обе камеры в PAUSED одновременно
        ret1 = self.pipeline_master.set_state(Gst.State.PAUSED)
        ret2 = self.pipeline_slave.set_state(Gst.State.PAUSED)

        if ret1 == Gst.StateChangeReturn.FAILURE or ret2 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Не удалось перевести камеры в PAUSED")
            return False

        # Ждем завершения перехода в PAUSED (блокирующий вызов)
        print("[INFO] ⏳ Ждем инициализации мастер-камеры...")
        ret1, state1, pending1 = self.pipeline_master.get_state(5 * Gst.SECOND)
        if ret1 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Мастер-камера не перешла в PAUSED")
            return False
        print(f"[INFO] ✅ Мастер-камера в состоянии: {state1.value_nick}")

        print("[INFO] ⏳ Ждем инициализации слейв-камеры...")
        ret2, state2, pending2 = self.pipeline_slave.get_state(5 * Gst.SECOND)
        if ret2 == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] ❌ Слейв-камера не перешла в PAUSED")
            return False
        print(f"[INFO] ✅ Слейв-камера в состоянии: {state2.value_nick}")

        # Теперь переводим обе камеры в PLAYING ОДНОВРЕМЕННО
        print("\n[INFO] ▶️ Запускаем обе камеры СИНХРОННО...")
        print("[INFO] 🎯 Переводим в PLAYING без задержки...")

        start_time = time.time()
        self.pipeline_master.set_state(Gst.State.PLAYING)
        self.pipeline_slave.set_state(Gst.State.PLAYING)
        end_time = time.time()

        launch_diff_ms = (end_time - start_time) * 1000
        print(f"[INFO] ⚡ Время между запусками: {launch_diff_ms:.2f}мс")

        self.start_time = time.time()
        print("\n[INFO] ✅ Обе камеры запущены!")
        print("[INFO] 📽️ Запись в процессе...")
        print("[INFO] 🛑 Нажмите Ctrl+C для остановки записи")

        # Создаем главный цикл
        loop = GLib.MainLoop()

        # Статистика каждые 10 секунд
        def show_progress():
            if self.running and self.start_time:
                elapsed = time.time() - self.start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                # Проверяем размеры файлов
                size_info = ""
                if os.path.exists(self.master_file):
                    size_master_mb = os.path.getsize(self.master_file) // 1024 // 1024
                    size_info += f"M: {size_master_mb}МБ"

                if os.path.exists(self.slave_file):
                    size_slave_mb = os.path.getsize(self.slave_file) // 1024 // 1024
                    if size_info:
                        size_info += ", "
                    size_info += f"S: {size_slave_mb}МБ"

                print(f"[INFO] ⏱️ Время записи: {minutes:02d}:{seconds:02d} | {size_info}")
            return self.running

        GLib.timeout_add_seconds(10, show_progress)

        # Обработчик для корректной остановки loop
        def stop_loop():
            loop.quit()
            return False

        self.stop_callback = stop_loop

        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] 🛑 Прерывание пользователем...")
            self.stop()

        # Проверяем результаты
        print("\n[INFO] 📊 Анализируем результаты...")
        return self.analyze_results()

    def analyze_results(self):
        """Анализирует результаты записи"""
        success = True
        total_size = 0

        # Вычисляем фактическую длительность
        if self.start_time:
            actual_duration = time.time() - self.start_time
        else:
            actual_duration = 0

        minutes = int(actual_duration // 60)
        seconds = int(actual_duration % 60)

        print("[INFO] 📋 Результаты синхронизированной записи:")
        print(f"[INFO] ⏱️ Общая длительность: {minutes:02d}:{seconds:02d}")

        # Анализируем файл мастер-камеры
        if os.path.exists(self.master_file):
            size_master = os.path.getsize(self.master_file)
            size_mb = size_master // 1024 // 1024
            total_size += size_master

            print(f"[INFO] 📁 Мастер камера: {size_mb} МБ")

            if size_master < 5 * 1024 * 1024:
                print("[WARNING] ⚠️ Файл мастера слишком маленький")
                success = False
            elif actual_duration > 0:
                actual_bitrate = (size_master * 8) / actual_duration / 1000000
                print(f"[INFO] 📊 Мастер: ~{actual_bitrate:.1f} Мбит/с")
        else:
            print("[ERROR] ❌ Файл мастера не создан")
            success = False

        # Анализируем файл слейв-камеры
        if os.path.exists(self.slave_file):
            size_slave = os.path.getsize(self.slave_file)
            size_mb = size_slave // 1024 // 1024
            total_size += size_slave

            print(f"[INFO] 📁 Слейв камера: {size_mb} МБ")

            if size_slave < 5 * 1024 * 1024:
                print("[WARNING] ⚠️ Файл слейва слишком маленький")
                success = False
            elif actual_duration > 0:
                actual_bitrate = (size_slave * 8) / actual_duration / 1000000
                print(f"[INFO] 📊 Слейв: ~{actual_bitrate:.1f} Мбит/с")
        else:
            print("[ERROR] ❌ Файл слейва не создан")
            success = False

        # Анализ синхронизации
        if self.master_started and self.slave_started:
            diff_ms = abs(self.master_started - self.slave_started) * 1000
            diff_frames = diff_ms / (1000.0 / self.fps)
            print(f"\n[INFO] 🔄 Анализ синхронизации:")
            print(f"[INFO] ⏱️ Разница старта: {diff_ms:.2f}мс")
            print(f"[INFO] 🎞️ Разница в кадрах: ~{diff_frames:.1f} кадров")

            if diff_frames < 1.0:
                print(f"[INFO] ✅ Отличная синхронизация! (< 1 кадра)")
            elif diff_frames < 3.0:
                print(f"[INFO] ✅ Хорошая синхронизация (< 3 кадров)")
            elif diff_frames < 5.0:
                print(f"[WARNING] ⚠️ Приемлемая синхронизация (< 5 кадров)")
            else:
                print(f"[WARNING] ⚠️ Слабая синхронизация (>= 5 кадров)")
                print(f"[INFO] 💡 Рекомендации:")
                print(f"    • Проверьте физическое подключение sync провода")
                print(f"    • Убедитесь, что в device tree настроен fsync-mode")
                print(f"    • Попробуйте использовать --use-shared-clock")

        # Общая статистика
        if success:
            total_mb = total_size // 1024 // 1024
            total_gb = total_mb / 1024
            print(f"\n[INFO] 📊 Общий размер: {total_mb} МБ ({total_gb:.2f} ГБ)")

            if actual_duration > 0:
                mb_per_sec = total_mb / actual_duration
                print(f"[INFO] 💾 Скорость записи: ~{mb_per_sec:.1f} МБ/сек")

            print("[INFO] ✅ Запись завершена успешно!")

            # Советы по использованию
            print(f"\n[INFO] 💡 Советы по использованию:")
            print(f"• Просмотр: vlc {self.master_file}")
            print(f"• Склейка: ffmpeg -i {self.master_file} -i {self.slave_file} -filter_complex hstack output.mp4")
            print(f"• Проверка синхронизации: ffplay -i {self.master_file} (сравните с {self.slave_file})")

        return success


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Синхронизированная двойная запись 4K камер (Master-Slave)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая запись с синхронизацией
  python3 synced_dual_record.py

  # Указать ID мастер и слейв камер
  python3 synced_dual_record.py --master 0 --slave 1

  # Использовать H.265 кодек
  python3 synced_dual_record.py --codec h265

  # Отключить общие часы (если есть проблемы)
  python3 synced_dual_record.py --no-shared-clock

  # Отключить hardware sync (только программная синхронизация)
  python3 synced_dual_record.py --no-hardware-sync

  # Повысить битрейт
  python3 synced_dual_record.py --bitrate 35

  # Использовать другой sensor-mode (для HDR или других режимов)
  python3 synced_dual_record.py --sensor-mode 1

Примечания:
  • Для hardware sync нужно физически соединить sync пины камер
  • Скрипт автоматически настраивает V4L2 controls для master/slave режима
  • IMX678 и IMX477 поддерживают hardware sync
  • sensor-mode=0 используется по умолчанию (без HDR для IMX678)
        """
    )

    parser.add_argument('--master', type=int, default=0,
                       help='ID мастер-камеры (по умолчанию: 0)')
    parser.add_argument('--slave', type=int, default=1,
                       help='ID слейв-камеры (по умолчанию: 1)')
    parser.add_argument('--codec', choices=['h264', 'h265'], default='h264',
                       help='Кодек для записи (по умолчанию: h264)')
    parser.add_argument('--bitrate', type=int, default=25,
                       help='Битрейт в Мбит/с (по умолчанию: 25)')
    parser.add_argument('--sensor-mode', type=int, default=0,
                       help='Режим сенсора: 0=без HDR (по умолчанию), 1=HDR и др.')
    parser.add_argument('--no-shared-clock', action='store_true',
                       help='Не использовать общие часы (может помочь при проблемах)')
    parser.add_argument('--no-hardware-sync', action='store_true',
                       help='Не использовать аппаратную синхронизацию через V4L2')

    args = parser.parse_args()

    recorder = SyncedDualCameraRecorder(master_id=args.master, slave_id=args.slave)

    # Настраиваем битрейт если указан
    if args.bitrate != 25:
        recorder.bitrate = args.bitrate * 1000000
        print(f"[INFO] 🎯 Пользовательский битрейт: {args.bitrate} Мбит/с")

    # Настраиваем sensor-mode если указан
    if args.sensor_mode != 0:
        recorder.sensor_mode = args.sensor_mode
        print(f"[INFO] 🎯 Режим сенсора: {args.sensor_mode}")

    # Запускаем синхронизированную запись
    use_shared_clock = not args.no_shared_clock
    use_hardware_sync = not args.no_hardware_sync
    success = recorder.run_synced_recording(args.codec, use_shared_clock, use_hardware_sync)

    if not success:
        print("\n[ERROR] ❌ Запись не удалась")
        sys.exit(1)
    else:
        print("\n[INFO] 🎉 Успех! Два синхронизированных 4K файла готовы!")


if __name__ == "__main__":
    main()
