#!/usr/bin/env python3
"""
Синхронизированная запись двух 4K камер с hardware sync
ROBUST версия - использует qtmux для более надёжной записи
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import signal
import sys
import time
import subprocess
from datetime import datetime
import os

# Импортируем базовый класс
import importlib.util
spec = importlib.util.spec_from_file_location("synced_dual_record",
                                               "/home/nvidia/deep_cv_football/soft_record_video/synced_dual_record.py")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

class RobustSyncedDualCameraRecorder(base_module.SyncedDualCameraRecorder):
    """
    Робастная версия записывателя с qtmux вместо mp4mux
    Лучше справляется с прерываниями и EOS
    """

    def create_camera_pipeline(self, sensor_id, output_file, is_master=True, codec='h264'):
        """
        Создает пайплайн для одной камеры с qtmux
        """

        if codec == 'h264':
            encoder = f"nvv4l2h264enc bitrate={self.bitrate} maxperf-enable=1 iframeinterval=30"
            parser = "h264parse"
        else:
            encoder = f"nvv4l2h265enc bitrate={self.bitrate-5000000} maxperf-enable=1 iframeinterval=30"
            parser = "h265parse"

        # Используем qtmux вместо mp4mux:
        # - moov-recovery-file - создаёт recovery файл для восстановления
        # - fragment-duration - пишет данные периодически (каждую секунду)
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
            qtmux fragment-duration=1000 moov-recovery-file={output_file}.recovery !
            filesink location={output_file} sync=false
        """

        camera_type = "Мастер" if is_master else "Слейв"
        print(f"[INFO] 📹 {camera_type} камера (ID {sensor_id}) -> {output_file}")
        print(f"[INFO] 🔄 Recovery файл: {output_file}.recovery")

        pipeline = Gst.parse_launch(pipeline_str)

        # Подключаем обработчик сообщений
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda bus, msg: self.on_message(bus, msg, sensor_id, is_master))

        return pipeline

    def stop(self):
        """Корректная остановка с cleanup recovery файлов"""
        print("[INFO] ⏹️ Останавливаем записи...")
        self.running = False

        # Останавливаем пайплайны ОДНОВРЕМЕННО
        print("[INFO] 📹 Отправляем EOS обеим камерам...")
        if self.pipeline_master:
            self.pipeline_master.send_event(Gst.Event.new_eos())
        if self.pipeline_slave:
            self.pipeline_slave.send_event(Gst.Event.new_eos())

        # Ждем обработки EOS (дольше для qtmux)
        time.sleep(3)

        # Устанавливаем NULL состояние
        if self.pipeline_master:
            self.pipeline_master.set_state(Gst.State.NULL)
        if self.pipeline_slave:
            self.pipeline_slave.set_state(Gst.State.NULL)

        print("[INFO] ✅ Все записи остановлены")

        # Удаляем recovery файлы если запись успешна
        for recovery_file in [f"{self.master_file}.recovery", f"{self.slave_file}.recovery"]:
            if os.path.exists(recovery_file):
                try:
                    os.remove(recovery_file)
                    print(f"[INFO] 🗑️ Удалён recovery файл: {recovery_file}")
                except:
                    pass

        # Вызываем callback для остановки loop
        if hasattr(self, 'stop_callback') and self.stop_callback:
            GLib.idle_add(self.stop_callback)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Робастная синхронизированная двойная запись 4K камер (qtmux)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая запись с синхронизацией
  python3 synced_dual_record_robust.py

  # С опциями
  python3 synced_dual_record_robust.py --codec h265 --bitrate 35

Отличия от обычной версии:
  • Использует qtmux вместо mp4mux
  • Создаёт recovery файлы для восстановления при сбое
  • Более надёжно при прерываниях (Ctrl+C)
  • Fragment-based запись каждую секунду
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

    recorder = RobustSyncedDualCameraRecorder(master_id=args.master, slave_id=args.slave)

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
