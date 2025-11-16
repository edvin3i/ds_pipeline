#!/usr/bin/env python3
"""
Сохранение панорамных кадров и тайлов из видеофайлов
Основан на panorama_stream.py с использованием рабочего пайплайна
"""

import sys
import os

# Добавляем путь к системным пакетам для виртуального окружения
# (решение проблемы с gi.repository в venv)
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime

Gst.init(None)


class PanoramaTileSaver:
    """Сохранение панорам и тайлов каждые N кадров"""

    def __init__(self, output_dir="panorama_output", interval=100, save_tiles=True):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        self.saved_count = 0
        self.interval = interval
        self.save_tiles = save_tiles
        self.start_time = None

        # Время запуска скрипта для имён файлов
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Координаты тайлов (из version_masr.py)
        self.tile_coords = [
            (192,  304, 1024, 1024),  # Tile 0
            (1216, 304, 1024, 1024),  # Tile 1
            (2240, 304, 1024, 1024),  # Tile 2
            (3264, 304, 1024, 1024),  # Tile 3
            (4288, 304, 1024, 1024),  # Tile 4
            (5312, 304, 1024, 1024),  # Tile 5
        ]

        # Создаём директории
        self.output_dir = Path(output_dir)
        self.setup_directories()

    def setup_directories(self):
        """Создание структуры директорий"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Директория для полных панорам
        self.pano_dir = self.output_dir / "panoramas"
        self.pano_dir.mkdir(exist_ok=True)

        if self.save_tiles:
            # Одна общая директория для всех тайлов
            self.tiles_dir = self.output_dir / "tiles"
            self.tiles_dir.mkdir(exist_ok=True)

        print(f"📁 Выходные директории созданы: {self.output_dir}")

    def create_pipeline(self, left_file, right_file):
        """
        Создает pipeline на основе рабочего кода из panorama_stream.py
        С добавлением appsink для сохранения кадров
        """

        # Pipeline строка - идентичная panorama_stream.py, но с appsink вместо дисплея
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

            nvdsstitch
                left-source-id=0
                right-source-id=1
                gpu-id=0 !

            queue max-size-buffers=3 !
            nvvideoconvert compute-hw=1 !
            video/x-raw,format=RGB !
            videoconvert !
            video/x-raw,format=BGR !
            appsink name=appsink
                emit-signals=true
                sync=false
                max-buffers=1
                drop=true
        """

        print("📋 Создание pipeline (на основе panorama_stream.py)")
        return Gst.parse_launch(pipeline_str)

    def on_new_sample(self, appsink):
        """Обработчик новых кадров из appsink"""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        self.frame_count += 1

        # Сохраняем только каждый N-й кадр
        if self.frame_count % self.interval == 0:
            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Получаем размеры из caps
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")

            # Извлекаем данные
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    # Преобразуем в numpy массив
                    data = np.frombuffer(map_info.data, dtype=np.uint8)

                    # BGR формат, 3 канала
                    if len(data) == width * height * 3:
                        panorama = data.reshape((height, width, 3))

                        # Генерируем имя файла с timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Сохраняем полную панораму
                        pano_filename = f"panorama_{self.saved_count:05d}_{timestamp}.jpg"
                        pano_path = self.pano_dir / pano_filename
                        cv2.imwrite(str(pano_path), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        # Извлекаем и сохраняем тайлы если нужно
                        if self.save_tiles:
                            self.extract_and_save_tiles(panorama, self.frame_count)

                        self.saved_count += 1

                        # Прогресс
                        elapsed = time.time() - self.start_time if self.start_time else 0
                        fps = self.frame_count / elapsed if elapsed > 0 else 0

                        print(f"📸 [{self.saved_count}] Кадр {self.frame_count}: "
                              f"сохранена панорама {pano_filename}"
                              f"{' + 6 тайлов' if self.save_tiles else ''} "
                              f"(FPS: {fps:.1f})")
                    else:
                        print(f"⚠️ Неверный размер данных на кадре {self.frame_count}")

                finally:
                    buffer.unmap(map_info)

        # Показываем прогресс каждые 30 кадров
        elif self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"⏳ Обработано кадров: {self.frame_count} (FPS: {fps:.1f})")

        return Gst.FlowReturn.OK

    def extract_and_save_tiles(self, panorama, frame_num):
        """Извлечение и сохранение тайлов из панорамы"""
        h, w = panorama.shape[:2]

        for i, (x, y, tw, th) in enumerate(self.tile_coords):
            # Проверяем границы
            if x + tw <= w and y + th <= h:
                # Вырезаем тайл
                tile = panorama[y:y+th, x:x+tw].copy()

                # Сохраняем тайл с форматом: номер_тайла_время_запуска_номер_кадра.jpg
                tile_filename = f"{i}_{self.run_timestamp}_{frame_num:06d}.jpg"
                tile_path = self.tiles_dir / tile_filename
                cv2.imwrite(str(tile_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def on_message(self, bus, message):
        """Обработчик сообщений GStreamer"""
        t = message.type

        if t == Gst.MessageType.EOS:
            print("\n🏁 Достигнут конец видео")
            self.loop.quit()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n❌ Ошибка: {err}")
            if debug:
                print(f"   Debug: {debug}")
            self.loop.quit()

        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            # Игнорируем предупреждения о NaN и out of range для LUT карт
            if "NaN" not in str(warn) and "out of range" not in str(warn):
                print(f"⚠️  Предупреждение: {warn}")

        elif t == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    self.start_time = time.time()
                    print("\n▶️  Pipeline запущен!")
                    print(f"📊 Сохранение каждые {self.interval} кадров...")
                    print("⏹️  Нажмите Ctrl+C для остановки\n")

    def run(self, left_file, right_file, max_frames=None):
        """Запуск обработки видео"""

        # Проверка файлов
        for f in [left_file, right_file]:
            if not os.path.exists(f):
                print(f"❌ Файл не найден: {f}")
                return False

        print("\n" + "="*70)
        print("🎬 ИЗВЛЕЧЕНИЕ ПАНОРАМ И ТАЙЛОВ")
        print("="*70)
        print(f"📹 Входные файлы:")
        print(f"   • Левый: {left_file}")
        print(f"   • Правый: {right_file}")
        print(f"📊 Параметры:")
        print(f"   • Интервал: каждые {self.interval} кадров")
        print(f"   • Сохранение тайлов: {'да' if self.save_tiles else 'нет'}")
        if max_frames:
            print(f"   • Максимум кадров: {max_frames}")
        print(f"   • Выход: {self.output_dir}/")
        print(f"   • Размер панорамы: 6528×1632")
        if self.save_tiles:
            print(f"   • Тайлы: 6 × 1024×1024")
        print("-"*70)

        # Создаем pipeline
        try:
            self.pipeline = self.create_pipeline(left_file, right_file)

            # Подключаем appsink
            appsink = self.pipeline.get_by_name("appsink")
            if not appsink:
                raise RuntimeError("appsink не найден в pipeline")

            appsink.connect("new-sample", self.on_new_sample)

        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Настройка bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        # Главный цикл
        self.loop = GLib.MainLoop()

        # Запуск pipeline
        print("\n⏳ Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            return False

        # Обработка
        try:
            # Если задан максимум кадров, используем таймер для остановки
            if max_frames:
                def check_max_frames():
                    if self.frame_count >= max_frames:
                        print(f"\n✅ Достигнут лимит: {max_frames} кадров")
                        self.loop.quit()
                        return False
                    return True

                GLib.timeout_add(100, check_max_frames)

            self.loop.run()

        except KeyboardInterrupt:
            print("\n⏹️  Остановлено пользователем")

        # Очистка
        print("\n🧹 Завершение...")
        self.pipeline.set_state(Gst.State.NULL)

        # Итоговая статистика
        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "="*70)
        print("📊 РЕЗУЛЬТАТЫ")
        print("="*70)
        print(f"⏱️  Время обработки: {elapsed:.1f} сек")
        print(f"📹 Обработано кадров: {self.frame_count}")
        print(f"💾 Сохранено наборов: {self.saved_count}")
        if self.save_tiles:
            print(f"🎯 Всего тайлов: {self.saved_count * 6}")
        if elapsed > 0:
            print(f"⚡ Средняя скорость: {self.frame_count/elapsed:.1f} fps")
        print(f"\n📁 Файлы сохранены в:")
        print(f"   • Панорамы: {self.pano_dir}/")
        if self.save_tiles:
            print(f"   • Тайлы: {self.tiles_dir}/ (формат: номер_{self.run_timestamp}_кадр.jpg)")
        print("="*70)

        return True


def main():
    """Главная функция"""
    if len(sys.argv) < 3:
        print("Использование: python3 panorama_tiles_saver.py left.mp4 right.mp4 [опции]")
        print("\nОпции:")
        print("  --interval N    - Сохранять каждый N-й кадр (по умолчанию: 100)")
        print("  --output DIR    - Выходная директория (по умолчанию: panorama_output)")
        print("  --no-tiles      - Не сохранять тайлы, только панорамы")
        print("  --max-frames N  - Максимальное количество кадров для обработки")
        print("\nПримеры:")
        print("  # Сохранить панорамы и тайлы каждые 100 кадров")
        print("  python3 panorama_tiles_saver.py left.mp4 right.mp4")
        print("\n  # Только панорамы каждые 50 кадров")
        print("  python3 panorama_tiles_saver.py left.mp4 right.mp4 --interval 50 --no-tiles")
        print("\n  # Обработать первые 1000 кадров")
        print("  python3 panorama_tiles_saver.py left.mp4 right.mp4 --max-frames 300")
        sys.exit(1)

    left_file = sys.argv[1]
    right_file = sys.argv[2]

    # Парсинг аргументов
    interval = 100
    output_dir = "panorama_output"
    save_tiles = True
    max_frames = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--interval" and i + 1 < len(sys.argv):
            interval = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-tiles":
            save_tiles = False
            i += 1
        elif sys.argv[i] == "--max-frames" and i + 1 < len(sys.argv):
            max_frames = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"❌ Неизвестная опция: {sys.argv[i]}")
            sys.exit(1)

    # Настройка путей к плагину
    plugin_path = os.getcwd()
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"

    # Минимальная отладка
    os.environ['GST_DEBUG'] = 'nvdsstitch:2'

    print(f"📁 Plugin path: {plugin_path}")

    # Создаем и запускаем приложение
    app = PanoramaTileSaver(
        output_dir=output_dir,
        interval=interval,
        save_tiles=save_tiles
    )

    try:
        if app.run(left_file, right_file, max_frames):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()