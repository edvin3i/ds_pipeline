#!/usr/bin/env python3
"""
Упрощённый захват калибровочных фото (без превью)
Использует прямое сохранение в файл через multifilesink
"""

import os
import time
import subprocess
import argparse
from datetime import datetime
import json

def capture_images(cam_id, output_dir, count, interval):
    """
    Захват изображений с одной камеры без превью

    Args:
        cam_id: ID камеры
        output_dir: директория для сохранения
        count: количество изображений
        interval: интервал между снимками (сек)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем информацию
    setup_info = {
        "board_size": [10, 7],
        "square_size_mm": 23.5,
        "capture_date": datetime.now().isoformat(),
        "camera_setup": {
            "cam_id": cam_id,
            "resolution": {"width": 3840, "height": 2160}
        }
    }
    with open(os.path.join(output_dir, "capture_info.json"), "w") as f:
        json.dump(setup_info, f, indent=2)

    print("=" * 70)
    print("УПРОЩЁННЫЙ ЗАХВАТ (без превью)")
    print("=" * 70)
    print(f"📷 Камера ID: {cam_id}")
    print(f"📁 Директория: {output_dir}")
    print(f"🔢 Количество: {count}")
    print(f"⏱️  Интервал: {interval} сек")
    print("=" * 70)
    print("\n⚠️  ВАЖНО: Превью НЕТ! Используйте внешний монитор или")
    print("   запустите отдельно gst-launch для просмотра камеры\n")

    captured = 0

    try:
        while captured < count:
            # Формируем имя файла
            filename = os.path.join(output_dir, f"cam{cam_id}_{captured:05d}.jpg")

            # GStreamer команда для захвата одного кадра
            gst_cmd = [
                "gst-launch-1.0",
                "-e",
                f"nvarguscamerasrc", f"sensor-id={cam_id}",
                "num-buffers=1",  # Только 1 кадр
                "!",
                "video/x-raw(memory:NVMM),width=3840,height=2160,format=NV12",
                "!",
                "nvjpegenc", "quality=100",
                "!",
                f"filesink", f"location={filename}"
            ]

            print(f"\r⏰ Захват #{captured + 1}/{count}...", end='', flush=True)

            # Запускаем GStreamer
            result = subprocess.run(gst_cmd,
                                  capture_output=True,
                                  timeout=10,
                                  text=True)

            if result.returncode == 0 and os.path.exists(filename):
                captured += 1
                print(f"\r💾 Изображение #{captured}/{count} сохранено: {os.path.basename(filename)}")

                if captured < count:
                    print(f"⏱️  Следующий захват через {interval:.1f} сек. Переместите доску!\n")

                    # Обратный отсчёт
                    for remaining in range(int(interval), 0, -1):
                        if remaining <= 3:
                            print(f"⏰ {remaining}... НЕ ДВИГАЙТЕ доску!", flush=True)
                        time.sleep(1)
            else:
                print(f"\n❌ Ошибка захвата!")
                if result.stderr:
                    print(f"   {result.stderr[:200]}")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n⏸️  Остановка пользователем...")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

    print(f"\n{'=' * 70}")
    print(f"✅ Захвачено {captured} изображений")
    print(f"📁 Файлы: {output_dir}/")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='Упрощённый захват без превью')
    parser.add_argument('--cam-id', type=int, default=0, help='ID камеры')
    parser.add_argument('--output', '-o', default='simple_capture_data', help='Директория вывода')
    parser.add_argument('--count', '-n', type=int, default=25, help='Количество изображений')
    parser.add_argument('--interval', '-i', type=float, default=4.0, help='Интервал (сек)')

    args = parser.parse_args()

    capture_images(args.cam_id, args.output, args.count, args.interval)


if __name__ == "__main__":
    main()
