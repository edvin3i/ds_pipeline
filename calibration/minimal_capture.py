#!/usr/bin/env python3
"""
Минимальный скрипт захвата - автоматически делает фото с интервалом
ESC или Ctrl+C для выхода
"""

import cv2
import os
import argparse
import time
from datetime import datetime

def capture_images(cam_id, output_dir, max_count, interval):
    """
    Простой захват с показом превью и автоматическим таймером
    ESC или Ctrl+C - выход
    """
    os.makedirs(output_dir, exist_ok=True)

    # Для Jetson используем GStreamer pipeline
    # CSI камеры доступны только через nvarguscamerasrc
    cam_width = 3840
    cam_height = 2160

    gst_pipeline = (
        f"nvarguscamerasrc sensor-id={cam_id} sensor-mode=0 ! "
        f"video/x-raw(memory:NVMM),width={cam_width},height={cam_height},format=NV12,framerate=30/1 ! "
        f"nvvideoconvert ! "
        f"video/x-raw,format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink"
    )

    # Открываем камеру через GStreamer
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {cam_id}")
        print("💡 Проверьте:")
        print(f"   1. Камера подключена и работает: ls /dev/video*")
        print(f"   2. Попробуйте другой sensor-id (0 или 1)")
        print(f"   3. Убедитесь, что камера не используется другим процессом")
        return

    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ ЗАХВАТ КАЛИБРОВОЧНЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 60)
    print(f"📷 Камера: {cam_id}")
    print(f"📁 Сохранение: {output_dir}/")
    print(f"🔢 Количество: {max_count}")
    print(f"⏱️  Интервал: {interval} сек")
    print("\nУправление:")
    print("  ESC - выход")
    print("=" * 60)
    print("\n💡 Перемещайте доску в разные позиции между снимками!\n")

    count = 0
    window_name = f"Камера {cam_id} - ESC=выход"
    last_capture_time = time.time() - interval + 3  # Первое фото через 3 сек

    try:
        while count < max_count:
            ret, frame = cap.read()

            if not ret:
                print("❌ Ошибка чтения кадра")
                break

            current_time = time.time()
            time_since_capture = current_time - last_capture_time
            remaining = max(0, interval - time_since_capture)

            # Показываем превью (уменьшенное для скорости)
            preview = cv2.resize(frame, (1280, 720))

            # Добавляем текст на экран
            text1 = f"Снято: {count}/{max_count}"
            text2 = f"След. фото через: {remaining:.1f} сек" if remaining > 0 else "СНИМАЕМ..."

            cv2.putText(preview, text1, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(preview, text2, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Если осталось меньше 3 секунд - предупреждение
            if 0 < remaining <= 3:
                warning = "НЕ ДВИГАЙТЕ ДОСКУ!"
                cv2.putText(preview, warning, (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow(window_name, preview)

            # Проверяем таймер
            if time_since_capture >= interval:
                filename = os.path.join(output_dir, f"cam{cam_id}_{count:05d}.jpg")
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count += 1
                last_capture_time = current_time
                print(f"💾 [{count}/{max_count}] Сохранено: {os.path.basename(filename)}")

                if count < max_count:
                    print(f"⏱️  Следующее фото через {interval} сек. Переместите доску!\n")

            # ESC - выход
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                print("\n⏸️  Выход...")
                break

    except KeyboardInterrupt:
        print("\n⏸️  Ctrl+C - выход...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'=' * 60}")
        print(f"✅ Всего сохранено: {count} изображений")
        print(f"📁 Директория: {output_dir}/")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='Автоматический захват калибровочных фото')
    parser.add_argument('--cam-id', type=int, default=0, help='ID камеры (0, 1, 2...)')
    parser.add_argument('--output', '-o', default='calibration_images', help='Директория')
    parser.add_argument('--count', '-n', type=int, default=30, help='Количество фото')
    parser.add_argument('--interval', '-i', type=float, default=4.0, help='Интервал (сек)')

    args = parser.parse_args()
    capture_images(args.cam_id, args.output, args.count, args.interval)


if __name__ == "__main__":
    main()
