#!/usr/bin/env python3
"""
Визуальный тест калибровки камеры
Захватывает кадр, применяет калибровку и показывает результат
"""

import cv2
import numpy as np
import pickle
import sys
import os

def create_gstreamer_pipeline(sensor_id=0, width=3840, height=2160):
    """Создает GStreamer pipeline для захвата с камеры"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=0 ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"format=NV12,framerate=30/1 ! "
        f"nvvideoconvert ! "
        f"video/x-raw,format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink sync=false"
    )

def capture_frame(camera_id=0):
    """Захватывает один кадр с камеры"""
    print(f"📷 Захват кадра с камеры {camera_id}...")

    pipeline = create_gstreamer_pipeline(camera_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {camera_id}")
        return None

    # Пропускаем первые кадры (инициализация)
    for _ in range(10):
        cap.read()

    # Захватываем кадр
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Не удалось захватить кадр")
        return None

    print(f"✅ Кадр захвачен: {frame.shape}")
    return frame

def load_calibration(calibration_file):
    """Загружает файл калибровки"""
    print(f"📂 Загрузка калибровки из {calibration_file}...")

    if not os.path.exists(calibration_file):
        print(f"❌ Файл калибровки не найден: {calibration_file}")
        return None

    with open(calibration_file, 'rb') as f:
        calib = pickle.load(f)

    print("✅ Калибровка загружена")
    return calib

def apply_calibration_standard(frame, camera_calib):
    """Применяет стандартную калибровку к кадру"""
    print("🔧 Применение стандартной калибровки...")

    mtx = camera_calib['mtx']
    dist = camera_calib['dist']
    h, w = frame.shape[:2]

    # Получаем оптимальную матрицу камеры
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Применяем undistort
    undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)

    # Обрезаем по ROI
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
    else:
        undistorted_cropped = undistorted

    print(f"✅ Калибровка применена")
    print(f"   Оригинал: {frame.shape}")
    print(f"   После коррекции: {undistorted.shape}")
    print(f"   После обрезки ROI: {undistorted_cropped.shape}")

    return undistorted, undistorted_cropped

def apply_calibration_fisheye(frame, camera_calib):
    """Применяет fisheye калибровку к кадру"""
    print("🔧 Применение fisheye калибровки...")

    K = camera_calib['K']
    D = camera_calib['D']
    h, w = frame.shape[:2]

    # Получаем оптимальную матрицу
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )

    # Создаем карты для remap
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    # Применяем remap
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    print(f"✅ Калибровка применена")
    print(f"   Оригинал: {frame.shape}")
    print(f"   После коррекции: {undistorted.shape}")

    return undistorted, undistorted

def draw_grid(img, color=(0, 255, 0)):
    """Рисует сетку на изображении для проверки искажений"""
    h, w = img.shape[:2]
    result = img.copy()

    # Вертикальные линии
    for x in range(0, w, w//10):
        cv2.line(result, (x, 0), (x, h), color, 2)

    # Горизонтальные линии
    for y in range(0, h, h//10):
        cv2.line(result, (0, y), (w, y), color, 2)

    return result

def create_comparison_image(original, undistorted, undistorted_cropped):
    """Создает изображение для сравнения"""
    print("🖼️  Создание сравнительного изображения...")

    # Уменьшаем для отображения (4K -> 1920x1080)
    scale = 0.5

    def resize_img(img):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    original_small = resize_img(original)
    undistorted_small = resize_img(undistorted)
    undistorted_cropped_small = resize_img(undistorted_cropped)

    # Добавляем сетку для наглядности
    original_grid = draw_grid(original_small, (0, 255, 0))
    undistorted_grid = draw_grid(undistorted_small, (0, 255, 0))
    undistorted_cropped_grid = draw_grid(undistorted_cropped_small, (0, 255, 0))

    # Добавляем текст
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_grid, "ORIGINAL (with distortion)", (10, 40),
                font, 1, (0, 255, 0), 2)
    cv2.putText(undistorted_grid, "UNDISTORTED (full)", (10, 40),
                font, 1, (0, 255, 0), 2)
    cv2.putText(undistorted_cropped_grid, "UNDISTORTED (cropped ROI)", (10, 40),
                font, 1, (0, 255, 0), 2)

    # Объединяем вертикально
    h, w = original_small.shape[:2]

    # Приводим все к одному размеру (берем максимальную ширину)
    max_w = max(original_small.shape[1], undistorted_small.shape[1],
                undistorted_cropped_small.shape[1])

    def pad_to_width(img, target_w):
        h, w = img.shape[:2]
        if w < target_w:
            padding = np.zeros((h, target_w - w, 3), dtype=np.uint8)
            return np.hstack([img, padding])
        return img

    original_padded = pad_to_width(original_grid, max_w)
    undistorted_padded = pad_to_width(undistorted_grid, max_w)
    undistorted_cropped_padded = pad_to_width(undistorted_cropped_grid, max_w)

    # Объединяем
    comparison = np.vstack([
        original_padded,
        undistorted_padded,
        undistorted_cropped_padded
    ])

    print("✅ Сравнительное изображение создано")
    return comparison

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Визуальный тест калибровки')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID камеры (0=левая, 1=правая)')
    parser.add_argument('--calibration', '-c', default='calibration_result_standard.pkl',
                       help='Файл калибровки')
    parser.add_argument('--output', '-o', default='calibration_test',
                       help='Префикс выходных файлов')
    parser.add_argument('--image', '-i', default=None,
                       help='Использовать существующее изображение вместо захвата с камеры')

    args = parser.parse_args()

    print("="*60)
    print("🔬 ВИЗУАЛЬНЫЙ ТЕСТ КАЛИБРОВКИ")
    print("="*60)

    # 1. Загружаем калибровку
    calib = load_calibration(args.calibration)
    if calib is None:
        print("\n❌ Не удалось загрузить калибровку")
        print(f"   Проверьте что файл существует: {args.calibration}")
        return

    # Определяем тип калибровки
    camera_key = 'left_camera' if args.camera == 0 else 'right_camera'
    if camera_key not in calib:
        print(f"❌ В калибровке нет данных для {camera_key}")
        return

    camera_calib = calib[camera_key]
    is_fisheye = camera_calib.get('model') == 'fisheye'

    print(f"\n📋 Информация о калибровке:")
    print(f"   Камера: {camera_key}")
    print(f"   Модель: {'fisheye' if is_fisheye else 'standard'}")
    print(f"   RMS ошибка: {camera_calib['ret']:.3f}")
    if not is_fisheye and 'fovx' in camera_calib:
        print(f"   FOV: {camera_calib['fovx']:.1f}° x {camera_calib['fovy']:.1f}°")

    # 2. Захватываем кадр или загружаем из файла
    print()
    if args.image:
        print(f"📂 Загрузка изображения из {args.image}...")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Не удалось загрузить изображение: {args.image}")
            return
        print(f"✅ Изображение загружено: {frame.shape}")
    else:
        frame = capture_frame(args.camera)
        if frame is None:
            print("\n❌ Не удалось захватить кадр")
            return

    # Сохраняем оригинал
    original_file = f"{args.output}_original.jpg"
    cv2.imwrite(original_file, frame)
    print(f"💾 Оригинал сохранен: {original_file}")

    # 3. Применяем калибровку
    print()
    if is_fisheye:
        undistorted, undistorted_cropped = apply_calibration_fisheye(frame, camera_calib)
    else:
        undistorted, undistorted_cropped = apply_calibration_standard(frame, camera_calib)

    # Сохраняем результаты
    undistorted_file = f"{args.output}_undistorted.jpg"
    undistorted_cropped_file = f"{args.output}_undistorted_cropped.jpg"

    cv2.imwrite(undistorted_file, undistorted)
    cv2.imwrite(undistorted_cropped_file, undistorted_cropped)

    print(f"💾 Исправлено (full): {undistorted_file}")
    print(f"💾 Исправлено (cropped): {undistorted_cropped_file}")

    # 4. Создаем сравнительное изображение
    print()
    comparison = create_comparison_image(frame, undistorted, undistorted_cropped)
    comparison_file = f"{args.output}_comparison.jpg"
    cv2.imwrite(comparison_file, comparison)
    print(f"💾 Сравнение: {comparison_file}")

    print("\n" + "="*60)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*60)
    print("\n📂 Созданные файлы:")
    print(f"   {original_file}")
    print(f"   {undistorted_file}")
    print(f"   {undistorted_cropped_file}")
    print(f"   {comparison_file}")

    print("\n💡 Что смотреть:")
    print("   1. Откройте *_comparison.jpg")
    print("   2. Зеленые линии должны быть ПРЯМЫМИ после коррекции")
    print("   3. Если линии кривые - калибровка плохая (высокая RMS)")
    print("   4. Обратите внимание на края/углы изображения")

    print("\n📊 Оценка качества:")
    rms = camera_calib['ret']
    if rms < 0.5:
        print(f"   ✅ Отлично! (RMS={rms:.3f})")
    elif rms < 1.0:
        print(f"   ✅ Хорошо (RMS={rms:.3f})")
    elif rms < 2.0:
        print(f"   ⚠️  Приемлемо, но не идеально (RMS={rms:.3f})")
    else:
        print(f"   ❌ Плохо! Нужна лучшая калибровка (RMS={rms:.3f})")

if __name__ == "__main__":
    main()
