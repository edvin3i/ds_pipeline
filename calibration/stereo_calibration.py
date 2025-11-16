#!/usr/bin/env python3
"""
Стерео-калибровка для 3D панорамы
Использует индивидуальные калибровки камер и парные изображения
"""

import cv2
import numpy as np
import glob
import os
import json

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 25.0  # мм

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def load_camera_calibration(json_file, camera_key):
    """Загрузка индивидуальной калибровки камеры"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    cam_data = data[camera_key]
    camera_matrix = np.array(cam_data['camera_matrix'])
    dist_coeffs = np.array(cam_data['dist_coeffs'])

    return camera_matrix, dist_coeffs

def find_chessboard_in_pair(img_left_path, img_right_path):
    """Поиск углов шахматной доски в паре изображений"""
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    if img_left is None or img_right is None:
        return None, None, None

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Поиск углов в левом изображении
    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Поиск углов в правом изображении
    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not (ret_left and ret_right):
        return None, None, None

    # Уточнение углов
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

    img_size = gray_left.shape[::-1]

    return corners_left, corners_right, img_size

def stereo_calibrate(pairs_dir_cam0, pairs_dir_cam1, cam0_matrix, cam0_dist, cam1_matrix, cam1_dist):
    """Стерео-калибровка"""
    print(f"\n{'='*80}")
    print("СТЕРЕО-КАЛИБРОВКА ДЛЯ 3D ПАНОРАМЫ")
    print(f"{'='*80}\n")

    # Получение списка парных изображений
    images_cam0 = sorted(glob.glob(os.path.join(pairs_dir_cam0, "*.jpg")))
    images_cam1 = sorted(glob.glob(os.path.join(pairs_dir_cam1, "*.jpg")))

    print(f"Найдено парных изображений:")
    print(f"  Камера 0: {len(images_cam0)}")
    print(f"  Камера 1: {len(images_cam1)}")

    if len(images_cam0) != len(images_cam1):
        print(f"\n❌ ОШИБКА: Количество изображений не совпадает!")
        return None

    # Подготовка объектных точек
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Массивы для хранения точек
    objpoints = []  # 3D точки
    imgpoints_left = []  # 2D точки левая камера
    imgpoints_right = []  # 2D точки правая камера

    successful_pairs = []
    img_size = None

    print(f"\nОбработка парных изображений...")
    for i, (img_left_path, img_right_path) in enumerate(zip(images_cam0, images_cam1)):
        basename_left = os.path.basename(img_left_path)
        basename_right = os.path.basename(img_right_path)

        # Проверка синхронизации (номера должны совпадать)
        num_left = basename_left.split('_')[1].split('.')[0]
        num_right = basename_right.split('_')[1].split('.')[0]

        if num_left != num_right:
            print(f"\n  ⚠ Пропуск несинхронной пары: {basename_left} != {basename_right}")
            continue

        corners_left, corners_right, size = find_chessboard_in_pair(img_left_path, img_right_path)

        if corners_left is not None:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            successful_pairs.append(basename_left)
            img_size = size
            print(f"\r  ✓ {len(successful_pairs)}/{len(images_cam0)} - {basename_left:<30}", end='', flush=True)
        else:
            print(f"\n  ✗ Доска не найдена в паре: {basename_left}")

    print(f"\n\nУспешно обработано: {len(successful_pairs)}/{len(images_cam0)} пар")

    if len(successful_pairs) < 10:
        print(f"❌ ОШИБКА: Недостаточно успешных пар (минимум 10)")
        return None

    # СТЕРЕО-КАЛИБРОВКА
    print(f"\nВыполняется стерео-калибровка...")

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC  # Используем фиксированные параметры индивидуальных калибровок

    stereo_rms, cam0_matrix_new, cam0_dist_new, cam1_matrix_new, cam1_dist_new, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cam0_matrix,
        cam0_dist,
        cam1_matrix,
        cam1_dist,
        img_size,
        criteria=criteria,
        flags=flags
    )

    if stereo_rms is None or stereo_rms < 0:
        print("❌ ОШИБКА: Стерео-калибровка не удалась")
        return None

    # stereo_rms уже является правильной ошибкой репроекции от OpenCV
    stereo_error = stereo_rms

    # РЕЗУЛЬТАТЫ
    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТЫ СТЕРЕО-КАЛИБРОВКИ")
    print(f"{'='*80}\n")

    print(f"Использовано пар: {len(successful_pairs)}")
    print(f"\n📊 Стерео RMS: {stereo_error:.6f} пикселей")

    if stereo_error < 0.5:
        print("   ✅ ОТЛИЧНО! Очень точная стерео-калибровка")
    elif stereo_error < 1.0:
        print("   ✓ ХОРОШО! Точная стерео-калибровка")
    elif stereo_error < 2.0:
        print("   ⚠ ПРИЕМЛЕМО. Стерео-калибровка удовлетворительная")
    else:
        print("   ❌ ВЫСОКИЙ RMS. Рекомендуется улучшить набор данных")

    print(f"\n📐 ГЕОМЕТРИЯ СТЕРЕО-СИСТЕМЫ:")
    print(f"\nМатрица поворота R (от камеры 0 к камере 1):")
    print(R)

    print(f"\nВектор переноса T (от камеры 0 к камере 1):")
    print(T.T)

    # Базовая линия (расстояние между камерами)
    baseline = np.linalg.norm(T)
    print(f"\nБазовая линия (расстояние между камерами): {baseline:.2f} мм ({baseline/10:.2f} см)")

    # Углы поворота
    rvec, _ = cv2.Rodrigues(R)
    angles_deg = np.degrees(rvec.flatten())
    print(f"\nУглы поворота (градусы):")
    print(f"  Вокруг X: {angles_deg[0]:.2f}°")
    print(f"  Вокруг Y: {angles_deg[1]:.2f}°")
    print(f"  Вокруг Z: {angles_deg[2]:.2f}°")

    print(f"\nЭссенциальная матрица E:")
    print(E)

    print(f"\nФундаментальная матрица F:")
    print(F)

    # Ректификация для 3D
    print(f"\n{'='*80}")
    print("РЕКТИФИКАЦИЯ ДЛЯ 3D ПАНОРАМЫ")
    print(f"{'='*80}\n")

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        cam0_matrix, cam0_dist,
        cam1_matrix, cam1_dist,
        img_size, R, T,
        alpha=0  # 0 = обрезать недействительные пиксели, 1 = сохранить все
    )

    print("Матрицы ректификации успешно вычислены:")
    print(f"  R1 (поворот для камеры 0)")
    print(f"  R2 (поворот для камеры 1)")
    print(f"  P1 (проекция для камеры 0)")
    print(f"  P2 (проекция для камеры 1)")
    print(f"  Q  (матрица диспаритета-к-глубине)")

    print(f"\nROI (Region of Interest) после ректификации:")
    print(f"  Левая камера:  x={roi_left[0]}, y={roi_left[1]}, w={roi_left[2]}, h={roi_left[3]}")
    print(f"  Правая камера: x={roi_right[0]}, y={roi_right[1]}, w={roi_right[2]}, h={roi_right[3]}")

    print(f"\n{'='*80}\n")

    result = {
        'num_pairs': len(successful_pairs),
        'stereo_rms': float(stereo_error),
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'baseline_mm': float(baseline),
        'baseline_cm': float(baseline/10),
        'rotation_angles_deg': angles_deg.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist(),
        'roi_left': roi_left,
        'roi_right': roi_right,
        'successful_pairs': successful_pairs
    }

    return result

def main():
    # Загрузка индивидуальных калибровок
    print("\nЗагрузка индивидуальных калибровок камер...")

    calib_file = 'calibration/calibration_results_cleaned.json'
    if not os.path.exists(calib_file):
        print(f"❌ ОШИБКА: Файл {calib_file} не найден!")
        print("Сначала выполните индивидуальную калибровку камер.")
        return

    cam0_matrix, cam0_dist = load_camera_calibration(calib_file, 'camera_0')
    cam1_matrix, cam1_dist = load_camera_calibration(calib_file, 'camera_1')

    print("✓ Калибровки загружены:")
    print(f"  Камера 0: fx={cam0_matrix[0,0]:.2f}, fy={cam0_matrix[1,1]:.2f}")
    print(f"  Камера 1: fx={cam1_matrix[0,0]:.2f}, fy={cam1_matrix[1,1]:.2f}")

    # Стерео-калибровка
    pairs_dir_cam0 = "calibration/calibration_result/pairs/cam0"
    pairs_dir_cam1 = "calibration/calibration_result/pairs/cam1"

    stereo_result = stereo_calibrate(
        pairs_dir_cam0, pairs_dir_cam1,
        cam0_matrix, cam0_dist,
        cam1_matrix, cam1_dist
    )

    if stereo_result:
        # Сохранение полного результата
        full_result = {
            'camera_0': {
                'camera_matrix': cam0_matrix.tolist(),
                'dist_coeffs': cam0_dist.tolist()
            },
            'camera_1': {
                'camera_matrix': cam1_matrix.tolist(),
                'dist_coeffs': cam1_dist.tolist()
            },
            'stereo': stereo_result
        }

        output_file = 'calibration/stereo_calibration_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("ИТОГИ")
        print(f"{'='*80}\n")
        print(f"✅ Стерео-калибровка завершена успешно!")
        print(f"✅ Использовано {stereo_result['num_pairs']} парных изображений")
        print(f"✅ Стерео RMS: {stereo_result['stereo_rms']:.6f} пикселей")
        print(f"✅ Базовая линия: {stereo_result['baseline_cm']:.2f} см")
        print(f"\n✓ Результаты сохранены: {output_file}")
        print(f"\n{'='*80}")
        print("ГОТОВО ДЛЯ 3D ПАНОРАМЫ!")
        print(f"{'='*80}\n")
        print("Теперь можно использовать эти параметры для:")
        print("  • Ректификации изображений")
        print("  • Вычисления карты глубины (disparity map)")
        print("  • Создания 3D панорамы")
        print("  • Стерео-зрения в реальном времени")
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
