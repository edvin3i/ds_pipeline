#!/usr/bin/env python3
"""
Стерео-калибровка БЕЗ фиксации внутренних параметров
Для широкоугольной конфигурации (85° между камерами)
"""

import cv2
import numpy as np
import glob
import os
import json
import sys

# Добавляем путь к существующему скрипту
sys.path.insert(0, 'calibration')
from stereo_calibration import find_chessboard_in_pair, load_camera_calibration

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 25.0  # мм

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def stereo_calibrate_unfixed(pairs_dir_cam0, pairs_dir_cam1, cam0_matrix_init, cam0_dist_init, cam1_matrix_init, cam1_dist_init):
    """Стерео-калибровка БЕЗ фиксации внутренних параметров"""
    print(f"\n{'='*80}")
    print("СТЕРЕО-КАЛИБРОВКА (БЕЗ ФИКСАЦИИ ВНУТРЕННИХ ПАРАМЕТРОВ)")
    print(f"{'='*80}\n")

    images_cam0 = sorted(glob.glob(os.path.join(pairs_dir_cam0, "*.jpg")))
    images_cam1 = sorted(glob.glob(os.path.join(pairs_dir_cam1, "*.jpg")))

    print(f"Найдено парных изображений: {len(images_cam0)}")

    # Подготовка объектных точек
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    successful_pairs = []
    img_size = None

    print(f"\nОбработка парных изображений...")
    for i, (img_left_path, img_right_path) in enumerate(zip(images_cam0, images_cam1)):
        basename_left = os.path.basename(img_left_path)
        basename_right = os.path.basename(img_right_path)

        num_left = basename_left.split('_')[1].split('.')[0]
        num_right = basename_right.split('_')[1].split('.')[0]

        if num_left != num_right:
            continue

        corners_left, corners_right, size = find_chessboard_in_pair(img_left_path, img_right_path)

        if corners_left is not None:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            successful_pairs.append(basename_left)
            img_size = size
            print(f"\r  ✓ {len(successful_pairs)}/{len(images_cam0)}", end='', flush=True)

    print(f"\n\nУспешно: {len(successful_pairs)} пар\n")

    if len(successful_pairs) < 10:
        print(f"❌ Недостаточно пар")
        return None

    # ВАРИАНТ 1: БЕЗ ФИКСАЦИИ (позволяем OpenCV оптимизировать всё)
    print("Вариант 1: Полная оптимизация (без фиксации)...")
    flags1 = 0
    # Не используем CALIB_FIX_INTRINSIC

    stereo_rms1, K0_1, D0_1, K1_1, D1_1, R1, T1, E1, F1 = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cam0_matrix_init.copy(),
        cam0_dist_init.copy(),
        cam1_matrix_init.copy(),
        cam1_dist_init.copy(),
        img_size,
        criteria=criteria,
        flags=flags1
    )

    # ВАРИАНТ 2: С ФИКСАЦИЕЙ (как раньше)
    print("Вариант 2: С фиксацией внутренних параметров...")
    flags2 = cv2.CALIB_FIX_INTRINSIC

    stereo_rms2, K0_2, D0_2, K1_2, D1_2, R2, T2, E2, F2 = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cam0_matrix_init.copy(),
        cam0_dist_init.copy(),
        cam1_matrix_init.copy(),
        cam1_dist_init.copy(),
        img_size,
        criteria=criteria,
        flags=flags2
    )

    # ВАРИАНТ 3: RATIONAL MODEL (для широкоугольных камер)
    print("Вариант 3: С рациональной моделью дисторсии...")
    flags3 = cv2.CALIB_RATIONAL_MODEL

    stereo_rms3, K0_3, D0_3, K1_3, D1_3, R3, T3, E3, F3 = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cam0_matrix_init.copy(),
        cam0_dist_init.copy(),
        cam1_matrix_init.copy(),
        cam1_dist_init.copy(),
        img_size,
        criteria=criteria,
        flags=flags3
    )

    # Анализ результатов
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ МЕТОДОВ")
    print(f"{'='*80}\n")

    variants = [
        ("Вариант 1 (без фиксации)", stereo_rms1, R1, T1),
        ("Вариант 2 (с фиксацией)", stereo_rms2, R2, T2),
        ("Вариант 3 (rational model)", stereo_rms3, R3, T3)
    ]

    for name, rms, R, T in variants:
        # Вычисление угла между оптическими осями
        axis_cam0 = np.array([0, 0, 1])
        axis_cam1 = R @ axis_cam0
        cos_angle = np.dot(axis_cam0, axis_cam1)
        angle_optical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        baseline = np.linalg.norm(T)

        # Углы Эйлера
        rvec, _ = cv2.Rodrigues(R)
        angles_deg = np.degrees(rvec.flatten())

        print(f"{name}:")
        print(f"  RMS: {rms:.3f} пикселей")
        print(f"  Угол между оптическими осями: {angle_optical:.2f}°")
        print(f"  Базовая линия: {baseline:.1f} мм ({baseline/10:.1f} см)")
        print(f"  Углы Эйлера: pitch={angles_deg[0]:.1f}°, yaw={angles_deg[1]:.1f}°, roll={angles_deg[2]:.1f}°")
        print()

    # Выбираем лучший вариант (ближайший к 85°)
    angles_list = []
    for name, rms, R, T in variants:
        axis_cam0 = np.array([0, 0, 1])
        axis_cam1 = R @ axis_cam0
        cos_angle = np.dot(axis_cam0, axis_cam1)
        angle_optical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles_list.append((name, angle_optical, rms, R, T))

    # Сортируем по близости к 85°
    angles_list_sorted = sorted(angles_list, key=lambda x: abs(x[1] - 85))

    best_name, best_angle, best_rms, best_R, best_T = angles_list_sorted[0]

    print(f"{'='*80}")
    print("ЛУЧШИЙ РЕЗУЛЬТАТ")
    print(f"{'='*80}\n")
    print(f"Метод: {best_name}")
    print(f"Угол между камерами: {best_angle:.2f}° (ожидаемый: 85°)")
    print(f"Отклонение: {abs(best_angle - 85):.2f}°")
    print(f"RMS: {best_rms:.3f} пикселей")

    if abs(best_angle - 85) < 10:
        print(f"\n✅ ОТЛИЧНО! Угол близок к ожидаемым 85°")
    elif abs(best_angle - 85) < 20:
        print(f"\n✓ ПРИЕМЛЕМО. Угол близок к ожидаемым 85°")
    else:
        print(f"\n⚠ ВНИМАНИЕ! Угол сильно отличается от ожидаемых 85°")

    baseline = np.linalg.norm(best_T)
    print(f"\nБазовая линия: {baseline/10:.1f} см")

    # Сохранение
    result = {
        'method': best_name,
        'num_pairs': len(successful_pairs),
        'stereo_rms': float(best_rms),
        'angle_between_cameras': float(best_angle),
        'baseline_mm': float(baseline),
        'baseline_cm': float(baseline/10),
        'R': best_R.tolist(),
        'T': best_T.tolist(),
        'all_variants': [
            {
                'name': name,
                'rms': float(rms),
                'angle': float(angle),
                'R': R.tolist(),
                'T': T.tolist()
            }
            for name, angle, rms, R, T in angles_list
        ]
    }

    return result

def main():
    print("\nЗагрузка индивидуальных калибровок...")

    calib_file = 'calibration/calibration_results_cleaned.json'
    cam0_matrix, cam0_dist = load_camera_calibration(calib_file, 'camera_0')
    cam1_matrix, cam1_dist = load_camera_calibration(calib_file, 'camera_1')

    print("✓ Калибровки загружены")

    pairs_dir_cam0 = "calibration/calibration_result/pairs/cam0"
    pairs_dir_cam1 = "calibration/calibration_result/pairs/cam1"

    result = stereo_calibrate_unfixed(
        pairs_dir_cam0, pairs_dir_cam1,
        cam0_matrix, cam0_dist,
        cam1_matrix, cam1_dist
    )

    if result:
        output_file = 'calibration/stereo_calibration_unfixed_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Результаты сохранены: {output_file}")
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
