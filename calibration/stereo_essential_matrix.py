#!/usr/bin/env python3
"""
Калибровка широкоугольной стерео-системы (85°) через Essential Matrix
Использует известные углы шахматной доски для вычисления R и T
"""

import cv2
import numpy as np
import glob
import os
import json
import sys

sys.path.insert(0, 'calibration')
from stereo_calibration import find_chessboard_in_pair, load_camera_calibration

CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def estimate_pose_from_stereo_pairs(pairs_dir_cam0, pairs_dir_cam1, K0, D0, K1, D1):
    """
    Оценка взаимного положения камер через соответствующие точки
    """
    print(f"\n{'='*80}")
    print("ОЦЕНКА ГЕОМЕТРИИ ЧЕРЕЗ ESSENTIAL MATRIX")
    print(f"{'='*80}\n")

    images_cam0 = sorted(glob.glob(os.path.join(pairs_dir_cam0, "*.jpg")))
    images_cam1 = sorted(glob.glob(os.path.join(pairs_dir_cam1, "*.jpg")))

    print(f"Найдено парных изображений: {len(images_cam0)}")

    all_points_cam0 = []
    all_points_cam1 = []
    successful_pairs = []

    print(f"\nИзвлечение соответствующих точек...")
    for i, (img0_path, img1_path) in enumerate(zip(images_cam0, images_cam1)):
        basename = os.path.basename(img0_path)

        corners0, corners1, img_size = find_chessboard_in_pair(img0_path, img1_path)

        if corners0 is not None:
            # Undistort точки
            corners0_undist = cv2.undistortPoints(corners0, K0, D0, None, K0)
            corners1_undist = cv2.undistortPoints(corners1, K1, D1, None, K1)

            all_points_cam0.extend(corners0_undist.reshape(-1, 2))
            all_points_cam1.extend(corners1_undist.reshape(-1, 2))
            successful_pairs.append(basename)
            print(f"\r  ✓ {len(successful_pairs)}/{ len(images_cam0)}", end='', flush=True)

    print(f"\n\nУспешно: {len(successful_pairs)} пар")
    print(f"Всего соответствующих точек: {len(all_points_cam0)}")

    if len(all_points_cam0) < 50:
        print("❌ Недостаточно точек")
        return None

    points_cam0 = np.array(all_points_cam0, dtype=np.float32)
    points_cam1 = np.array(all_points_cam1, dtype=np.float32)

    # Вычисление Essential Matrix
    print(f"\nВычисление Essential Matrix...")
    E, mask = cv2.findEssentialMat(
        points_cam0, points_cam1,
        K0,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        print("❌ Не удалось вычислить Essential Matrix")
        return None

    inliers = np.sum(mask)
    print(f"Инлайеры: {inliers}/{len(points_cam0)} ({inliers/len(points_cam0)*100:.1f}%)")

    # Восстановление R и T из Essential Matrix
    print(f"\nВосстановление R и T...")
    points_inliers_cam0 = points_cam0[mask.ravel() == 1]
    points_inliers_cam1 = points_cam1[mask.ravel() == 1]

    _, R, T, mask_pose = cv2.recoverPose(
        E,
        points_inliers_cam0,
        points_inliers_cam1,
        K0
    )

    # Анализ результата
    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*80}\n")

    # Угол между оптическими осями
    axis_cam0 = np.array([0, 0, 1])
    axis_cam1 = R @ axis_cam0
    cos_angle = np.dot(axis_cam0, axis_cam1)
    angle_optical = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    print(f"Угол между оптическими осями камер: {angle_optical:.2f}°")

    if abs(angle_optical - 85) < 10:
        print(f"✅ ОТЛИЧНО! Близко к ожидаемым 85°")
    elif abs(angle_optical - 85) < 20:
        print(f"✓ ПРИЕМЛЕМО. Близко к ожидаемым 85°")
    else:
        print(f"⚠ Отклонение: {abs(angle_optical - 85):.2f}°")

    # ВАЖНО: T из recoverPose имеет нормализованную длину!
    # Нужно вычислить реальный масштаб через известное расстояние
    print(f"\nВектор переноса T (нормализованный):")
    print(T.T)
    print(f"\n⚠ ВНИМАНИЕ: Масштаб T неизвестен (Essential Matrix теряет масштаб)")
    print(f"Для получения реального расстояния нужна дополнительная информация:")
    print(f"  - Известное расстояние между камерами")
    print(f"  - Или размер известного объекта в сцене")

    # Углы Эйлера
    rvec, _ = cv2.Rodrigues(R)
    angles_deg = np.degrees(rvec.flatten())

    print(f"\nУглы Эйлера:")
    print(f"  Pitch (X): {angles_deg[0]:7.2f}°")
    print(f"  Yaw   (Y): {angles_deg[1]:7.2f}°")
    print(f"  Roll  (Z): {angles_deg[2]:7.2f}°")

    print(f"\nМатрица поворота R:")
    print(R)

    # Вычисление фундаментальной матрицы
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

    result = {
        'method': 'Essential Matrix (RANSAC)',
        'num_pairs': len(successful_pairs),
        'num_points': len(all_points_cam0),
        'inliers': int(inliers),
        'inlier_ratio': float(inliers/len(all_points_cam0)),
        'angle_between_cameras': float(angle_optical),
        'R': R.tolist(),
        'T_normalized': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'rotation_angles_deg': angles_deg.tolist(),
        'note': 'T is normalized - real baseline unknown without additional info'
    }

    return result

def main():
    print("\nЗагрузка индивидуальных калибровок...")

    calib_file = 'calibration/calibration_results_cleaned.json'
    K0, D0 = load_camera_calibration(calib_file, 'camera_0')
    K1, D1 = load_camera_calibration(calib_file, 'camera_1')

    print("✓ Калибровки загружены")

    pairs_dir_cam0 = "calibration/calibration_result/pairs/cam0"
    pairs_dir_cam1 = "calibration/calibration_result/pairs/cam1"

    result = estimate_pose_from_stereo_pairs(
        pairs_dir_cam0, pairs_dir_cam1,
        K0, D0, K1, D1
    )

    if result:
        output_file = 'calibration/stereo_essential_matrix_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Результаты сохранены: {output_file}")

        print(f"\n{'='*80}")
        print("ВЫВОДЫ")
        print(f"{'='*80}\n")
        print(f"Угол между камерами: {result['angle_between_cameras']:.2f}°")
        print(f"Инлайеры: {result['inlier_ratio']*100:.1f}%")
        print()
        print("Essential Matrix метод лучше подходит для широкоугольных конфигураций,")
        print("но не может определить реальное расстояние между камерами.")
        print()
        print("Для получения полного решения нужно:")
        print("  1. Измерить физическое расстояние между камерами")
        print("  2. Масштабировать вектор T соответственно")
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
