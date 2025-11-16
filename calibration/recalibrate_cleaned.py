#!/usr/bin/env python3
"""
Калибровка камер с очищенным набором изображений
Выполняет индивидуальную калибровку каждой камеры и выводит RMS
"""

import cv2
import numpy as np
import glob
import os
import json

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 25.0  # мм (размер клетки шахматной доски)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate_camera(images_path, camera_name):
    """Калибровка одной камеры"""
    print(f"\n{'='*80}")
    print(f"КАЛИБРОВКА: {camera_name}")
    print(f"{'='*80}")

    images = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    print(f"Найдено изображений: {len(images)}")

    if len(images) < 10:
        print(f"❌ ОШИБКА: Недостаточно изображений (минимум 10)")
        return None

    # Подготовка объектных точек
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Массивы для хранения точек
    objpoints = []  # 3D точки в реальном мире
    imgpoints = []  # 2D точки на изображении

    successful = 0
    img_size = None

    print("\nОбработка изображений...")
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Поиск углов
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)

            # Уточнение углов
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            successful += 1
            print(f"\r  ✓ {successful}/{len(images)} - {os.path.basename(img_path):<40}", end='', flush=True)
        else:
            print(f"\n  ✗ Доска не найдена: {os.path.basename(img_path)}")

    print(f"\n\nУспешно обработано: {successful}/{len(images)} изображений")

    if successful < 10:
        print(f"❌ ОШИБКА: Недостаточно успешных изображений")
        return None

    # КАЛИБРОВКА
    print("\nВыполняется калибровка...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    if not ret:
        print("❌ ОШИБКА: Калибровка не удалась")
        return None

    # Вычисление RMS для каждого изображения
    print("\nВычисление ошибок репроекции...")
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)

    mean_error = np.mean(errors)

    # Результаты
    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    print(f"{'='*80}")
    print(f"\nКамера: {camera_name}")
    print(f"Использовано изображений: {successful}")
    print(f"\n📊 RMS (Root Mean Square Error): {mean_error:.6f} пикселей")

    if mean_error < 0.3:
        print("   ✅ ОТЛИЧНО! Очень точная калибровка")
    elif mean_error < 0.5:
        print("   ✅ ХОРОШО! Точная калибровка")
    elif mean_error < 1.0:
        print("   ✓ ПРИЕМЛЕМО. Калибровка удовлетворительная")
    else:
        print("   ⚠ ВЫСОКИЙ RMS. Рекомендуется улучшить набор данных")

    print(f"\nМинимальная ошибка: {np.min(errors):.6f} пикселей")
    print(f"Максимальная ошибка: {np.max(errors):.6f} пикселей")
    print(f"Стандартное отклонение: {np.std(errors):.6f} пикселей")

    print(f"\nМатрица камеры (K):")
    print(mtx)
    print(f"\nКоэффициенты дисторсии:")
    print(dist)

    # Параметры камеры
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    print(f"\nФокальная длина:")
    print(f"  fx = {fx:.2f} пикселей")
    print(f"  fy = {fy:.2f} пикселей")
    print(f"\nГлавная точка:")
    print(f"  cx = {cx:.2f} пикселей")
    print(f"  cy = {cy:.2f} пикселей")

    print(f"\n{'='*80}\n")

    result = {
        'camera_name': camera_name,
        'num_images': successful,
        'rms': float(mean_error),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors)),
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist(),
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy)
    }

    return result


def main():
    print(f"\n{'='*80}")
    print("КАЛИБРОВКА КАМЕР С ОЧИЩЕННЫМ НАБОРОМ ДАННЫХ")
    print(f"{'='*80}\n")

    # Калибровка камеры 0
    cam0_result = calibrate_camera(
        "calibration/calibration_result/left",
        "Камера 0 (левая)"
    )

    # Калибровка камеры 1
    cam1_result = calibrate_camera(
        "calibration/calibration_result/right",
        "Камера 1 (правая)"
    )

    # Итоговое сравнение
    if cam0_result and cam1_result:
        print(f"\n{'='*80}")
        print("ИТОГОВОЕ СРАВНЕНИЕ")
        print(f"{'='*80}\n")

        print(f"{'Параметр':<30} {'Камера 0':>20} {'Камера 1':>20}")
        print(f"{'-'*75}")
        print(f"{'Изображений':<30} {cam0_result['num_images']:>20} {cam1_result['num_images']:>20}")
        print(f"{'RMS (пикселей)':<30} {cam0_result['rms']:>20.6f} {cam1_result['rms']:>20.6f}")
        print(f"{'Фокальная длина fx':<30} {cam0_result['fx']:>20.2f} {cam1_result['fx']:>20.2f}")
        print(f"{'Фокальная длина fy':<30} {cam0_result['fy']:>20.2f} {cam1_result['fy']:>20.2f}")

        print(f"\n{'='*80}")
        print("ОЦЕНКА")
        print(f"{'='*80}\n")

        if cam0_result['rms'] < 0.5 and cam1_result['rms'] < 0.5:
            print("✅ ОБЕ КАМЕРЫ: Отличная калибровка! RMS < 0.5 пикселей")
        elif cam0_result['rms'] < 1.0 and cam1_result['rms'] < 1.0:
            print("✓ ОБЕ КАМЕРЫ: Хорошая калибровка. RMS < 1.0 пикселей")
        else:
            if cam0_result['rms'] >= 1.0:
                print(f"⚠ Камера 0: RMS = {cam0_result['rms']:.3f} всё ещё высокий")
            if cam1_result['rms'] >= 1.0:
                print(f"⚠ Камера 1: RMS = {cam1_result['rms']:.3f} всё ещё высокий")

        # Сравнение фокальных длин (должны быть близки для одинаковых камер)
        fx_diff = abs(cam0_result['fx'] - cam1_result['fx'])
        fy_diff = abs(cam0_result['fy'] - cam1_result['fy'])

        print(f"\nРазница фокальных длин:")
        print(f"  fx: {fx_diff:.2f} пикселей ({fx_diff/cam0_result['fx']*100:.2f}%)")
        print(f"  fy: {fy_diff:.2f} пикселей ({fy_diff/cam0_result['fy']*100:.2f}%)")

        if fx_diff < 20 and fy_diff < 20:
            print("  ✅ Камеры хорошо согласованы (одинаковые параметры)")
        else:
            print("  ⚠ Большая разница в параметрах камер")

        # Сохранение результатов
        results = {
            'camera_0': cam0_result,
            'camera_1': cam1_result,
            'comparison': {
                'rms_diff': abs(cam0_result['rms'] - cam1_result['rms']),
                'fx_diff': fx_diff,
                'fy_diff': fy_diff
            }
        }

        output_file = 'calibration/calibration_results_cleaned.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Результаты сохранены: {output_file}")
        print(f"\n{'='*80}\n")

    # Сравнение ДО и ПОСЛЕ
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ ДО И ПОСЛЕ ОЧИСТКИ")
    print(f"{'='*80}\n")
    print("ДО очистки:")
    print("  Камера 0 RMS: 12.000 пикселей ❌")
    print("  Камера 1 RMS:  1.774 пикселей ⚠")
    print()
    print("ПОСЛЕ очистки:")
    if cam0_result and cam1_result:
        status0 = "✅" if cam0_result['rms'] < 0.5 else "✓" if cam0_result['rms'] < 1.0 else "⚠"
        status1 = "✅" if cam1_result['rms'] < 0.5 else "✓" if cam1_result['rms'] < 1.0 else "⚠"
        print(f"  Камера 0 RMS: {cam0_result['rms']:.6f} пикселей {status0}")
        print(f"  Камера 1 RMS: {cam1_result['rms']:.6f} пикселей {status1}")

        improvement_cam0 = 12.0 - cam0_result['rms']
        improvement_cam1 = 1.774 - cam1_result['rms']

        print()
        print(f"Улучшение:")
        print(f"  Камера 0: -{improvement_cam0:.3f} пикселей (улучшение на {improvement_cam0/12.0*100:.1f}%)")
        print(f"  Камера 1: -{improvement_cam1:.3f} пикселей (улучшение на {improvement_cam1/1.774*100:.1f}%)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
