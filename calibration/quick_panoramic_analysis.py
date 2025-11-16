#!/usr/bin/env python3
"""
Быстрый анализ смещения между камерами
используя синхронные пары где видна доска
"""

import cv2
import numpy as np
import glob
import os

def quick_analysis():
    """Быстрый анализ вертикального смещения"""

    chessboard_size = (8, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Загружаем пары
    left_images = sorted(glob.glob("calibration_data/pairs/cam0/*.jpg"))
    right_images = sorted(glob.glob("calibration_data/pairs/cam1/*.jpg"))

    print(f"Найдено изображений:")
    print(f"  Левая камера: {len(left_images)}")
    print(f"  Правая камера: {len(right_images)}")

    if not left_images or not right_images:
        print("Нет изображений!")
        return

    y_offsets = []
    successful_pairs = 0

    # Обрабатываем только первые 5 пар для скорости
    for i in range(min(5, len(left_images))):
        left_path = left_images[i]
        right_path = right_images[i]

        print(f"\nПара {i+1}: {os.path.basename(left_path)}")

        # Загружаем изображения
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            print("  Ошибка загрузки")
            continue

        # Конвертируем в grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Ищем доску
        ret_l, corners_l = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_l and ret_r:
            # Уточняем углы
            corners_l = cv2.cornerSubPix(gray_left, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_right, corners_r, (11, 11), (-1, -1), criteria)

            # Берём первую точку (левый верхний угол доски)
            y_left_top = corners_l[0][0][1]
            y_right_top = corners_r[0][0][1]

            # Берём центральную точку
            center = len(corners_l) // 2
            y_left_center = corners_l[center][0][1]
            y_right_center = corners_r[center][0][1]

            # Берём последнюю точку (правый нижний угол)
            y_left_bottom = corners_l[-1][0][1]
            y_right_bottom = corners_r[-1][0][1]

            offset_top = y_right_top - y_left_top
            offset_center = y_right_center - y_left_center
            offset_bottom = y_right_bottom - y_left_bottom

            print(f"  ✓ Доска найдена")
            print(f"    Смещение верха: {offset_top:.1f} px")
            print(f"    Смещение центра: {offset_center:.1f} px")
            print(f"    Смещение низа: {offset_bottom:.1f} px")

            y_offsets.append(offset_center)
            successful_pairs += 1

            # Проверяем наклон
            tilt = offset_bottom - offset_top
            if abs(tilt) > 5:
                print(f"    ⚠️ Обнаружен наклон: {tilt:.1f} px")
        else:
            print(f"  ✗ Доска не найдена")
            if not ret_l:
                print(f"    - не найдена на левом кадре")
            if not ret_r:
                print(f"    - не найдена на правом кадре")

    if y_offsets:
        mean_offset = np.mean(y_offsets)
        std_offset = np.std(y_offsets)

        print(f"\n{'='*50}")
        print(f"РЕЗУЛЬТАТЫ АНАЛИЗА")
        print(f"{'='*50}")
        print(f"Успешных пар: {successful_pairs} из {min(5, len(left_images))}")
        print(f"Среднее вертикальное смещение: {mean_offset:.1f} ± {std_offset:.1f} пикселей")
        print(f"\nРЕКОМЕНДАЦИЯ для pano_cuda_debug_stages_new.py:")
        print(f"  K_R[1, 2] += {mean_offset:.0f}  # Добавить к cy правой камеры")

        # Сравнение с текущими значениями
        print(f"\nСРАВНЕНИЕ:")
        print(f"  Из стереокалибровки: cy_R - cy_L = 24.6 px")
        print(f"  Из панорамного анализа: {mean_offset:.1f} px")
        print(f"  В оригинальном коде использовалось: +45 px")

        if abs(mean_offset - 24.6) > 10:
            print(f"\n⚠️ Большое расхождение с калибровкой!")
            print(f"Возможные причины:")
            print(f"  - Камеры сместились после калибровки")
            print(f"  - Разные наборы изображений для калибровки")
            print(f"  - Погрешность из-за угла 85°")
    else:
        print("\nНе удалось найти доску ни на одной паре!")

if __name__ == "__main__":
    quick_analysis()