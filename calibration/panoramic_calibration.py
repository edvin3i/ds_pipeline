#!/usr/bin/env python3
"""
Панорамная калибровка для камер с большим углом между ними.
Использует синхронные пары изображений где доска видна на обоих кадрах.
Цель: найти трансформацию для выравнивания горизонтов.
"""

import cv2
import numpy as np
import glob
import json
import pickle
import os
from typing import Tuple, List, Dict, Any

class PanoramicCalibration:
    def __init__(self, chessboard_size=(8, 6), square_size=25.0):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def find_chessboard_corners(self, img):
        """Находит углы шахматной доски"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        return ret, corners

    def load_synchronized_pairs(self, base_path="calibration_data/pairs"):
        """Загружает синхронные пары изображений"""
        left_images = sorted(glob.glob(f"{base_path}/cam0/*.jpg"))
        right_images = sorted(glob.glob(f"{base_path}/cam1/*.jpg"))

        pairs = []
        for left_path, right_path in zip(left_images, right_images):
            # Проверяем, что номера совпадают
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            if left_name.replace('cam0', '') == right_name.replace('cam1', ''):
                pairs.append((left_path, right_path))

        return pairs

    def analyze_panoramic_alignment(self, pairs):
        """
        Анализирует выравнивание для панорамной установки
        Цель: найти вертикальное смещение между камерами
        """
        valid_pairs = []
        y_offsets = []

        for left_path, right_path in pairs:
            print(f"\nОбрабатываем пару: {os.path.basename(left_path)}")

            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                print(f"  Не удалось загрузить изображения")
                continue

            # Находим углы на обоих изображениях
            ret_l, corners_l = self.find_chessboard_corners(img_left)
            ret_r, corners_r = self.find_chessboard_corners(img_right)

            if ret_l and ret_r:
                print(f"  ✓ Доска найдена на обоих кадрах")

                # Анализируем соответствующие точки
                # Берём центральные точки доски для анализа
                center_idx = len(corners_l) // 2

                # Y-координаты центральной точки
                y_left = corners_l[center_idx][0][1]
                y_right = corners_r[center_idx][0][1]

                y_offset = y_right - y_left
                y_offsets.append(y_offset)

                print(f"  Y смещение центра доски: {y_offset:.1f} пикселей")

                # Сохраняем для дальнейшего анализа
                valid_pairs.append({
                    'left_path': left_path,
                    'right_path': right_path,
                    'corners_left': corners_l,
                    'corners_right': corners_r,
                    'y_offset': y_offset
                })
            else:
                print(f"  ✗ Доска не найдена на одном из кадров")

        if y_offsets:
            mean_offset = np.mean(y_offsets)
            std_offset = np.std(y_offsets)

            print(f"\n=== АНАЛИЗ ВЕРТИКАЛЬНОГО ВЫРАВНИВАНИЯ ===")
            print(f"Найдено пар с доской: {len(y_offsets)}")
            print(f"Среднее смещение по Y: {mean_offset:.1f} ± {std_offset:.1f} пикселей")
            print(f"Минимальное: {min(y_offsets):.1f}")
            print(f"Максимальное: {max(y_offsets):.1f}")

            return valid_pairs, mean_offset
        else:
            print("Не найдено ни одной пары с доской!")
            return [], 0

    def compute_homography_alignment(self, valid_pairs):
        """
        Вычисляет гомографию для выравнивания изображений
        Это более продвинутый метод чем просто смещение cy
        """
        if not valid_pairs:
            return None

        all_pts_left = []
        all_pts_right = []

        for pair in valid_pairs:
            # Используем все точки доски
            pts_l = pair['corners_left'].reshape(-1, 2)
            pts_r = pair['corners_right'].reshape(-1, 2)

            all_pts_left.append(pts_l)
            all_pts_right.append(pts_r)

        # Объединяем все точки
        all_pts_left = np.vstack(all_pts_left)
        all_pts_right = np.vstack(all_pts_right)

        # Вычисляем гомографию для выравнивания
        # H преобразует точки из правой камеры так, чтобы они совпали с левой по высоте
        H, mask = cv2.findHomography(all_pts_right, all_pts_left, cv2.RANSAC, 5.0)

        print(f"\n=== ГОМОГРАФИЯ ДЛЯ ВЫРАВНИВАНИЯ ===")
        print(f"Матрица 3x3:")
        print(H)

        # Анализ гомографии
        # В идеальном случае для простого вертикального смещения:
        # H должна быть близка к единичной матрице с смещением в H[1,2]

        print(f"\nАнализ компонентов:")
        print(f"Масштаб X: {H[0,0]:.4f} (должен быть ≈ 1.0)")
        print(f"Масштаб Y: {H[1,1]:.4f} (должен быть ≈ 1.0)")
        print(f"Смещение X: {H[0,2]:.1f} пикселей")
        print(f"Смещение Y: {H[1,2]:.1f} пикселей")
        print(f"Поворот/Сдвиг: {H[0,1]:.4f}, {H[1,0]:.4f} (должны быть ≈ 0.0)")

        return H

    def create_panoramic_calibration(self):
        """Главная функция для создания панорамной калибровки"""

        print("=== ПАНОРАМНАЯ КАЛИБРОВКА ===")
        print("Используем синхронные пары где доска видна на обоих кадрах")

        # Загружаем существующую калибровку для K и D
        with open('../calibration_result_standard.pkl', 'rb') as f:
            standard_calib = pickle.load(f)

        # Загружаем синхронные пары
        pairs = self.load_synchronized_pairs()
        print(f"\nНайдено синхронных пар: {len(pairs)}")

        if not pairs:
            print("Нет синхронных пар! Проверьте путь к файлам.")
            return None

        # Анализируем выравнивание
        valid_pairs, mean_y_offset = self.analyze_panoramic_alignment(pairs)

        if not valid_pairs:
            print("Не удалось найти доску на синхронных парах!")
            return None

        # Вычисляем гомографию
        H = self.compute_homography_alignment(valid_pairs)

        # Создаём улучшенную калибровку
        panoramic_calib = {
            'left_camera': standard_calib['left_camera'],
            'right_camera': standard_calib['right_camera'],
            'stereo': standard_calib['stereo'],
            'panoramic': {
                'mean_y_offset': mean_y_offset,
                'homography': H,
                'num_pairs_used': len(valid_pairs),
                'recommended_cy_correction': mean_y_offset,  # Прямая коррекция
                'method': 'synchronized_pairs'
            }
        }

        # Сохраняем
        output_path = 'calibration_panoramic.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(panoramic_calib, f)

        print(f"\n=== РЕЗУЛЬТАТ ===")
        print(f"Панорамная калибровка сохранена в: {output_path}")
        print(f"\nРЕКОМЕНДАЦИИ ДЛЯ pano_cuda_debug_stages_new.py:")
        print(f"1. Добавьте к cy правой камеры: {mean_y_offset:.1f} пикселей")
        print(f"2. Или используйте гомографию для более точного выравнивания")

        # Дополнительный анализ для отладки
        print(f"\n=== ДЕТАЛЬНЫЙ АНАЛИЗ ===")

        # Проверяем, одинаково ли смещение по всей доске
        for pair in valid_pairs[:3]:  # Берём первые 3 пары для анализа
            corners_l = pair['corners_left'].reshape(-1, 2)
            corners_r = pair['corners_right'].reshape(-1, 2)

            # Смещение для разных частей доски
            top_offset = corners_r[0][1] - corners_l[0][1]
            middle_offset = corners_r[len(corners_r)//2][1] - corners_l[len(corners_l)//2][1]
            bottom_offset = corners_r[-1][1] - corners_l[-1][1]

            print(f"\nПара {os.path.basename(pair['left_path'])}:")
            print(f"  Верх доски: {top_offset:.1f} пикселей")
            print(f"  Центр доски: {middle_offset:.1f} пикселей")
            print(f"  Низ доски: {bottom_offset:.1f} пикселей")
            print(f"  Разница (низ-верх): {bottom_offset - top_offset:.1f} пикселей")

            if abs(bottom_offset - top_offset) > 5:
                print("  ⚠️ Обнаружен наклон! Простое смещение cy недостаточно.")
                print("  Рекомендуется использовать гомографию или ручную коррекцию.")

        return panoramic_calib

if __name__ == "__main__":
    calibrator = PanoramicCalibration()
    result = calibrator.create_panoramic_calibration()

    if result:
        print("\n✅ Панорамная калибровка успешно создана!")
        print("Используйте файл 'calibration_panoramic.pkl' вместо стандартного.")
    else:
        print("\n❌ Не удалось создать панорамную калибровку")