#!/usr/bin/env python3
"""
Стичинг изображений с двух камер с учетом их геометрии
Камеры расположены под углом 85° и наклонены вниз на 15°
"""

import cv2
import numpy as np
import pickle
import os
import argparse
from datetime import datetime

class StereoStitcher:
    def __init__(self, calibration_file='calibration_result_standard.pkl'):
        """
        Инициализация стичера
        
        Args:
            calibration_file: файл с калибровкой камер
        """
        # Загружаем калибровку
        with open(calibration_file, 'rb') as f:
            self.calib = pickle.load(f)
        
        self.left_calib = self.calib['left_camera']
        self.right_calib = self.calib['right_camera']
        
        # Известная геометрия установки
        self.camera_distance = 50  # мм
        self.camera_angle = 85  # градусов между камерами
        self.tilt_angle = 15  # градусов наклон вниз
        
        print(f"[INFO] Калибровка загружена")
        print(f"  Левая камера:  FOV={self.left_calib['fovx']:.1f}°, RMS={self.left_calib['ret']:.3f}")
        print(f"  Правая камера: FOV={self.right_calib['fovx']:.1f}°, RMS={self.right_calib['ret']:.3f}")
        print(f"  Геометрия: угол={self.camera_angle}°, наклон={self.tilt_angle}°, расстояние={self.camera_distance}мм")
    
    def undistort_image(self, img, camera='left'):
        """
        Исправляет дисторсию изображения
        
        Args:
            img: исходное изображение
            camera: 'left' или 'right'
        
        Returns:
            Изображение с исправленной дисторсией
        """
        if camera == 'left':
            mtx = self.left_calib['mtx']
            dist = self.left_calib['dist']
        else:
            mtx = self.right_calib['mtx']
            dist = self.right_calib['dist']
        
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        return undistorted
    
    def find_features(self, img):
        """
        Находит особые точки на изображении
        
        Args:
            img: изображение
        
        Returns:
            keypoints, descriptors
        """
        # Используем SIFT для лучшего качества
        sift = cv2.SIFT_create(nfeatures=2000)
        
        # Конвертируем в grayscale если нужно
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Находим ключевые точки и дескрипторы
        kp, des = sift.detectAndCompute(gray, None)
        
        return kp, des
    
    def match_features(self, des1, des2):
        """
        Сопоставляет особые точки между изображениями
        
        Args:
            des1: дескрипторы первого изображения
            des2: дескрипторы второго изображения
        
        Returns:
            Список хороших соответствий
        """
        # Используем FLANN matcher для быстрого поиска
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Находим соответствия
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Фильтруем хорошие соответствия (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography_ransac(self, kp1, kp2, matches):
        """
        Находит гомографию между изображениями
        
        Args:
            kp1: ключевые точки первого изображения
            kp2: ключевые точки второго изображения
            matches: соответствия
        
        Returns:
            Матрица гомографии или None
        """
        if len(matches) < 4:
            print("[WARNING] Недостаточно соответствий для гомографии")
            return None
        
        # Извлекаем координаты точек
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Находим гомографию с RANSAC
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Подсчитываем inliers
        inliers = np.sum(mask)
        print(f"  Найдено {inliers}/{len(matches)} inliers")
        
        return homography
    
    def create_panorama_cylindrical(self, img_left, img_right):
        """
        Создает панораму с цилиндрической проекцией
        Подходит для камер с большим углом между ними
        
        Args:
            img_left: левое изображение
            img_right: правое изображение
        
        Returns:
            Панорама
        """
        h, w = img_left.shape[:2]
        
        # Фокусное расстояние для цилиндрической проекции
        f_left = self.left_calib['mtx'][0, 0]
        f_right = self.right_calib['mtx'][0, 0]
        
        # Конвертируем в цилиндрическую проекцию
        def cylindrical_warp(img, f):
            h, w = img.shape[:2]
            K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
            
            # Создаем цилиндрическую проекцию
            cylinder = np.zeros_like(img)
            for y in range(h):
                for x in range(w):
                    theta = (x - w/2) / f
                    h_cyl = (y - h/2) / f
                    
                    X = np.sin(theta)
                    Y = h_cyl
                    Z = np.cos(theta)
                    
                    x_im = f * X / Z + w/2
                    y_im = f * Y / Z + h/2
                    
                    if 0 <= x_im < w and 0 <= y_im < h:
                        cylinder[y, x] = img[int(y_im), int(x_im)]
            
            return cylinder
        
        print("[INFO] Применяем цилиндрическую проекцию...")
        cyl_left = cylindrical_warp(img_left, f_left)
        cyl_right = cylindrical_warp(img_right, f_right)
        
        # Находим особые точки
        kp1, des1 = self.find_features(cyl_left)
        kp2, des2 = self.find_features(cyl_right)
        
        print(f"  Найдено особых точек: левая={len(kp1)}, правая={len(kp2)}")
        
        # Сопоставляем
        matches = self.match_features(des1, des2)
        print(f"  Найдено соответствий: {len(matches)}")
        
        if len(matches) < 10:
            print("[ERROR] Недостаточно соответствий для стичинга")
            return None
        
        # Находим гомографию
        H = self.find_homography_ransac(kp1, kp2, matches)
        
        if H is None:
            return None
        
        # Создаем панораму
        width = w + int(w * 0.5)  # Ширина с запасом
        result = cv2.warpPerspective(cyl_right, H, (width, h))
        result[0:h, 0:w] = cyl_left
        
        return result
    
    def create_panorama_simple(self, img_left, img_right):
        """
        Создает простую панораму через гомографию
        
        Args:
            img_left: левое изображение
            img_right: правое изображение
        
        Returns:
            Панорама
        """
        # Исправляем дисторсию
        print("[INFO] Исправляем дисторсию...")
        img_left_undist = self.undistort_image(img_left, 'left')
        img_right_undist = self.undistort_image(img_right, 'right')
        
        # Уменьшаем для ускорения (опционально)
        scale = 0.5
        if scale < 1.0:
            h, w = img_left_undist.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            img_left_small = cv2.resize(img_left_undist, new_size)
            img_right_small = cv2.resize(img_right_undist, new_size)
        else:
            img_left_small = img_left_undist
            img_right_small = img_right_undist
        
        # Находим особые точки
        print("[INFO] Поиск особых точек...")
        kp1, des1 = self.find_features(img_left_small)
        kp2, des2 = self.find_features(img_right_small)
        
        print(f"  Найдено: левая={len(kp1)}, правая={len(kp2)}")
        
        # Сопоставляем
        print("[INFO] Сопоставление точек...")
        matches = self.match_features(des1, des2)
        print(f"  Найдено соответствий: {len(matches)}")
        
        if len(matches) < 10:
            print("[ERROR] Недостаточно соответствий для стичинга")
            print("  Попробуйте изображения с большим перекрытием")
            return None
        
        # Визуализация соответствий
        match_img = cv2.drawMatches(img_left_small, kp1, img_right_small, kp2, 
                                    matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('matches.jpg', match_img)
        print("  Соответствия сохранены в matches.jpg")
        
        # Находим гомографию
        print("[INFO] Вычисление гомографии...")
        
        # Извлекаем точки
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Масштабируем обратно если уменьшали
        if scale < 1.0:
            src_pts *= (1.0 / scale)
            dst_pts *= (1.0 / scale)
        
        # Находим гомографию
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        inliers = np.sum(mask)
        print(f"  Inliers: {inliers}/{len(matches)}")
        
        # Создаем панораму
        print("[INFO] Создание панорамы...")
        h, w = img_left_undist.shape[:2]
        
        # Определяем размер выходного изображения
        # Учитываем угол между камерами (85°)
        angle_factor = np.cos(np.radians(self.camera_angle))
        width = int(w * (1 + angle_factor))
        
        # Применяем трансформацию
        result = cv2.warpPerspective(img_right_undist, H, (width, h))
        
        # Смешиваем изображения
        # Находим область перекрытия для блендинга
        overlap_start = 0
        overlap_end = w
        
        for x in range(w):
            if np.any(result[:, x] > 0):
                overlap_start = x
                break
        
        if overlap_start > 0:
            # Простое смешивание в зоне перекрытия
            overlap_width = overlap_end - overlap_start
            for x in range(overlap_start, overlap_end):
                alpha = (x - overlap_start) / overlap_width
                result[:, x] = (1 - alpha) * img_left_undist[:, x] + alpha * result[:, x]
        else:
            # Если нет перекрытия, просто копируем левое изображение
            result[0:h, 0:w] = img_left_undist
        
        return result
    
    def stitch_images(self, img_left_path, img_right_path, method='simple'):
        """
        Главная функция стичинга
        
        Args:
            img_left_path: путь к левому изображению
            img_right_path: путь к правому изображению
            method: 'simple' или 'cylindrical'
        
        Returns:
            Панорама
        """
        print(f"\n{'='*60}")
        print(f"СТИЧИНГ ИЗОБРАЖЕНИЙ")
        print(f"{'='*60}")
        
        # Загружаем изображения
        print("[INFO] Загрузка изображений...")
        img_left = cv2.imread(img_left_path)
        img_right = cv2.imread(img_right_path)
        
        if img_left is None or img_right is None:
            print("[ERROR] Не удалось загрузить изображения")
            return None
        
        print(f"  Размер: {img_left.shape[1]}x{img_left.shape[0]}")
        
        # Выполняем стичинг
        if method == 'cylindrical':
            panorama = self.create_panorama_cylindrical(img_left, img_right)
        else:
            panorama = self.create_panorama_simple(img_left, img_right)
        
        if panorama is not None:
            # Обрезаем черные края
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                panorama = panorama[y:y+h, x:x+w]
            
            print(f"[SUCCESS] Панорама создана: {panorama.shape[1]}x{panorama.shape[0]}")
        
        return panorama


def main():
    parser = argparse.ArgumentParser(description='Стичинг изображений с двух камер')
    parser.add_argument('left_image', help='Путь к левому изображению')
    parser.add_argument('right_image', help='Путь к правому изображению')
    parser.add_argument('--calibration', default='calibration_result_standard.pkl',
                       help='Файл калибровки')
    parser.add_argument('--output', '-o', default='panorama.jpg',
                       help='Выходной файл')
    parser.add_argument('--method', choices=['simple', 'cylindrical'], default='simple',
                       help='Метод стичинга')
    parser.add_argument('--live', action='store_true',
                       help='Режим реального времени')
    
    args = parser.parse_args()
    
    # Создаем стичер
    stitcher = StereoStitcher(args.calibration)
    
    if args.live:
        # Режим реального времени
        print("[INFO] Режим реального времени")
        print("  Нажмите 's' для захвата и стичинга")
        print("  Нажмите 'q' для выхода")
        
        # Здесь можно добавить захват с камер в реальном времени
        # Пока просто обрабатываем указанные файлы
    
    # Выполняем стичинг
    panorama = stitcher.stitch_images(args.left_image, args.right_image, args.method)
    
    if panorama is not None:
        # Сохраняем результат
        cv2.imwrite(args.output, panorama)
        print(f"[INFO] Результат сохранен: {args.output}")
        
        # Создаем превью для сравнения
        h, w = panorama.shape[:2]
        scale = 1920 / w if w > 1920 else 1.0
        preview_size = (int(w * scale), int(h * scale))
        
        img_left = cv2.imread(args.left_image)
        img_right = cv2.imread(args.right_image)
        
        # Создаем композицию
        preview_left = cv2.resize(img_left, (preview_size[0]//2, preview_size[1]//2))
        preview_right = cv2.resize(img_right, (preview_size[0]//2, preview_size[1]//2))
        preview_pano = cv2.resize(panorama, preview_size)
        
        top_row = np.hstack([preview_left, preview_right])
        comparison = np.vstack([top_row, preview_pano])
        
        cv2.imwrite('comparison.jpg', comparison)
        print("[INFO] Сравнение сохранено: comparison.jpg")
    else:
        print("[ERROR] Не удалось создать панораму")
        print("  Проверьте, что изображения имеют достаточное перекрытие")


if __name__ == "__main__":
    main()