#!/usr/bin/env python3
"""
Полная калибровка стерео системы
Оптимизировано для 27 качественных фото с каждой камеры
"""

import cv2
import numpy as np
import glob
import os
import pickle
from datetime import datetime

class StereoCalibrator:
    def __init__(self, board_size=(10, 7), square_size=23.5):
        """
        Args:
            board_size: количество внутренних углов (10x7 для доски 11x8)
            square_size: размер клетки в мм
        """
        self.board_size = board_size
        self.square_size = square_size
        
        # Подготовка эталонных точек
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Критерии для уточнения углов
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"📐 Инициализация калибратора")
        print(f"   Доска: {board_size[0]}x{board_size[1]} углов")
        print(f"   Размер клетки: {square_size} мм")
    
    def find_corners(self, img_path, visualize=False):
        """Находит углы шахматной доски с улучшенной обработкой"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Не удалось загрузить: {img_path}")
            return None, None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Улучшение контраста для лучшего поиска углов
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Поиск углов с разными флагами
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        
        if ret:
            # Уточнение позиции углов
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
            if visualize:
                img_vis = cv2.drawChessboardCorners(img.copy(), self.board_size, corners_refined, ret)
                cv2.imshow('Corners', cv2.resize(img_vis, (960, 540)))
                cv2.waitKey(100)
            
            return corners_refined, img.shape[:2][::-1], img
        
        return None, None, None
    
    def calibrate_camera(self, image_dir, camera_name="camera"):
        """Калибрует одну камеру с расширенной диагностикой"""
        print(f"\n🎯 Калибровка {camera_name}")
        print("-" * 50)
        
        # Получаем список изображений
        images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                       glob.glob(os.path.join(image_dir, "*.png")))
        
        print(f"📂 Найдено изображений: {len(images)}")
        
        # Массивы для точек
        objpoints = []
        imgpoints = []
        img_shape = None
        
        # Обработка изображений с прогрессом
        successful = 0
        failed_images = []
        
        for i, img_path in enumerate(images):
            print(f"\r⏳ Обработка: {i+1}/{len(images)}", end='', flush=True)
            
            corners, shape, img = self.find_corners(img_path)
            
            if corners is not None:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                img_shape = shape
                successful += 1
                
                # Анализируем покрытие кадра
                if successful == 1:  # Первое успешное изображение
                    self._analyze_coverage(corners, shape)
            else:
                failed_images.append(os.path.basename(img_path))
        
        print(f"\n✅ Успешно обработано: {successful}/{len(images)}")
        
        if failed_images and len(failed_images) <= 5:
            print(f"⚠️  Не найдены углы в: {', '.join(failed_images)}")
        
        if successful < 10:
            print("❌ Недостаточно изображений для калибровки!")
            return None
        
        # Калибровка
        print(f"\n🔄 Выполнение калибровки...")
        
        # Начальное приближение для матрицы камеры
        focal_length = img_shape[0]  # Приблизительная оценка
        K_init = np.array([[focal_length, 0, img_shape[0]/2],
                          [0, focal_length, img_shape[1]/2],
                          [0, 0, 1]], dtype=float)
        
        # Калибровка с разными наборами флагов для сравнения
        calibrations = []
        
        # Вариант 1: Полная модель
        flags1 = 0
        ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, K_init.copy(), None, flags=flags1
        )
        calibrations.append(("Полная модель", ret1, K1, dist1, flags1))
        
        # Вариант 2: Без тангенциальной дисторсии
        flags2 = cv2.CALIB_ZERO_TANGENT_DIST
        ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, K_init.copy(), None, flags=flags2
        )
        calibrations.append(("Без тангенциальной", ret2, K2, dist2, flags2))
        
        # Вариант 3: Фиксированный центр
        flags3 = cv2.CALIB_FIX_PRINCIPAL_POINT
        ret3, K3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, K_init.copy(), None, flags=flags3
        )
        calibrations.append(("Фиксированный центр", ret3, K3, dist3, flags3))
        
        # Выбираем лучший результат
        best_idx = min(range(len(calibrations)), key=lambda i: calibrations[i][1])
        best_name, best_ret, best_K, best_dist, best_flags = calibrations[best_idx]
        
        print(f"\n📊 Результаты калибровки:")
        for name, ret, K, dist, _ in calibrations:
            print(f"   {name}: RMS = {ret:.4f}")
        
        print(f"\n✅ Выбран: {best_name} (RMS = {best_ret:.4f})")
        
        # Детальный анализ результатов
        print(f"\n📋 Параметры камеры:")
        print(f"   Фокусное расстояние: fx={best_K[0,0]:.1f}, fy={best_K[1,1]:.1f}")
        print(f"   Главная точка: cx={best_K[0,2]:.1f}, cy={best_K[1,2]:.1f}")
        print(f"   Коэффициенты дисторсии: {best_dist.ravel()}")
        
        # Оценка качества
        mean_error = self._calculate_reprojection_error(
            objpoints, imgpoints, best_K, best_dist, rvecs1, tvecs1
        )
        print(f"   Средняя ошибка репроекции: {mean_error:.3f} пикселей")
        
        # Вычисляем поле зрения
        fovx, fovy, _, _, _ = cv2.calibrationMatrixValues(
            best_K, img_shape, self.square_size, self.square_size
        )
        print(f"   Поле зрения: {fovx:.1f}° x {fovy:.1f}°")
        
        return {
            'ret': best_ret,
            'mtx': best_K,
            'dist': best_dist,
            'rvecs': rvecs1,
            'tvecs': tvecs1,
            'img_shape': img_shape,
            'fovx': fovx,
            'fovy': fovy,
            'model': 'standard',
            'flags': best_flags,
            'mean_error': mean_error,
            'successful_images': successful,
            'objpoints': objpoints,
            'imgpoints': imgpoints
        }
    
    def _analyze_coverage(self, corners, img_shape):
        """Анализирует покрытие кадра доской"""
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]
        
        board_width = x_coords.max() - x_coords.min()
        board_height = y_coords.max() - y_coords.min()
        board_area = board_width * board_height
        
        img_area = img_shape[0] * img_shape[1]
        coverage = (board_area / img_area) * 100
        
        print(f"\n📏 Анализ покрытия:")
        print(f"   Размер доски: {board_width:.0f}x{board_height:.0f} пикселей")
        print(f"   Покрытие кадра: {coverage:.1f}%")
        
        if coverage < 10:
            print("   ⚠️  Слишком маленькая доска!")
        elif coverage > 50:
            print("   ⚠️  Слишком большая доска!")
        else:
            print("   ✅ Хороший размер доски")
    
    def _calculate_reprojection_error(self, objpoints, imgpoints, K, dist, rvecs, tvecs):
        """Вычисляет среднюю ошибку репроекции"""
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        return total_error / len(objpoints)
    
    def test_undistortion(self, img_path, calib_result, output_path):
        """Тестирует undistortion на изображении"""
        img = cv2.imread(img_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        # Применяем undistortion
        undist = cv2.undistort(img, calib_result['mtx'], calib_result['dist'])
        
        # Добавляем сетку для проверки
        step = 100
        for x in range(0, w, step):
            cv2.line(undist, (x, 0), (x, h), (0, 255, 0), 1)
        for y in range(0, h, step):
            cv2.line(undist, (0, y), (w, y), (0, 255, 0), 1)
        
        # Сохраняем сравнение
        comparison = np.hstack([img, undist])
        comparison = cv2.resize(comparison, (1920, 540))
        cv2.imwrite(output_path, comparison)
        print(f"   💾 Тест сохранен: {output_path}")
    
    def save_calibration(self, calib_data, filename):
        """Сохраняет результаты калибровки"""
        # Убираем тяжелые данные перед сохранением
        data_to_save = {}
        for key, value in calib_data.items():
            if isinstance(value, dict):
                data_to_save[key] = {k: v for k, v in value.items() 
                                    if k not in ['objpoints', 'imgpoints']}
            else:
                data_to_save[key] = value
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"\n💾 Калибровка сохранена: {filename}")

def main():
    print("="*60)
    print("🎯 КАЛИБРОВКА СТЕРЕО СИСТЕМЫ")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Создаем калибратор
    calibrator = StereoCalibrator(board_size=(10, 7), square_size=23.5)
    
    # Пути к изображениям
    left_dir = "calibration_data/left"
    right_dir = "calibration_data/right"
    
    # Проверяем наличие директорий
    if not os.path.exists(left_dir):
        print(f"❌ Директория не найдена: {left_dir}")
        return
    if not os.path.exists(right_dir):
        print(f"❌ Директория не найдена: {right_dir}")
        return
    
    # Калибруем левую камеру
    calib_left = calibrator.calibrate_camera(left_dir, "Левая камера")
    if calib_left is None:
        return
    
    # Калибруем правую камеру
    calib_right = calibrator.calibrate_camera(right_dir, "Правая камера")
    if calib_right is None:
        return
    
    # Тестируем результаты
    print("\n🧪 Тестирование результатов...")
    
    # Берем первое изображение из каждой папки для теста
    left_images = glob.glob(os.path.join(left_dir, "*.jpg"))
    right_images = glob.glob(os.path.join(right_dir, "*.jpg"))
    
    if left_images:
        calibrator.test_undistortion(left_images[0], calib_left, "test_undist_left.jpg")
    if right_images:
        calibrator.test_undistortion(right_images[0], calib_right, "test_undist_right.jpg")
    
    # Формируем финальную структуру
    calibration_data = {
        'left_camera': calib_left,
        'right_camera': calib_right,
        'board_size': calibrator.board_size,
        'square_size': calibrator.square_size,
        'calibration_date': datetime.now().isoformat()
    }
    
    # Сохраняем результат
    output_file = "calibration_stereo_27.pkl"
    calibrator.save_calibration(calibration_data, output_file)
    
    # Итоговая статистика
    print("\n" + "="*60)
    print("✅ КАЛИБРОВКА ЗАВЕРШЕНА!")
    print("="*60)
    print("\n📊 Итоговые результаты:")
    print(f"├─ Левая камера:")
    print(f"│  ├─ RMS: {calib_left['ret']:.4f}")
    print(f"│  ├─ FOV: {calib_left['fovx']:.1f}° x {calib_left['fovy']:.1f}°")
    print(f"│  └─ Средняя ошибка: {calib_left['mean_error']:.3f} px")
    print(f"└─ Правая камера:")
    print(f"   ├─ RMS: {calib_right['ret']:.4f}")
    print(f"   ├─ FOV: {calib_right['fovx']:.1f}° x {calib_right['fovy']:.1f}°")
    print(f"   └─ Средняя ошибка: {calib_right['mean_error']:.3f} px")
    
    # Проверка согласованности
    fov_diff = abs(calib_left['fovx'] - calib_right['fovx'])
    if fov_diff < 2.0:
        print(f"\n✅ Камеры хорошо согласованы (разница FOV: {fov_diff:.1f}°)")
    else:
        print(f"\n⚠️  Заметная разница между камерами (FOV: {fov_diff:.1f}°)")
    
    print(f"\n💾 Файл сохранен: {output_file}")
    print("\n📝 Следующие шаги:")
    print("1. Проверьте test_undist_left.jpg и test_undist_right.jpg")
    print("2. Если линии прямые - запустите generate_warp_universal.py")
    print("3. Проверьте сшивку: python3 extract_frame_50.py left.mp4 right.mp4")

if __name__ == "__main__":
    main()