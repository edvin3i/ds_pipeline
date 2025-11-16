#!/usr/bin/env python3
"""
Полная калибровка системы из двух камер
Включает внутреннюю калибровку каждой камеры и стерео калибровку
"""

import cv2
import numpy as np
import glob
import os
import pickle
import json
from pathlib import Path

class DualCameraCalibrator:
    def __init__(self, data_dir="calibration_data", board_size=(10, 7), square_size=23.5):
        """
        Инициализация калибратора
        
        Args:
            data_dir: директория с фото
            board_size: количество внутренних углов (10x7 для доски 11x8)
            square_size: размер клетки в мм
        """
        self.data_dir = data_dir
        self.board_size = board_size
        self.square_size = square_size
        
        # Пути к данным
        self.left_dir = os.path.join(data_dir, "left")
        self.right_dir = os.path.join(data_dir, "right")
        self.pairs_dir = os.path.join(data_dir, "pairs")
        
        # Проверяем наличие директорий
        self.check_directories()
        
        # Подготовка эталонных точек доски
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Критерии для уточнения углов
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"[INFO] Калибратор инициализирован")
        print(f"  - Доска: {board_size[0]}x{board_size[1]} углов")
        print(f"  - Размер клетки: {square_size} мм")
        print(f"  - Директория данных: {data_dir}")
    
    def check_directories(self):
        """Проверка наличия необходимых директорий и файлов"""
        # Проверяем новую структуру pairs/cam0 и pairs/cam1
        if os.path.exists(os.path.join(self.pairs_dir, "cam0")):
            self.pairs_cam0_dir = os.path.join(self.pairs_dir, "cam0")
            self.pairs_cam1_dir = os.path.join(self.pairs_dir, "cam1")
        else:
            self.pairs_cam0_dir = self.pairs_dir
            self.pairs_cam1_dir = self.pairs_dir
        
        print(f"\n[INFO] Проверка данных:")
        
        # Подсчет файлов для внутренней калибровки
        left_images = glob.glob(os.path.join(self.left_dir, "*.jpg")) + \
                     glob.glob(os.path.join(self.left_dir, "*.png"))
        right_images = glob.glob(os.path.join(self.right_dir, "*.jpg")) + \
                      glob.glob(os.path.join(self.right_dir, "*.png"))
        
        print(f"  - Левая камера: {len(left_images)} изображений")
        print(f"  - Правая камера: {len(right_images)} изображений")
        
        # Подсчет пар для стерео калибровки
        pairs_cam0 = glob.glob(os.path.join(self.pairs_cam0_dir, "*.jpg")) + \
                     glob.glob(os.path.join(self.pairs_cam0_dir, "*.png"))
        pairs_cam1 = glob.glob(os.path.join(self.pairs_cam1_dir, "*.jpg")) + \
                     glob.glob(os.path.join(self.pairs_cam1_dir, "*.png"))
        
        print(f"  - Синхронные пары: {min(len(pairs_cam0), len(pairs_cam1))} пар")
        
        if len(left_images) < 10:
            print("  ⚠️  Мало изображений для левой камеры (рекомендуется >20)")
        if len(right_images) < 10:
            print("  ⚠️  Мало изображений для правой камеры (рекомендуется >20)")
        if min(len(pairs_cam0), len(pairs_cam1)) < 10:
            print("  ⚠️  Мало синхронных пар (рекомендуется >15)")
    
    def find_corners(self, img_path, show=False):
        """Находит углы шахматной доски на изображении"""
        try:
            # Добавляем имя файла для отладки
            filename = os.path.basename(img_path)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"\n    ⚠️ Не удалось загрузить: {filename}")
                return False, None, None, None
            
            # Уменьшаем изображение если оно слишком большое
            h, w = img.shape[:2]
            if w > 1920:
                scale = 1920 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_small = cv2.resize(img, (new_w, new_h))
                gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Поиск углов с таймаутом
            ret, corners = cv2.findChessboardCorners(
                gray, self.board_size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Масштабируем углы обратно если уменьшали изображение
                if w > 1920:
                    corners = corners * (w / 1920)
                    # Уточняем на оригинальном изображении
                    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    corners = cv2.cornerSubPix(gray_orig, corners, (11, 11), (-1, -1), self.criteria)
                else:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                
                if show:
                    # Визуализация найденных углов
                    img_vis = cv2.resize(img, (960, 540))
                    corners_vis = corners * (960 / w)
                    cv2.drawChessboardCorners(img_vis, self.board_size, corners_vis, ret)
                    cv2.imshow('Corners', img_vis)
                    cv2.waitKey(100)
                
                return ret, corners, (w, h), img
            else:
                print(f"\n    ⚠️ Углы не найдены: {filename}")
                return False, None, None, None
                
        except Exception as e:
            print(f"\n    ❌ Ошибка обработки {img_path}: {e}")
            return False, None, None, None
    
    def calibrate_single_camera(self, images_dir, camera_name="camera", use_fisheye=False):
        """
        Калибрует одну камеру
        
        Args:
            images_dir: директория с изображениями
            camera_name: имя камеры для отображения
            use_fisheye: использовать fisheye модель
        
        Returns:
            Словарь с параметрами калибровки
        """
        print(f"\n[INFO] Калибровка {camera_name}...")
        
        # Массивы для точек
        objpoints = []
        imgpoints = []
        
        # Получаем список изображений
        images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + 
                       glob.glob(os.path.join(images_dir, "*.png")))
        
        print(f"  Найдено {len(images)} изображений")
        
        img_shape = None
        successful = 0
        failed_images = []
        
        for i, fname in enumerate(images):
            print(f"\r  Обработка изображения {i+1}/{len(images)} ({os.path.basename(fname)[:20]})...", end='', flush=True)
            
            ret, corners, shape, img = self.find_corners(fname)
            
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                img_shape = shape
                successful += 1
            else:
                failed_images.append(os.path.basename(fname))
        
        print(f"\n  ✅ Успешно обработано {successful}/{len(images)} изображений")
        
        if failed_images and len(failed_images) <= 5:
            print(f"  ⚠️ Не удалось обработать: {', '.join(failed_images[:5])}")
        elif failed_images:
            print(f"  ⚠️ Не удалось обработать {len(failed_images)} изображений")
        
        if successful < 5:
            print(f"  ❌ Недостаточно изображений для калибровки!")
            return None
        
        print(f"  📐 Размер изображений: {img_shape}")
        
        if use_fisheye:
            # Fisheye калибровка
            print("  Используем fisheye модель...")
            
            N_OK = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
            
            # Подготовка данных для fisheye
            objpoints_fisheye = [x.reshape(1, -1, 3).astype(np.float64) for x in objpoints]
            imgpoints_fisheye = [x.reshape(1, -1, 2).astype(np.float64) for x in imgpoints]
            
            # Калибровка
            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                              cv2.fisheye.CALIB_CHECK_COND + \
                              cv2.fisheye.CALIB_FIX_SKEW
            
            print("  🔄 Выполняется калибровка (fisheye)...")
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints_fisheye, imgpoints_fisheye, img_shape,
                K, D, rvecs, tvecs, calibration_flags, self.criteria
            )
            
            print(f"  📊 RMS ошибка: {ret:.3f}")
            
            return {
                'model': 'fisheye',
                'ret': ret,
                'K': K,
                'D': D,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'img_shape': img_shape,
                'objpoints': objpoints,
                'imgpoints': imgpoints
            }
        else:
            # Стандартная калибровка
            print("  Используем стандартную модель...")
            print("  🔄 Выполняется калибровка (может занять 1-2 минуты)...")
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_shape, None, None
            )
            
            print(f"  📊 RMS ошибка: {ret:.3f}")
            
            # Вычисляем поле зрения
            fovx, fovy, focalLength, principalPoint, aspectRatio = \
                cv2.calibrationMatrixValues(mtx, img_shape, self.square_size, self.square_size)
            
            print(f"  📐 Поле зрения: {fovx:.1f}° x {fovy:.1f}°")
            print(f"  📍 Фокусное расстояние: {mtx[0,0]:.1f} px")
            
            return {
                'model': 'standard',
                'ret': ret,
                'mtx': mtx,
                'dist': dist,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'img_shape': img_shape,
                'fovx': fovx,
                'fovy': fovy,
                'objpoints': objpoints,
                'imgpoints': imgpoints
            }
    
    def stereo_calibrate(self, calib_left, calib_right, use_fisheye=False):
        """
        Стерео калибровка - находит взаимное расположение камер
        
        Args:
            calib_left: результаты калибровки левой камеры
            calib_right: результаты калибровки правой камеры
            use_fisheye: использовать fisheye стерео калибровку
        """
        print("\n[INFO] Стерео калибровка...")
        
        # Массивы для точек
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        
        # Получаем пары изображений
        pairs_left = sorted(glob.glob(os.path.join(self.pairs_cam0_dir, "*.jpg")) + 
                           glob.glob(os.path.join(self.pairs_cam0_dir, "*.png")))
        pairs_right = sorted(glob.glob(os.path.join(self.pairs_cam1_dir, "*.jpg")) + 
                            glob.glob(os.path.join(self.pairs_cam1_dir, "*.png")))
        
        # Сортируем для соответствия пар
        pairs_left.sort()
        pairs_right.sort()
        
        print(f"  Найдено {len(pairs_left)} / {len(pairs_right)} изображений")
        
        successful = 0
        for i, (img_l, img_r) in enumerate(zip(pairs_left, pairs_right)):
            print(f"\r  Обработка пары {i+1}/{len(pairs_left)}...", end='')
            
            ret_l, corners_l, _, _ = self.find_corners(img_l)
            ret_r, corners_r, _, _ = self.find_corners(img_r)
            
            if ret_l and ret_r:
                objpoints.append(self.objp)
                imgpoints_left.append(corners_l)
                imgpoints_right.append(corners_r)
                successful += 1
        
        print(f"\n  ✅ Успешно обработано {successful} пар")
        
        if successful < 5:
            print("  ❌ Недостаточно пар для стерео калибровки!")
            return None
        
        img_shape = calib_left['img_shape']
        
        if use_fisheye:
            # Fisheye стерео калибровка
            print("  Используем fisheye стерео модель...")
            
            # Подготовка данных
            objpoints_fisheye = [x.reshape(1, -1, 3).astype(np.float64) for x in objpoints]
            imgpoints_left_fisheye = [x.reshape(1, -1, 2).astype(np.float64) for x in imgpoints_left]
            imgpoints_right_fisheye = [x.reshape(1, -1, 2).astype(np.float64) for x in imgpoints_right]
            
            K1 = calib_left['K']
            D1 = calib_left['D']
            K2 = calib_right['K']
            D2 = calib_right['D']
            
            R = np.zeros((3, 3))
            T = np.zeros((3, 1))
            
            # Стерео калибровка
            flags = cv2.fisheye.CALIB_FIX_INTRINSIC
            
            ret, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
                objpoints_fisheye, imgpoints_left_fisheye, imgpoints_right_fisheye,
                K1, D1, K2, D2, img_shape, R, T, flags, self.criteria
            )
            
            print(f"  📊 RMS ошибка стерео: {ret:.3f}")
            
            # Стерео ректификация
            R1 = np.zeros((3, 3))
            R2 = np.zeros((3, 3))
            P1 = np.zeros((3, 4))
            P2 = np.zeros((3, 4))
            Q = np.zeros((4, 4))

            # Правильный вызов fisheye.stereoRectify для OpenCV 4.x
            flags = cv2.CALIB_ZERO_DISPARITY
            cv2.fisheye.stereoRectify(
                K1, D1, K2, D2, img_shape, R, T,
                R1, R2, P1, P2, Q,
                flags, img_shape, 0, 0
            )
            
            # Создаем карты для ректификации
            map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
                K1, D1, R1, P1, img_shape, cv2.CV_16SC2
            )
            map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
                K2, D2, R2, P2, img_shape, cv2.CV_16SC2
            )
            
        else:
            # Стандартная стерео калибровка
            print("  Используем стандартную стерео модель...")
            
            ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                calib_left['mtx'], calib_left['dist'],
                calib_right['mtx'], calib_right['dist'],
                img_shape,
                criteria=self.criteria,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            print(f"  📊 RMS ошибка стерео: {ret:.3f}")
            
            # Стерео ректификация
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                mtx1, dist1, mtx2, dist2, img_shape, R, T,
                alpha=0, newImageSize=img_shape
            )
            
            # Создаем карты для ректификации
            map1_left, map2_left = cv2.initUndistortRectifyMap(
                mtx1, dist1, R1, P1, img_shape, cv2.CV_16SC2
            )
            map1_right, map2_right = cv2.initUndistortRectifyMap(
                mtx2, dist2, R2, P2, img_shape, cv2.CV_16SC2
            )
        
        # Вычисляем параметры стерео системы
        baseline = np.linalg.norm(T)
        
        # Угол между камерами (предполагаем поворот вокруг Y)
        angle_rad = np.arctan2(R[2, 0], R[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        print(f"\n  📏 Базовая линия: {baseline:.1f} мм")
        print(f"  📐 Угол между камерами: {angle_deg:.1f}°")
        
        return {
            'ret': ret,
            'R': R,
            'T': T,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'map1_left': map1_left,
            'map2_left': map2_left,
            'map1_right': map1_right,
            'map2_right': map2_right,
            'baseline': baseline,
            'angle': angle_deg
        }
    
    def test_rectification(self, stereo_calib):
        """Тестирует ректификацию на примере пары изображений"""
        print("\n[INFO] Тестирование ректификации...")
        
        # Берем первую пару
        pairs_left = sorted(glob.glob(os.path.join(self.pairs_cam0_dir, "*.jpg")))
        pairs_right = sorted(glob.glob(os.path.join(self.pairs_cam1_dir, "*.jpg")))
        
        if not pairs_left or not pairs_right:
            print("  ❌ Нет изображений для теста")
            return
        
        img_left = cv2.imread(pairs_left[0])
        img_right = cv2.imread(pairs_right[0])
        
        # Применяем ректификацию
        rect_left = cv2.remap(img_left, stereo_calib['map1_left'], 
                              stereo_calib['map2_left'], cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, stereo_calib['map1_right'], 
                               stereo_calib['map2_right'], cv2.INTER_LINEAR)
        
        # Создаем стерео пару с эпиполярными линиями
        h, w = rect_left.shape[:2]
        stereo_pair = np.hstack([rect_left, rect_right])
        
        # Рисуем горизонтальные линии для проверки выравнивания
        for y in range(0, h, 50):
            cv2.line(stereo_pair, (0, y), (w*2, y), (0, 255, 0), 1)
        
        # Сохраняем результат
        output_file = "stereo_rectified_test.jpg"
        cv2.imwrite(output_file, stereo_pair)
        print(f"  ✅ Тест сохранен в {output_file}")
        print("  📝 Зеленые линии должны быть горизонтальными и выровненными")
    
    def save_calibration(self, calibration_data, filename="calibration_result.pkl"):
        """Сохраняет результаты калибровки"""
        with open(filename, 'wb') as f:
            # Убираем objpoints и imgpoints для экономии места
            data_to_save = {}
            for key, value in calibration_data.items():
                if isinstance(value, dict):
                    data_to_save[key] = {k: v for k, v in value.items() 
                                        if k not in ['objpoints', 'imgpoints']}
                else:
                    data_to_save[key] = value
            
            pickle.dump(data_to_save, f)
        print(f"\n[INFO] Калибровка сохранена в {filename}")
    
    def run_full_calibration(self, use_fisheye=False):
        """Запускает полный процесс калибровки"""
        print("\n" + "="*60)
        print("ПОЛНАЯ КАЛИБРОВКА СТЕРЕО СИСТЕМЫ")
        print("="*60)
        
        # 1. Калибровка левой камеры
        print("\n📷 ЭТАП 1: Калибровка левой камеры")
        print("-"*40)
        calib_left = self.calibrate_single_camera(self.left_dir, "Левая камера", use_fisheye)
        
        if calib_left is None:
            print("❌ Не удалось откалибровать левую камеру")
            return
        
        # 2. Калибровка правой камеры
        print("\n📷 ЭТАП 2: Калибровка правой камеры")
        print("-"*40)
        calib_right = self.calibrate_single_camera(self.right_dir, "Правая камера", use_fisheye)
        
        if calib_right is None:
            print("❌ Не удалось откалибровать правую камеру")
            return
        
        # 3. Стерео калибровка
        print("\n🔄 ЭТАП 3: Стерео калибровка")
        print("-"*40)
        stereo_calib = self.stereo_calibrate(calib_left, calib_right, use_fisheye)
        
        if stereo_calib is None:
            print("❌ Не удалось выполнить стерео калибровку")
            return
        
        # 4. Тестирование
        self.test_rectification(stereo_calib)
        
        # 5. Сохранение результатов
        calibration_data = {
            'left_camera': calib_left,
            'right_camera': calib_right,
            'stereo': stereo_calib,
            'board_size': self.board_size,
            'square_size': self.square_size
        }
        
        model_suffix = "_fisheye" if use_fisheye else "_standard"
        self.save_calibration(calibration_data, f"calibration_result{model_suffix}.pkl")
        
        print("\n" + "="*60)
        print("✅ КАЛИБРОВКА ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print("\nРезультаты:")
        print(f"  • Левая камера RMS: {calib_left['ret']:.3f}")
        print(f"  • Правая камера RMS: {calib_right['ret']:.3f}")
        print(f"  • Стерео RMS: {stereo_calib['ret']:.3f}")
        print(f"  • Базовая линия: {stereo_calib['baseline']:.1f} мм")
        print(f"  • Угол: {stereo_calib['angle']:.1f}°")
        
        return calibration_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Полная калибровка стерео системы')
    parser.add_argument('--data-dir', default='calibration_data', 
                       help='Директория с калибровочными изображениями')
    parser.add_argument('--board-rows', type=int, default=6,
                       help='Количество внутренних углов по вертикали')
    parser.add_argument('--board-cols', type=int, default=9,
                       help='Количество внутренних углов по горизонтали')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Размер клетки в мм')
    parser.add_argument('--fisheye', action='store_true',
                       help='Использовать fisheye модель')
    
    args = parser.parse_args()
    
    # Создаем калибратор
    calibrator = DualCameraCalibrator(
        data_dir=args.data_dir,
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size
    )
    
    # Запускаем калибровку
    calibrator.run_full_calibration(use_fisheye=args.fisheye)

if __name__ == "__main__":
    main()