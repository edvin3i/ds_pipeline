#!/usr/bin/env python3
"""
Тестирование коррекции дисторсии на реальных кадрах
Сравнение старой и новой калибровки
"""

import cv2
import numpy as np
import pickle
import os
from datetime import datetime

def load_calibration(filename):
    """Загружает файл калибровки"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки {filename}: {e}")
        return None

def draw_grid(img, step=100, color=(0, 255, 0), thickness=1):
    """Рисует сетку на изображении для проверки прямых линий"""
    h, w = img.shape[:2]
    img_with_grid = img.copy()
    
    # Вертикальные линии
    for x in range(0, w, step):
        cv2.line(img_with_grid, (x, 0), (x, h), color, thickness)
    
    # Горизонтальные линии
    for y in range(0, h, step):
        cv2.line(img_with_grid, (0, y), (w, y), color, thickness)
    
    return img_with_grid

def process_frame(img, calib_params, camera_name, alpha=1.0):
    """Применяет коррекцию дисторсии к кадру"""
    h, w = img.shape[:2]
    
    if calib_params is None:
        print(f"⚠️  Нет параметров для {camera_name}")
        return img
    
    # Получаем параметры
    mtx = calib_params.get('mtx')
    dist = calib_params.get('dist')
    
    if mtx is None or dist is None:
        print(f"⚠️  Неполные параметры для {camera_name}")
        return img
    
    # Вычисляем новую матрицу камеры
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
    
    # Применяем коррекцию
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # Обрезка по ROI если нужно
    if alpha < 1.0 and roi is not None:
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            dst = dst[y:y+h_roi, x:x+w_roi]
    
    return dst

def create_comparison(original, undist_old, undist_new, title="Comparison"):
    """Создает изображение для сравнения: оригинал, старая и новая коррекция"""
    h, w = original.shape[:2]
    
    # Уменьшаем для отображения
    scale = 0.4  # Масштаб для preview
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Ресайзим все изображения
    orig_small = cv2.resize(original, (new_w, new_h))
    old_small = cv2.resize(undist_old, (new_w, new_h))
    new_small = cv2.resize(undist_new, (new_w, new_h))
    
    # Добавляем сетку
    orig_grid = draw_grid(orig_small, step=50, thickness=1)
    old_grid = draw_grid(old_small, step=50, thickness=1)
    new_grid = draw_grid(new_small, step=50, thickness=1)
    
    # Создаем общее изображение
    gap = 20
    total_width = new_w * 3 + gap * 2
    total_height = new_h + 100  # Место для подписей
    
    comparison = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Размещаем изображения
    comparison[50:50+new_h, 0:new_w] = orig_grid
    comparison[50:50+new_h, new_w+gap:new_w*2+gap] = old_grid
    comparison[50:50+new_h, new_w*2+gap*2:] = new_grid
    
    # Добавляем подписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (new_w//4, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(comparison, "Old Calibration", (new_w + gap + new_w//4, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(comparison, "Fixed (Left params)", (new_w*2 + gap*2 + new_w//4, 30), font, 0.8, (0, 0, 0), 2)
    
    # Заголовок
    cv2.putText(comparison, title, (total_width//2 - 100, total_height - 20), 
                font, 1.0, (0, 0, 0), 2)
    
    return comparison

def main():
    print("🔍 Тестирование коррекции дисторсии")
    print("=" * 60)
    
    # Загружаем калибровки
    print("📂 Загрузка файлов калибровки...")
    old_calib = load_calibration('calibration_result_standard.pkl')
    new_calib = load_calibration('calibration_mono_fixed.pkl')
    
    if old_calib is None or new_calib is None:
        print("❌ Не удалось загрузить файлы калибровки")
        return
    
    # Директории с изображениями
    left_dir = "data/left"
    right_dir = "data/right"
    pairs_dir = "data/pairs"
    
    # Создаем директорию для результатов
    output_dir = "undistortion_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Результаты будут сохранены в: {output_dir}/")
    
    # Тест 1: Левая камера (должна быть одинаковой)
    print("\n📷 Тест левой камеры...")
    left_images = [f for f in os.listdir(left_dir) if f.endswith(('.jpg', '.png'))]
    if left_images:
        test_img_path = os.path.join(left_dir, left_images[0])
        img_left = cv2.imread(test_img_path)
        
        if img_left is not None:
            # Старая калибровка
            undist_old = process_frame(img_left, old_calib['left_camera'], "Left (old)")
            # Новая калибровка (должна быть такой же)
            undist_new = process_frame(img_left, new_calib['left_camera'], "Left (new)")
            
            # Сравнение
            comparison = create_comparison(img_left, undist_old, undist_new, "Left Camera Test")
            cv2.imwrite(os.path.join(output_dir, "left_camera_comparison.jpg"), comparison)
            print("  ✅ Сохранено: left_camera_comparison.jpg")
    
    # Тест 2: Правая камера (здесь должна быть разница!)
    print("\n📷 Тест правой камеры...")
    right_images = [f for f in os.listdir(right_dir) if f.endswith(('.jpg', '.png'))]
    if right_images:
        test_img_path = os.path.join(right_dir, right_images[0])
        img_right = cv2.imread(test_img_path)
        
        if img_right is not None:
            # Старая калибровка (с проблемой)
            undist_old = process_frame(img_right, old_calib['right_camera'], "Right (old)")
            # Новая калибровка (с параметрами левой)
            undist_new = process_frame(img_right, new_calib['right_camera'], "Right (fixed)")
            
            # Сравнение
            comparison = create_comparison(img_right, undist_old, undist_new, "Right Camera Test - MAIN DIFFERENCE HERE!")
            cv2.imwrite(os.path.join(output_dir, "right_camera_comparison.jpg"), comparison)
            print("  ✅ Сохранено: right_camera_comparison.jpg")
    
    # Тест 3: Создаем детальное сравнение для правой камеры
    print("\n🔬 Детальный анализ правой камеры...")
    if 'img_right' in locals():
        # Тестируем разные значения alpha
        alphas = [0.0, 0.5, 1.0]
        for alpha in alphas:
            undist_alpha = process_frame(img_right, new_calib['right_camera'], 
                                       f"Right alpha={alpha}", alpha)
            output_name = f"right_camera_alpha_{alpha}.jpg"
            
            # Добавляем сетку для проверки
            with_grid = draw_grid(undist_alpha, step=200, thickness=2)
            cv2.imwrite(os.path.join(output_dir, output_name), with_grid)
            print(f"  ✅ Сохранено: {output_name}")
    
    # Тест 4: Проверка на реальном футбольном поле (если есть)
    print("\n⚽ Поиск кадров с футбольным полем...")
    # Ищем в pairs директории
    if os.path.exists(pairs_dir):
        field_images = [f for f in os.listdir(pairs_dir) if 'field' in f.lower() or 'pitch' in f.lower()]
        if not field_images:  # Если нет специальных, берем любые
            field_images = [f for f in os.listdir(pairs_dir) if f.endswith(('.jpg', '.png'))][:2]
        
        for img_file in field_images[:2]:  # Максимум 2 изображения
            img_path = os.path.join(pairs_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Определяем какая камера
                camera_type = 'left' if 'left' in img_file or 'cam0' in img_file else 'right'
                
                # Применяем новую калибровку
                undist = process_frame(img, new_calib[f'{camera_type}_camera'], 
                                     f"{camera_type} camera", alpha=1.0)
                
                # Сохраняем с сеткой
                with_grid = draw_grid(undist, step=150, color=(0, 255, 255), thickness=2)
                output_name = f"field_test_{camera_type}_{os.path.basename(img_file)}"
                cv2.imwrite(os.path.join(output_dir, output_name), with_grid)
                print(f"  ✅ Сохранено: {output_name}")
    
    print(f"\n✅ Готово! Проверьте результаты в папке '{output_dir}/'")
    print("\n📊 Что смотреть:")
    print("  1. right_camera_comparison.jpg - основное сравнение")
    print("  2. Зеленые линии сетки должны быть прямыми")
    print("  3. На правой камере должна исчезнуть 'бочка'")
    print("  4. Линии футбольного поля должны быть прямыми")

if __name__ == "__main__":
    main()