import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

# Загружаем калибровку
with open('calibration_result_standard.pkl', 'rb') as f:
    calib = pickle.load(f)

def test_single_camera(camera_data, test_image_path, camera_name):
    """Тестирует калибровку одной камеры"""
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]
    
    # Исправляем дисторсию
    mtx = camera_data['mtx']
    dist = camera_data['dist']
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # Находим углы на исправленном изображении
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
    
    if ret:
        # Рисуем углы
        cv2.drawChessboardCorners(undistorted, (10, 7), corners, ret)
        
        # Вычисляем ошибку репроекции для этого изображения
        objp = np.zeros((10*7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * 23.5
        
        _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        projected, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        
        error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
        print(f"{camera_name} - ошибка репроекции на тестовом изображении: {error:.3f} px")
    
    # Сохраняем результат
    result = np.hstack([
        cv2.resize(img, (960, 540)),
        cv2.resize(undistorted, (960, 540))
    ])
    cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, "Undistorted", (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, f"RMS: {camera_data['ret']:.3f}", (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(f'test_{camera_name}.jpg', result)
    print(f"Сохранено: test_{camera_name}.jpg")
    
    return error

# Тестируем обе камеры
left_images = glob.glob("calibration_data/left/*.jpg")
right_images = glob.glob("calibration_data/right/*.jpg")

if left_images:
    test_single_camera(calib['left_camera'], left_images[0], 'left')
    
if right_images:
    test_single_camera(calib['right_camera'], right_images[0], 'right')

# Проверяем параметры
print("\n📊 Сравнение параметров камер:")
print(f"Левая:  FOV={calib['left_camera']['fovx']:.1f}°, f={calib['left_camera']['mtx'][0,0]:.0f}px")
print(f"Правая: FOV={calib['right_camera']['fovx']:.1f}°, f={calib['right_camera']['mtx'][0,0]:.0f}px")
print(f"\nКоэффициенты дисторсии:")
print(f"Левая:  {calib['left_camera']['dist'].ravel()[:5]}")
print(f"Правая: {calib['right_camera']['dist'].ravel()[:5]}")