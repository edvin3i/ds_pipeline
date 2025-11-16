#!/usr/bin/env python3
"""
Создание calibration_result_standard.pkl для pano_cuda_debug_stages.py
Формат из run_full_calibration.py
"""

import json
import pickle
import numpy as np
from pathlib import Path

def main():
    calib_dir = Path(__file__).parent

    print("=" * 70)
    print("Создание calibration_result_standard.pkl")
    print("=" * 70)

    # 1. Загрузка индивидуальной калибровки
    print("\n[1] Загрузка калибровки из JSON...")
    individual_json = calib_dir / "calibration_results_cleaned.json"
    stereo_json = calib_dir / "stereo_essential_matrix_results.json"

    with open(individual_json, 'r') as f:
        individual_data = json.load(f)

    with open(stereo_json, 'r') as f:
        stereo_data = json.load(f)

    # 2. Формирование структуры данных как в run_full_calibration.py
    print("[2] Формирование структуры данных...")

    # pano_cuda_debug_stages.py ожидает структуру:
    # data['left_camera']['mtx'], data['left_camera']['dist']
    # data['right_camera']['mtx'], data['right_camera']['dist']
    # data['stereo']['R'], data['stereo']['T']

    calibration_data = {
        'left_camera': {
            'mtx': np.array(individual_data['camera_0']['camera_matrix'], dtype=np.float64),
            'dist': np.array(individual_data['camera_0']['dist_coeffs'], dtype=np.float64),
            'rms': individual_data['camera_0']['rms'],
            'num_images': individual_data['camera_0']['num_images']
        },
        'right_camera': {
            'mtx': np.array(individual_data['camera_1']['camera_matrix'], dtype=np.float64),
            'dist': np.array(individual_data['camera_1']['dist_coeffs'], dtype=np.float64),
            'rms': individual_data['camera_1']['rms'],
            'num_images': individual_data['camera_1']['num_images']
        },
        'stereo': {
            'R': np.array(stereo_data['R'], dtype=np.float64),
            'T': np.array(stereo_data['T_normalized'], dtype=np.float64),
            'E': np.array(stereo_data['E'], dtype=np.float64),
            'F': np.array(stereo_data['F'], dtype=np.float64),
            'angle_deg': stereo_data['angle_between_cameras'],
            'num_inliers': stereo_data['inliers'],
            'total_points': stereo_data['num_points'],
            'method': 'essential_matrix'
        },
        'metadata': {
            'chessboard_size': (8, 6),
            'square_size_mm': 25.0,
            'camera_model': 'FSM:GO-IMX678C-M12-L100A-PM-A1Q1',
            'fov_deg': 100
        }
    }

    # 3. Вывод информации
    print("\n[3] Параметры калибровки:")
    print(f"\n  Левая камера (camera_0):")
    print(f"    RMS: {calibration_data['left_camera']['rms']:.3f} px")
    print(f"    Изображений: {calibration_data['left_camera']['num_images']}")
    print(f"    K:\n{calibration_data['left_camera']['mtx']}")
    print(f"    D: {calibration_data['left_camera']['dist'].ravel()}")

    print(f"\n  Правая камера (camera_1):")
    print(f"    RMS: {calibration_data['right_camera']['rms']:.3f} px")
    print(f"    Изображений: {calibration_data['right_camera']['num_images']}")
    print(f"    K:\n{calibration_data['right_camera']['mtx']}")
    print(f"    D: {calibration_data['right_camera']['dist'].ravel()}")

    print(f"\n  Стерео калибровка:")
    print(f"    Угол между камерами: {calibration_data['stereo']['angle_deg']:.2f}°")
    print(f"    Инлаеров: {calibration_data['stereo']['num_inliers']}/{calibration_data['stereo']['total_points']}")
    print(f"    R:\n{calibration_data['stereo']['R']}")
    print(f"    T (нормализованный): {calibration_data['stereo']['T'].ravel()}")

    # 4. Сохранение в .pkl
    print("\n[4] Сохранение в .pkl файлы...")

    # Основной файл для pano_cuda_debug_stages.py
    pkl_standard = calib_dir / "calibration_result_standard.pkl"
    with open(pkl_standard, 'wb') as f:
        pickle.dump(calibration_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Стандартный файл: {pkl_standard}")

    # Также сохраним в корень проекта (где ищет pano_cuda_debug_stages.py)
    pkl_root = calib_dir.parent / "calibration_result_standard.pkl"
    with open(pkl_root, 'wb') as f:
        pickle.dump(calibration_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Корень проекта: {pkl_root}")

    # 5. Проверка загрузки
    print("\n[5] Проверка загрузки...")
    with open(pkl_standard, 'rb') as f:
        loaded = pickle.load(f)

    # Проверка соответствия формату pano_cuda_debug_stages.py
    print("✓ Проверка структуры:")
    print(f"  'left_camera' in data: {('left_camera' in loaded)}")
    print(f"  'right_camera' in data: {('right_camera' in loaded)}")
    print(f"  'stereo' in data: {('stereo' in loaded)}")

    K_L = loaded['left_camera']['mtx']
    D_L = loaded['left_camera']['dist'].ravel()
    K_R = loaded['right_camera']['mtx']
    D_R = loaded['right_camera']['dist'].ravel()
    R_LR = loaded['stereo']['R']
    T_LR = loaded['stereo']['T']

    print(f"  K_L shape: {K_L.shape} (ожидается 3x3)")
    print(f"  D_L shape: {D_L.shape} (ожидается (5,))")
    print(f"  K_R shape: {K_R.shape} (ожидается 3x3)")
    print(f"  D_R shape: {D_R.shape} (ожидается (5,))")
    print(f"  R_LR shape: {R_LR.shape} (ожидается 3x3)")
    print(f"  T_LR shape: {T_LR.shape} (ожидается (3,1) или (3,))")

    # 6. Пример использования
    print("\n" + "=" * 70)
    print("Пример использования в pano_cuda_debug_stages.py:")
    print("=" * 70)
    print("""
# В функции load_calibration():
with open('calibration_result_standard.pkl', 'rb') as f:
    data = pickle.load(f)

K_L = data['left_camera']['mtx']
D_L = data['left_camera']['dist'].ravel()
K_R = data['right_camera']['mtx']
D_R = data['right_camera']['dist'].ravel()
R_LR = data['stereo']['R']
T_LR = data['stereo']['T']
""")

    print("\n✅ Калибровочный файл создан успешно!")
    print(f"\nФайлы:")
    print(f"  • {pkl_standard}")
    print(f"  • {pkl_root}")
    print(f"\nТеперь можно запускать: python3 pano_cuda_debug_stages.py")

if __name__ == "__main__":
    main()
