#!/usr/bin/env python3
"""
Скрипт для копирования параметров левой камеры в правую
и удаления информации о стереокалибровке
"""

import pickle
import numpy as np
import sys
import os

def copy_left_to_right_camera(input_file='calibration_result_standard.pkl', 
                             output_file='calibration_mono_fixed.pkl'):
    """
    Загружает файл калибровки, копирует параметры левой камеры в правую
    и удаляет информацию о стереокалибровке
    
    Args:
        input_file: исходный файл калибровки
        output_file: выходной файл с исправленными параметрами
    """
    
    print(f"📂 Загрузка файла: {input_file}")
    
    # Загружаем исходный файл
    try:
        with open(input_file, 'rb') as f:
            calib_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {input_file} не найден!")
        return False
    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return False
    
    # Проверяем наличие необходимых данных
    if 'left_camera' not in calib_data:
        print("❌ В файле нет данных левой камеры!")
        return False
    
    print("\n📊 Исходные параметры:")
    print("-" * 50)
    
    # Показываем текущие параметры
    left_cam = calib_data['left_camera']
    print("Левая камера:")
    print(f"  - Модель: {left_cam.get('model', 'standard')}")
    print(f"  - RMS ошибка: {left_cam.get('ret', 0):.3f}")
    if 'mtx' in left_cam:
        print(f"  - Фокусное расстояние: fx={left_cam['mtx'][0,0]:.1f}, fy={left_cam['mtx'][1,1]:.1f}")
        print(f"  - Главная точка: cx={left_cam['mtx'][0,2]:.1f}, cy={left_cam['mtx'][1,2]:.1f}")
    print(f"  - Коэффициенты дисторсии: {left_cam['dist'].ravel()}")
    
    if 'right_camera' in calib_data:
        right_cam = calib_data['right_camera']
        print("\nПравая камера (будет заменена):")
        print(f"  - Модель: {right_cam.get('model', 'standard')}")
        print(f"  - RMS ошибка: {right_cam.get('ret', 0):.3f}")
        if 'mtx' in right_cam:
            print(f"  - Фокусное расстояние: fx={right_cam['mtx'][0,0]:.1f}, fy={right_cam['mtx'][1,1]:.1f}")
        print(f"  - Коэффициенты дисторсии: {right_cam['dist'].ravel()}")
    
    # Создаем новую структуру данных
    new_calib_data = {}
    
    # Копируем параметры левой камеры
    new_calib_data['left_camera'] = left_cam.copy()
    
    # Создаем идентичные параметры для правой камеры
    # Глубокое копирование, чтобы избежать ссылок на одни и те же объекты
    right_cam_new = {}
    for key, value in left_cam.items():
        if isinstance(value, np.ndarray):
            right_cam_new[key] = value.copy()
        elif isinstance(value, list):
            right_cam_new[key] = [item.copy() if isinstance(item, np.ndarray) else item 
                                  for item in value]
        else:
            right_cam_new[key] = value
    
    new_calib_data['right_camera'] = right_cam_new
    
    # Копируем дополнительные параметры (размер доски и т.д.)
    if 'board_size' in calib_data:
        new_calib_data['board_size'] = calib_data['board_size']
    if 'square_size' in calib_data:
        new_calib_data['square_size'] = calib_data['square_size']
    
    # НЕ копируем стереокалибровку
    print("\n🗑️  Удаление данных стереокалибровки")
    
    # Сохраняем новый файл
    print(f"\n💾 Сохранение в файл: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(new_calib_data, f)
    
    print("\n✅ Готово! Результат:")
    print("-" * 50)
    print("  - Параметры правой камеры теперь идентичны левой")
    print("  - Информация о стереокалибровке удалена")
    print("  - Сохранены только монокулярные параметры камер")
    
    # Проверяем размер файлов
    original_size = os.path.getsize(input_file) / 1024
    new_size = os.path.getsize(output_file) / 1024
    print(f"\n📦 Размер файла: {original_size:.1f} KB → {new_size:.1f} KB")
    
    return True


def verify_calibration(filename):
    """
    Проверяет содержимое файла калибровки
    """
    print(f"\n🔍 Проверка файла: {filename}")
    print("-" * 50)
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print("Содержимое файла:")
        for key in data.keys():
            print(f"  - {key}")
            
        if 'left_camera' in data and 'right_camera' in data:
            left_dist = data['left_camera']['dist'].ravel()
            right_dist = data['right_camera']['dist'].ravel()
            
            print("\nСравнение дисторсии:")
            print(f"  Левая:  {left_dist}")
            print(f"  Правая: {right_dist}")
            print(f"  Идентичны: {np.allclose(left_dist, right_dist)}")
            
            if 'mtx' in data['left_camera'] and 'mtx' in data['right_camera']:
                left_fx = data['left_camera']['mtx'][0,0]
                right_fx = data['right_camera']['mtx'][0,0]
                print(f"\nФокусное расстояние:")
                print(f"  Левая:  {left_fx:.1f} px")
                print(f"  Правая: {right_fx:.1f} px")
                print(f"  Идентичны: {np.isclose(left_fx, right_fx)}")
                
    except Exception as e:
        print(f"❌ Ошибка при проверке: {e}")


def main():
    print("🔧 Исправление калибровки камер")
    print("=" * 60)
    
    # Определяем имена файлов
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'calibration_result_standard.pkl'
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'calibration_mono_fixed.pkl'
    
    # Выполняем копирование
    success = copy_left_to_right_camera(input_file, output_file)
    
    if success:
        # Проверяем результат
        verify_calibration(output_file)
        
        print("\n📝 Использование:")
        print(f"  Теперь вы можете использовать файл '{output_file}'")
        print("  Обе камеры будут иметь одинаковые параметры")
        print("  Линии на футбольном поле должны быть прямыми")
    else:
        print("\n❌ Операция не выполнена")
        

if __name__ == "__main__":
    print("Использование:")
    print("  python3 fix_calibration.py [входной_файл] [выходной_файл]")
    print("  По умолчанию:")
    print("    входной:  calibration_result_standard.pkl")
    print("    выходной: calibration_mono_fixed.pkl")
    print()
    
    main()