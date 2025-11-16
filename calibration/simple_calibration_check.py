#!/usr/bin/env python3
"""
Простая проверка калибровочных изображений с реалистичными критериями
Находит ДЕЙСТВИТЕЛЬНО проблемные снимки, влияющие на высокий RMS
"""

import cv2
import numpy as np
import os
import glob
import json

CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def simple_check_image(img_path):
    """Простая проверка одного изображения"""
    img = cv2.imread(img_path)
    if img is None:
        return None, "Не загружается"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        return None, "Доска не найдена"

    # Уточнение
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Метрики
    metrics = {}

    # 1. Размытие (главный враг точности!)
    metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Яркость
    metrics['brightness'] = np.mean(gray)

    # 3. Покрытие
    corners_array = corners.reshape(-1, 2)
    min_x, min_y = corners_array.min(axis=0)
    max_x, max_y = corners_array.max(axis=0)
    board_area = (max_x - min_x) * (max_y - min_y)
    img_area = gray.shape[0] * gray.shape[1]
    metrics['coverage'] = board_area / img_area * 100  # в процентах

    # 4. Попытка одиночной калибровки (самый важный показатель!)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners], gray.shape[::-1], None, None
    )

    if ret:
        # Репроекция
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
        error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        metrics['reproj_error'] = error
    else:
        metrics['reproj_error'] = 999

    return metrics, None


def analyze_camera(img_dir, camera_name):
    """Анализ одной камеры"""
    print(f"\n{'='*80}")
    print(f"{camera_name}")
    print(f"{'='*80}")

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Найдено: {len(images)} изображений\n")

    results = {
        'total': len(images),
        'analyzed': [],
        'bad': [],
        'very_bad': [],  # Критические проблемы
        'not_found': []
    }

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        print(f"\r[{i+1:2d}/{len(images)}] {fname:<30}", end='', flush=True)

        metrics, error = simple_check_image(img_path)

        if error:
            results['not_found'].append({
                'filename': fname,
                'reason': error
            })
            continue

        # Классификация проблем
        problems = []
        severity = 0  # 0=OK, 1=minor, 2=moderate, 3=severe

        # КРИТЕРИЙ 1: Размытие (ОЧЕНЬ ВАЖНО!)
        if metrics['sharpness'] < 30:
            problems.append(f"Очень размыто ({metrics['sharpness']:.0f})")
            severity = max(severity, 3)  # КРИТИЧНО
        elif metrics['sharpness'] < 60:
            problems.append(f"Размыто ({metrics['sharpness']:.0f})")
            severity = max(severity, 2)

        # КРИТЕРИЙ 2: Покрытие
        if metrics['coverage'] < 1.0:  # Меньше 1%
            problems.append(f"Слишком далеко ({metrics['coverage']:.1f}%)")
            severity = max(severity, 3)  # КРИТИЧНО - доска слишком мала
        elif metrics['coverage'] > 75:  # Больше 75%
            problems.append(f"Слишком близко ({metrics['coverage']:.1f}%)")
            severity = max(severity, 2)
        elif metrics['coverage'] < 3:
            problems.append(f"Далеко ({metrics['coverage']:.1f}%)")
            severity = max(severity, 1)

        # КРИТЕРИЙ 3: Яркость
        if metrics['brightness'] < 30:
            problems.append(f"Очень темно ({metrics['brightness']:.0f})")
            severity = max(severity, 2)
        elif metrics['brightness'] > 230:
            problems.append(f"Пересвет ({metrics['brightness']:.0f})")
            severity = max(severity, 2)

        # КРИТЕРИЙ 4: Ошибка репроекции (ГЛАВНЫЙ ПОКАЗАТЕЛЬ!)
        if metrics['reproj_error'] > 2.0:
            problems.append(f"Высокая ошибка репроекции ({metrics['reproj_error']:.2f})")
            severity = max(severity, 3)  # КРИТИЧНО
        elif metrics['reproj_error'] > 1.0:
            problems.append(f"Повышенная ошибка ({metrics['reproj_error']:.2f})")
            severity = max(severity, 2)

        entry = {
            'filename': fname,
            'path': img_path,
            'metrics': metrics,
            'problems': problems,
            'severity': severity
        }

        results['analyzed'].append(entry)

        if severity >= 3:
            results['very_bad'].append(entry)
        elif severity >= 1:
            results['bad'].append(entry)

    print(f"\n\n{'='*80}")
    print("РЕЗУЛЬТАТЫ:")
    print(f"{'='*80}")
    print(f"Всего изображений:          {results['total']}")
    print(f"Проанализировано:           {len(results['analyzed'])}")
    print(f"✓ Хорошие:                  {results['total'] - len(results['bad']) - len(results['very_bad']) - len(results['not_found'])} ({(results['total'] - len(results['bad']) - len(results['very_bad']) - len(results['not_found']))/results['total']*100:.1f}%)")
    print(f"⚠ Проблемные:               {len(results['bad'])} ({len(results['bad'])/results['total']*100:.1f}%)")
    print(f"❌ КРИТИЧЕСКИ плохие:       {len(results['very_bad'])} ({len(results['very_bad'])/results['total']*100:.1f}%)")
    print(f"❌ Доска не найдена:        {len(results['not_found'])} ({len(results['not_found'])/results['total']*100:.1f}%)")

    # Детали КРИТИЧЕСКИ плохих
    if results['very_bad']:
        print(f"\n❌ КРИТИЧЕСКИ ПЛОХИЕ ИЗОБРАЖЕНИЯ (УДАЛИТЬ!):")
        for entry in results['very_bad']:
            print(f"\n  {entry['filename']}:")
            for problem in entry['problems']:
                print(f"    - {problem}")
            print(f"    Метрики: sharpness={entry['metrics']['sharpness']:.1f}, "
                  f"coverage={entry['metrics']['coverage']:.1f}%, "
                  f"reproj_error={entry['metrics']['reproj_error']:.3f}")

    # Детали проблемных
    if results['bad'] and len(results['bad']) <= 15:
        print(f"\n⚠ ПРОБЛЕМНЫЕ ИЗОБРАЖЕНИЯ (рассмотреть удаление):")
        for entry in results['bad']:
            print(f"\n  {entry['filename']}:")
            for problem in entry['problems']:
                print(f"    - {problem}")

    # Изображения без доски
    if results['not_found']:
        print(f"\n❌ НЕ НАЙДЕНА ДОСКА (УДАЛИТЬ!):")
        for entry in results['not_found']:
            print(f"  • {entry['filename']}: {entry['reason']}")

    # Средние метрики хороших изображений
    good_entries = [e for e in results['analyzed'] if e['severity'] == 0]
    if good_entries:
        print(f"\n✓ СРЕДНИЕ МЕТРИКИ ХОРОШИХ ИЗОБРАЖЕНИЙ ({len(good_entries)} шт):")
        avg_sharp = np.mean([e['metrics']['sharpness'] for e in good_entries])
        avg_cover = np.mean([e['metrics']['coverage'] for e in good_entries])
        avg_reproj = np.mean([e['metrics']['reproj_error'] for e in good_entries])
        print(f"  Резкость (sharpness):       {avg_sharp:.1f}")
        print(f"  Покрытие (coverage):        {avg_cover:.1f}%")
        print(f"  Ошибка репроекции:          {avg_reproj:.3f} пикселей")

    return results


def compare_and_recommend(cam0_results, cam1_results):
    """Сравнение камер и рекомендации"""
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ КАМЕР")
    print(f"{'='*80}\n")

    print(f"{'Параметр':<35} {'Камера 0':>15} {'Камера 1':>15}")
    print(f"{'-'*70}")
    print(f"{'Всего изображений':<35} {cam0_results['total']:>15} {cam1_results['total']:>15}")

    cam0_good = cam0_results['total'] - len(cam0_results['bad']) - len(cam0_results['very_bad']) - len(cam0_results['not_found'])
    cam1_good = cam1_results['total'] - len(cam1_results['bad']) - len(cam1_results['very_bad']) - len(cam1_results['not_found'])

    print(f"{'Хороших':<35} {cam0_good:>15} {cam1_good:>15}")
    print(f"{'Проблемных':<35} {len(cam0_results['bad']):>15} {len(cam1_results['bad']):>15}")
    print(f"{'КРИТИЧЕСКИ плохих':<35} {len(cam0_results['very_bad']):>15} {len(cam1_results['very_bad']):>15}")
    print(f"{'Доска не найдена':<35} {len(cam0_results['not_found']):>15} {len(cam1_results['not_found']):>15}")

    # Средние метрики
    cam0_good_entries = [e for e in cam0_results['analyzed'] if e['severity'] == 0]
    cam1_good_entries = [e for e in cam1_results['analyzed'] if e['severity'] == 0]

    if cam0_good_entries and cam1_good_entries:
        print(f"\nСРЕДНИЕ МЕТРИКИ (только хорошие изображения):")
        print(f"{'Метрика':<35} {'Камера 0':>15} {'Камера 1':>15}")
        print(f"{'-'*70}")

        cam0_sharp = np.mean([e['metrics']['sharpness'] for e in cam0_good_entries])
        cam1_sharp = np.mean([e['metrics']['sharpness'] for e in cam1_good_entries])
        print(f"{'Резкость (sharpness)':<35} {cam0_sharp:>15.1f} {cam1_sharp:>15.1f}")

        cam0_cover = np.mean([e['metrics']['coverage'] for e in cam0_good_entries])
        cam1_cover = np.mean([e['metrics']['coverage'] for e in cam1_good_entries])
        print(f"{'Покрытие (%)':<35} {cam0_cover:>15.1f} {cam1_cover:>15.1f}")

        cam0_reproj = np.mean([e['metrics']['reproj_error'] for e in cam0_good_entries])
        cam1_reproj = np.mean([e['metrics']['reproj_error'] for e in cam1_good_entries])
        print(f"{'Ошибка репроекции (px)':<35} {cam0_reproj:>15.3f} {cam1_reproj:>15.3f}")

    # РЕКОМЕНДАЦИИ
    print(f"\n{'='*80}")
    print("РЕКОМЕНДАЦИИ")
    print(f"{'='*80}\n")

    total_bad = len(cam0_results['very_bad']) + len(cam0_results['not_found']) + len(cam1_results['very_bad']) + len(cam1_results['not_found'])

    if total_bad > 0:
        print(f"❌ НЕОБХОДИМО УДАЛИТЬ {total_bad} КРИТИЧЕСКИ ПЛОХИХ ИЗОБРАЖЕНИЙ:\n")

        if len(cam0_results['very_bad']) + len(cam0_results['not_found']) > 0:
            print(f"Камера 0 (левая) - удалить {len(cam0_results['very_bad']) + len(cam0_results['not_found'])} файлов:")
            for e in cam0_results['very_bad']:
                print(f"  rm calibration/calibration_result/left/{e['filename']}")
            for e in cam0_results['not_found']:
                print(f"  rm calibration/calibration_result/left/{e['filename']}")

        if len(cam1_results['very_bad']) + len(cam1_results['not_found']) > 0:
            print(f"\nКамера 1 (правая) - удалить {len(cam1_results['very_bad']) + len(cam1_results['not_found'])} файлов:")
            for e in cam1_results['very_bad']:
                print(f"  rm calibration/calibration_result/right/{e['filename']}")
            for e in cam1_results['not_found']:
                print(f"  rm calibration/calibration_result/right/{e['filename']}")

    if cam0_good < 20 or cam1_good < 20:
        print(f"\n⚠ НЕДОСТАТОЧНО хороших изображений!")
        print(f"  Камера 0: {cam0_good} хороших (нужно минимум 20)")
        print(f"  Камера 1: {cam1_good} хороших (нужно минимум 20)")
        print(f"\n  💡 После удаления плохих изображений, добавьте:")
        print(f"     - Снимки с разных расстояний (5-40% покрытия кадра)")
        print(f"     - Разные углы наклона доски")
        print(f"     - Убедитесь в хорошем освещении и фокусе")
    else:
        print(f"\n✓ Достаточно хороших изображений для калибровки!")
        print(f"  Камера 0: {cam0_good} хороших")
        print(f"  Камера 1: {cam1_good} хороших")

    print(f"\n{'='*80}")
    print("ОБЪЯСНЕНИЕ ВЫСОКОГО RMS У КАМЕРЫ 0")
    print(f"{'='*80}\n")

    print(f"RMS (Root Mean Square) = 12.0 пикселей слишком высокий!")
    print(f"Нормальный RMS для хорошей калибровки: 0.1-0.5 пикселей")
    print(f"\nВозможные причины:")
    print(f"  1. Размытые изображения (низкий sharpness)")
    print(f"  2. Слишком маленькая доска в кадре (coverage < 3%)")
    print(f"  3. Ошибки обнаружения углов на некоторых кадрах")
    print(f"  4. Недостаточное разнообразие позиций доски")
    print(f"\nРешение:")
    print(f"  1. Удалите критически плохие изображения (см. выше)")
    print(f"  2. Повторите калибровку с очищенным набором")
    print(f"  3. Если RMS всё равно высокий - добавьте новые качественные снимки")

    print(f"\n{'='*80}\n")

    # Сохранение в JSON
    report = {
        'camera_0': {
            'total': cam0_results['total'],
            'good': cam0_good,
            'bad': len(cam0_results['bad']),
            'very_bad': len(cam0_results['very_bad']),
            'not_found': len(cam0_results['not_found']),
            'files_to_remove': [e['filename'] for e in cam0_results['very_bad']] + [e['filename'] for e in cam0_results['not_found']],
            'problematic_files': [{'file': e['filename'], 'problems': e['problems']} for e in cam0_results['bad']]
        },
        'camera_1': {
            'total': cam1_results['total'],
            'good': cam1_good,
            'bad': len(cam1_results['bad']),
            'very_bad': len(cam1_results['very_bad']),
            'not_found': len(cam1_results['not_found']),
            'files_to_remove': [e['filename'] for e in cam1_results['very_bad']] + [e['filename'] for e in cam1_results['not_found']],
            'problematic_files': [{'file': e['filename'], 'problems': e['problems']} for e in cam1_results['bad']]
        }
    }

    with open('calibration/calibration_cleanup_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Отчет сохранен: calibration/calibration_cleanup_report.json\n")


def main():
    cam0_dir = "calibration/calibration_result/left"
    cam1_dir = "calibration/calibration_result/right"

    cam0_results = analyze_camera(cam0_dir, "КАМЕРА 0 (ЛЕВАЯ)")
    cam1_results = analyze_camera(cam1_dir, "КАМЕРА 1 (ПРАВАЯ)")

    compare_and_recommend(cam0_results, cam1_results)


if __name__ == "__main__":
    main()
