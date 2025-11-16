#!/usr/bin/env python3
"""
Комплексный анализ качества калибровочных изображений для стерео-системы
Сравнивает две камеры и находит проблемные изображения
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import json

# Параметры шахматной доски
CHECKERBOARD = (8, 6)  # Внутренние углы (8 по горизонтали, 6 по вертикали)
SQUARE_SIZE = 1.0  # Размер клетки в мм (не важно для этого анализа)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class CalibrationImageAnalyzer:
    def __init__(self, image_path, board_size=CHECKERBOARD):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        self.board_size = board_size
        self.img = None
        self.gray = None
        self.corners = None
        self.found = False
        self.metrics = {}

    def analyze(self):
        """Полный анализ изображения"""
        # Загрузка изображения
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            return False

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Поиск углов
        ret, corners = cv2.findChessboardCorners(
            self.gray,
            self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        self.found = ret

        if not ret:
            self.metrics['found'] = False
            return False

        # Уточнение углов
        self.corners = cv2.cornerSubPix(
            self.gray, corners, (11, 11), (-1, -1), criteria
        )

        # Вычисление всех метрик
        self.metrics['found'] = True
        self._compute_sharpness()
        self._compute_brightness()
        self._compute_contrast()
        self._compute_coverage()
        self._compute_corner_quality()
        self._compute_perspective_distortion()
        self._compute_reprojection_error()

        return True

    def _compute_sharpness(self):
        """Резкость изображения (Laplacian variance)"""
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        self.metrics['sharpness'] = laplacian.var()

    def _compute_brightness(self):
        """Яркость изображения"""
        self.metrics['brightness'] = np.mean(self.gray)
        self.metrics['brightness_std'] = np.std(self.gray)

    def _compute_contrast(self):
        """Контраст изображения"""
        self.metrics['contrast'] = self.gray.max() - self.gray.min()

    def _compute_coverage(self):
        """Покрытие кадра доской"""
        corners_array = self.corners.reshape(-1, 2)
        min_x, min_y = corners_array.min(axis=0)
        max_x, max_y = corners_array.max(axis=0)

        board_width = max_x - min_x
        board_height = max_y - min_y
        board_area = board_width * board_height

        img_area = self.gray.shape[0] * self.gray.shape[1]

        self.metrics['coverage'] = board_area / img_area
        self.metrics['board_width'] = board_width
        self.metrics['board_height'] = board_height

    def _compute_corner_quality(self):
        """Качество обнаруженных углов"""
        corners_array = self.corners.reshape(-1, 2)

        # Расстояния между соседними углами
        distances = []
        for i in range(len(corners_array) - 1):
            dist = np.linalg.norm(corners_array[i] - corners_array[i+1])
            distances.append(dist)

        self.metrics['corner_distance_mean'] = np.mean(distances)
        self.metrics['corner_distance_std'] = np.std(distances)
        self.metrics['corner_uniformity'] = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 999

    def _compute_perspective_distortion(self):
        """Искажение перспективы"""
        corners_array = self.corners.reshape(-1, 2)

        # Углы доски
        tl = corners_array[0]
        tr = corners_array[self.board_size[0]-1]
        bl = corners_array[-self.board_size[0]]
        br = corners_array[-1]

        # Ширина сверху и снизу
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)

        # Высота слева и справа
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        # Соотношение сторон
        aspect_top = width_top / height_left if height_left > 0 else 0
        aspect_bottom = width_bottom / height_right if height_right > 0 else 0

        self.metrics['width_ratio'] = width_bottom / width_top if width_top > 0 else 0
        self.metrics['height_ratio'] = height_right / height_left if height_left > 0 else 0
        self.metrics['perspective_distortion'] = abs(width_top - width_bottom) + abs(height_left - height_right)

    def _compute_reprojection_error(self):
        """Оценка ошибки репроекции для этого кадра"""
        # Подготовка объектных точек
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)

        # Калибровка только по этому кадру
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [objp], [self.corners], self.gray.shape[::-1], None, None
        )

        if ret:
            # Репроекция
            imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
            error = cv2.norm(self.corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.metrics['single_image_reproj_error'] = error
        else:
            self.metrics['single_image_reproj_error'] = 999

    def is_good_quality(self):
        """Проверка качества изображения по всем критериям"""
        if not self.found:
            return False, ["Доска не найдена"]

        issues = []

        # Критерий 1: Резкость
        if self.metrics['sharpness'] < 50:
            issues.append(f"Размытое (sharpness={self.metrics['sharpness']:.1f})")

        # Критерий 2: Яркость
        if self.metrics['brightness'] < 40 or self.metrics['brightness'] > 220:
            issues.append(f"Плохая яркость ({self.metrics['brightness']:.1f})")

        # Критерий 3: Покрытие
        if self.metrics['coverage'] < 0.05 or self.metrics['coverage'] > 0.7:
            issues.append(f"Плохое покрытие ({self.metrics['coverage']:.2%})")

        # Критерий 4: Однородность углов
        if self.metrics['corner_uniformity'] > 0.15:
            issues.append(f"Неравномерные углы ({self.metrics['corner_uniformity']:.3f})")

        # Критерий 5: Перспективное искажение
        if self.metrics['perspective_distortion'] > 300:
            issues.append(f"Сильная перспектива ({self.metrics['perspective_distortion']:.1f})")

        # Критерий 6: Ошибка репроекции
        if self.metrics.get('single_image_reproj_error', 999) > 1.0:
            issues.append(f"Высокая ошибка репроекции ({self.metrics.get('single_image_reproj_error', 999):.3f})")

        return len(issues) == 0, issues


def analyze_camera_set(image_dir, camera_name):
    """Анализ набора изображений одной камеры"""
    print(f"\n{'='*80}")
    print(f"АНАЛИЗ КАМЕРЫ: {camera_name}")
    print(f"{'='*80}")
    print(f"Директория: {image_dir}")

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    print(f"Найдено изображений: {len(images)}\n")

    results = {
        'camera': camera_name,
        'total': len(images),
        'analyzed': [],
        'not_found': [],
        'good': [],
        'bad': []
    }

    # Анализ каждого изображения
    for i, img_path in enumerate(images):
        print(f"\r  Обработка: {i+1}/{len(images)} - {os.path.basename(img_path):<40}", end='', flush=True)

        analyzer = CalibrationImageAnalyzer(img_path)
        success = analyzer.analyze()

        if not success or not analyzer.found:
            results['not_found'].append({
                'filename': analyzer.filename,
                'path': img_path
            })
            continue

        is_good, issues = analyzer.is_good_quality()

        result_entry = {
            'filename': analyzer.filename,
            'path': img_path,
            'metrics': analyzer.metrics,
            'is_good': is_good,
            'issues': issues
        }

        results['analyzed'].append(result_entry)

        if is_good:
            results['good'].append(result_entry)
        else:
            results['bad'].append(result_entry)

    print("\n")

    # Статистика
    print(f"{'='*80}")
    print(f"РЕЗУЛЬТАТЫ:")
    print(f"{'='*80}")
    print(f"Всего изображений:         {results['total']}")
    print(f"Проанализировано:          {len(results['analyzed'])}")
    print(f"✓ Хорошие:                 {len(results['good'])} ({len(results['good'])/results['total']*100:.1f}%)")
    print(f"⚠ Проблемные:              {len(results['bad'])} ({len(results['bad'])/results['total']*100:.1f}%)")
    print(f"❌ Доска не найдена:       {len(results['not_found'])} ({len(results['not_found'])/results['total']*100:.1f}%)")

    # Проблемные изображения
    if results['bad']:
        print(f"\n⚠ ПРОБЛЕМНЫЕ ИЗОБРАЖЕНИЯ:")
        for entry in results['bad']:
            print(f"  • {entry['filename']}")
            for issue in entry['issues']:
                print(f"      - {issue}")

    # Изображения без доски
    if results['not_found']:
        print(f"\n❌ ИЗОБРАЖЕНИЯ БЕЗ ДОСКИ:")
        for entry in results['not_found']:
            print(f"  • {entry['filename']}")

    # Средние метрики для хороших изображений
    if results['good']:
        print(f"\nСРЕДНИЕ МЕТРИКИ (хорошие изображения):")
        metrics_keys = ['sharpness', 'brightness', 'coverage', 'corner_uniformity',
                       'perspective_distortion', 'single_image_reproj_error']
        for key in metrics_keys:
            values = [r['metrics'][key] for r in results['good'] if key in r['metrics']]
            if values:
                print(f"  {key:30s}: {np.mean(values):8.3f} ± {np.std(values):6.3f}")

    return results


def compare_cameras(cam0_results, cam1_results):
    """Сравнение двух камер"""
    print(f"\n{'='*80}")
    print(f"СРАВНЕНИЕ КАМЕР")
    print(f"{'='*80}\n")

    print(f"{'Параметр':<30} {'Камера 0':>15} {'Камера 1':>15} {'Разница':>15}")
    print(f"{'-'*80}")

    print(f"{'Всего изображений':<30} {cam0_results['total']:>15} {cam1_results['total']:>15} {abs(cam0_results['total']-cam1_results['total']):>15}")
    print(f"{'Хороших':<30} {len(cam0_results['good']):>15} {len(cam1_results['good']):>15} {abs(len(cam0_results['good'])-len(cam1_results['good'])):>15}")
    print(f"{'Проблемных':<30} {len(cam0_results['bad']):>15} {len(cam1_results['bad']):>15} {abs(len(cam0_results['bad'])-len(cam1_results['bad'])):>15}")
    print(f"{'Не найдена доска':<30} {len(cam0_results['not_found']):>15} {len(cam1_results['not_found']):>15} {abs(len(cam0_results['not_found'])-len(cam1_results['not_found'])):>15}")

    # Сравнение средних метрик
    if cam0_results['good'] and cam1_results['good']:
        print(f"\n{'='*80}")
        print(f"СРАВНЕНИЕ СРЕДНИХ МЕТРИК (только хорошие изображения)")
        print(f"{'='*80}\n")

        print(f"{'Метрика':<30} {'Камера 0':>15} {'Камера 1':>15} {'Разница %':>15}")
        print(f"{'-'*80}")

        metrics_keys = ['sharpness', 'brightness', 'coverage', 'corner_uniformity',
                       'perspective_distortion', 'single_image_reproj_error']

        for key in metrics_keys:
            cam0_vals = [r['metrics'][key] for r in cam0_results['good'] if key in r['metrics']]
            cam1_vals = [r['metrics'][key] for r in cam1_results['good'] if key in r['metrics']]

            if cam0_vals and cam1_vals:
                cam0_mean = np.mean(cam0_vals)
                cam1_mean = np.mean(cam1_vals)
                diff_pct = abs(cam0_mean - cam1_mean) / max(cam0_mean, cam1_mean) * 100 if max(cam0_mean, cam1_mean) > 0 else 0

                print(f"{key:<30} {cam0_mean:>15.3f} {cam1_mean:>15.3f} {diff_pct:>14.1f}%")


def save_report(cam0_results, cam1_results, output_file):
    """Сохранение отчета в JSON"""
    report = {
        'camera_0': {
            'total': cam0_results['total'],
            'good_count': len(cam0_results['good']),
            'bad_count': len(cam0_results['bad']),
            'not_found_count': len(cam0_results['not_found']),
            'good_files': [r['filename'] for r in cam0_results['good']],
            'bad_files': [{'filename': r['filename'], 'issues': r['issues']} for r in cam0_results['bad']],
            'not_found_files': [r['filename'] for r in cam0_results['not_found']]
        },
        'camera_1': {
            'total': cam1_results['total'],
            'good_count': len(cam1_results['good']),
            'bad_count': len(cam1_results['bad']),
            'not_found_count': len(cam1_results['not_found']),
            'good_files': [r['filename'] for r in cam1_results['good']],
            'bad_files': [{'filename': r['filename'], 'issues': r['issues']} for r in cam1_results['bad']],
            'not_found_files': [r['filename'] for r in cam1_results['not_found']]
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Отчет сохранен: {output_file}")


def print_recommendations(cam0_results, cam1_results):
    """Печать рекомендаций"""
    print(f"\n{'='*80}")
    print(f"РЕКОМЕНДАЦИИ")
    print(f"{'='*80}\n")

    # Общие рекомендации
    total_good = len(cam0_results['good']) + len(cam1_results['good'])
    total_bad = len(cam0_results['bad']) + len(cam1_results['bad'])
    total_not_found = len(cam0_results['not_found']) + len(cam1_results['not_found'])

    if total_not_found > 0:
        print(f"❌ КРИТИЧНО: На {total_not_found} изображениях доска не найдена!")
        print(f"   Рекомендуется удалить эти файлы перед калибровкой.\n")

    if total_bad > total_good * 0.3:
        print(f"⚠ ВНИМАНИЕ: Более 30% изображений имеют проблемы!")
        print(f"   Рекомендуется пересмотреть процесс съемки.\n")

    # Специфические рекомендации для каждой камеры
    for cam_name, results in [("Камера 0", cam0_results), ("Камера 1", cam1_results)]:
        if len(results['bad']) > 0 or len(results['not_found']) > 0:
            print(f"\n{cam_name}:")

            if len(results['not_found']) > 0:
                print(f"  ❌ Удалить {len(results['not_found'])} файлов (доска не найдена)")

            if len(results['bad']) > 0:
                # Анализ типов проблем
                issue_types = {}
                for entry in results['bad']:
                    for issue in entry['issues']:
                        issue_type = issue.split('(')[0].strip()
                        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

                print(f"  ⚠ {len(results['bad'])} проблемных изображений:")
                for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
                    print(f"     - {issue_type}: {count} изображений")

                # Рекомендации по типам проблем
                if "Размытое" in issue_types:
                    print(f"\n  💡 Для устранения размытия:")
                    print(f"     - Держите камеру неподвижно")
                    print(f"     - Используйте штатив")
                    print(f"     - Увеличьте освещение")

                if "Плохое покрытие" in issue_types:
                    print(f"\n  💡 Для улучшения покрытия:")
                    print(f"     - Доска должна занимать 15-40% кадра")
                    print(f"     - Снимайте с разных расстояний")

                if "Сильная перспектива" in issue_types:
                    print(f"\n  💡 Для уменьшения перспективных искажений:")
                    print(f"     - Добавьте фронтальные снимки (доска параллельна камере)")
                    print(f"     - Сбалансируйте количество наклонных и фронтальных снимков")

    # Итоговые выводы
    print(f"\n{'='*80}")
    print(f"ВЫВОДЫ")
    print(f"{'='*80}\n")

    if len(cam0_results['good']) < 15 or len(cam1_results['good']) < 15:
        print(f"❌ НЕДОСТАТОЧНО хороших изображений для качественной калибровки!")
        print(f"   Необходимо минимум 20 хороших изображений для каждой камеры.")
        print(f"   Камера 0: {len(cam0_results['good'])} хороших")
        print(f"   Камера 1: {len(cam1_results['good'])} хороших")
    else:
        print(f"✓ Достаточно хороших изображений для калибровки")
        print(f"  Камера 0: {len(cam0_results['good'])} хороших изображений")
        print(f"  Камера 1: {len(cam1_results['good'])} хороших изображений")

    # Разница между камерами
    diff = abs(len(cam0_results['good']) - len(cam1_results['good']))
    if diff > 10:
        print(f"\n⚠ БОЛЬШАЯ разница в количестве хороших изображений между камерами ({diff})")
        print(f"   Это может привести к дисбалансу при стерео-калибровке.")

    print(f"\n{'='*80}\n")


def main():
    base_dir = "calibration/calibration_result"

    # Анализ камеры 0 (левая)
    cam0_dir = os.path.join(base_dir, "left")
    cam0_results = analyze_camera_set(cam0_dir, "Камера 0 (левая)")

    # Анализ камеры 1 (правая)
    cam1_dir = os.path.join(base_dir, "right")
    cam1_results = analyze_camera_set(cam1_dir, "Камера 1 (правая)")

    # Сравнение
    compare_cameras(cam0_results, cam1_results)

    # Сохранение отчета
    report_file = "calibration/calibration_analysis_report.json"
    save_report(cam0_results, cam1_results, report_file)

    # Рекомендации
    print_recommendations(cam0_results, cam1_results)


if __name__ == "__main__":
    main()
