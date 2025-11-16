#!/usr/bin/env python3
"""
Анализатор качества калибровочных изображений
Проверяет видимость доски, покрытие областей кадра, оценивает качество данных
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json

class CalibrationQualityChecker:
    def __init__(self, images_dir, board_size=(10, 7), camera_name="Camera"):
        """
        Инициализация проверки качества

        Args:
            images_dir: директория с изображениями
            board_size: размер доски (внутренние углы)
            camera_name: имя камеры для отчёта
        """
        self.images_dir = images_dir
        self.board_size = board_size
        self.camera_name = camera_name

        # Критерии для уточнения углов
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def analyze_board_position(self, corners, img_shape):
        """
        Анализирует позицию доски в кадре

        Returns:
            dict: информация о позиции и покрытии
        """
        h, w = img_shape[:2]
        corners_flat = corners.reshape(-1, 2)

        # Границы доски
        min_x, min_y = corners_flat.min(axis=0)
        max_x, max_y = corners_flat.max(axis=0)

        # Центр доски
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Покрытие кадра (в процентах)
        coverage_x = (max_x - min_x) / w * 100
        coverage_y = (max_y - min_y) / h * 100

        # Определяем зону (делим кадр на 9 зон: 3x3)
        zone_x = 1  # 0=left, 1=center, 2=right
        zone_y = 1  # 0=top, 1=middle, 2=bottom

        if center_x < w * 0.33:
            zone_x = 0
        elif center_x > w * 0.67:
            zone_x = 2

        if center_y < h * 0.33:
            zone_y = 0
        elif center_y > h * 0.67:
            zone_y = 2

        # Определяем категорию позиции
        zone_names_h = ["СЛЕВА", "ЦЕНТР", "СПРАВА"]
        zone_names_v = ["ВВЕРХУ", "ЦЕНТР", "ВНИЗУ"]

        pos_h = zone_names_h[zone_x]
        pos_v = zone_names_v[zone_y]

        if pos_v == "ЦЕНТР" and pos_h == "ЦЕНТР":
            position = "ЦЕНТР"
        else:
            position = f"{pos_v}-{pos_h}" if pos_v != "ЦЕНТР" or pos_h != "ЦЕНТР" else pos_v if pos_v != "ЦЕНТР" else pos_h

        # Определяем, является ли позиция угловой
        is_corner = zone_x in [0, 2] and zone_y in [0, 2]
        is_edge = (zone_x in [0, 2] or zone_y in [0, 2]) and not is_corner
        is_center = zone_x == 1 and zone_y == 1

        # Расстояние (оценка на основе размера доски в кадре)
        avg_coverage = (coverage_x + coverage_y) / 2
        if avg_coverage < 20:
            distance = "ДАЛЕКО"
        elif avg_coverage > 50:
            distance = "БЛИЗКО"
        else:
            distance = "СРЕДНЕ"

        return {
            'position': position,
            'zone_x': zone_x,
            'zone_y': zone_y,
            'is_corner': is_corner,
            'is_edge': is_edge,
            'is_center': is_center,
            'coverage_x': coverage_x,
            'coverage_y': coverage_y,
            'avg_coverage': avg_coverage,
            'distance': distance,
            'center_x': center_x / w,  # нормализованная координата
            'center_y': center_y / h
        }

    def check_image(self, img_path):
        """
        Проверяет одно изображение

        Returns:
            dict: результаты проверки или None если доска не найдена
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Поиск углов с адаптивным порогом
            ret, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if not ret:
                return None

            # Уточняем углы с субпиксельной точностью
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # Анализируем позицию
            position_info = self.analyze_board_position(corners, img.shape)

            # Оцениваем качество углов (резкость)
            sharpness_scores = []
            for corner in corners:
                x, y = corner[0]
                x, y = int(x), int(y)
                # Берем окно 11x11 вокруг угла
                window = gray[max(0, y-5):min(gray.shape[0], y+6),
                             max(0, x-5):min(gray.shape[1], x+6)]
                if window.size > 0:
                    # Вычисляем Лапласиан (мера резкости)
                    laplacian = cv2.Laplacian(window, cv2.CV_64F)
                    sharpness = laplacian.var()
                    sharpness_scores.append(sharpness)

            avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0

            # Оценка качества
            if avg_sharpness > 100:
                quality = "ОТЛИЧНО"
            elif avg_sharpness > 50:
                quality = "ХОРОШО"
            elif avg_sharpness > 20:
                quality = "УДОВЛЕТВОРИТЕЛЬНО"
            else:
                quality = "ПЛОХО (размыто)"

            return {
                'filename': os.path.basename(img_path),
                'found': True,
                'corners': corners,
                'position_info': position_info,
                'sharpness': avg_sharpness,
                'quality': quality
            }

        except Exception as e:
            print(f"[ERROR] Ошибка обработки {img_path}: {e}")
            return None

    def analyze_directory(self):
        """
        Анализирует все изображения в директории

        Returns:
            dict: полный отчёт
        """
        print(f"\n{'='*70}")
        print(f"АНАЛИЗ КАЧЕСТВА КАЛИБРОВОЧНЫХ ИЗОБРАЖЕНИЙ: {self.camera_name}")
        print(f"{'='*70}")
        print(f"📁 Директория: {self.images_dir}")
        print(f"🎯 Доска: {self.board_size[0]}x{self.board_size[1]} углов")

        # Получаем список изображений
        images = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")) +
                       glob.glob(os.path.join(self.images_dir, "*.png")))

        if not images:
            print(f"\n❌ Изображения не найдены!")
            return None

        print(f"📷 Найдено изображений: {len(images)}")
        print(f"\n{'='*70}")
        print("ОБРАБОТКА...")
        print(f"{'='*70}\n")

        # Результаты
        results = {
            'total': len(images),
            'found': 0,
            'not_found': 0,
            'images': [],
            'not_found_files': [],
            'zone_coverage': {
                # 9 зон: комбинации (zone_y, zone_x)
                (0, 0): 0,  # Верхний-левый
                (0, 1): 0,  # Верхний-центр
                (0, 2): 0,  # Верхний-правый
                (1, 0): 0,  # Средний-левый
                (1, 1): 0,  # Центр
                (1, 2): 0,  # Средний-правый
                (2, 0): 0,  # Нижний-левый
                (2, 1): 0,  # Нижний-центр
                (2, 2): 0,  # Нижний-правый
            },
            'distance_coverage': {
                'БЛИЗКО': 0,
                'СРЕДНЕ': 0,
                'ДАЛЕКО': 0
            }
        }

        # Обрабатываем все изображения
        for i, img_path in enumerate(images):
            print(f"\r  [{i+1}/{len(images)}] {os.path.basename(img_path)[:40]:<40}", end='', flush=True)

            result = self.check_image(img_path)

            if result and result['found']:
                results['found'] += 1
                results['images'].append(result)

                # Подсчет покрытия зон
                pos_info = result['position_info']
                zone_key = (pos_info['zone_y'], pos_info['zone_x'])
                results['zone_coverage'][zone_key] += 1

                # Подсчет покрытия расстояний
                results['distance_coverage'][pos_info['distance']] += 1
            else:
                results['not_found'] += 1
                results['not_found_files'].append(os.path.basename(img_path))

        print(f"\n\n{'='*70}")
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")
        print(f"{'='*70}\n")

        # Общая статистика
        print(f"📊 Общая статистика:")
        print(f"   • Всего изображений: {results['total']}")
        print(f"   • Доска найдена: {results['found']} ({results['found']/results['total']*100:.1f}%)")
        print(f"   • Доска НЕ найдена: {results['not_found']} ({results['not_found']/results['total']*100:.1f}%)")

        if results['not_found'] > 0:
            print(f"\n❌ Изображения БЕЗ доски:")
            for fname in results['not_found_files'][:10]:
                print(f"      • {fname}")
            if len(results['not_found_files']) > 10:
                print(f"      ... и ещё {len(results['not_found_files']) - 10}")

        if results['found'] == 0:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Ни на одном изображении не найдена доска!")
            return results

        # Анализ качества изображений
        print(f"\n📐 Качество изображений:")
        sharpness_values = [img['sharpness'] for img in results['images']]
        quality_counts = {}
        for img in results['images']:
            q = img['quality']
            quality_counts[q] = quality_counts.get(q, 0) + 1

        print(f"   • Средняя резкость: {np.mean(sharpness_values):.1f}")
        print(f"   • Мин/Макс резкость: {np.min(sharpness_values):.1f} / {np.max(sharpness_values):.1f}")
        for quality, count in sorted(quality_counts.items()):
            print(f"   • {quality}: {count} изображений")

        # Карта покрытия (9 зон)
        print(f"\n🗺️  КАРТА ПОКРЫТИЯ КАДРА (9 зон):")
        print(f"   ┌──────────────┬──────────────┬──────────────┐")
        zone_names = [
            ["Верх-Лево", "Верх-Центр", "Верх-Право"],
            ["Лево", "ЦЕНТР", "Право"],
            ["Низ-Лево", "Низ-Центр", "Низ-Право"]
        ]

        for row in range(3):
            zone_counts = []
            for col in range(3):
                count = results['zone_coverage'][(row, col)]
                zone_counts.append(count)

            # Форматирование строки
            row_str = "   │"
            for col, count in enumerate(zone_counts):
                zone_name = zone_names[row][col]
                # Цветовая индикация
                if count == 0:
                    indicator = "❌"
                elif count < 3:
                    indicator = "⚠️"
                else:
                    indicator = "✅"

                row_str += f" {indicator} {count:2d} {zone_name:<8} │"

            print(row_str)
            if row < 2:
                print(f"   ├──────────────┼──────────────┼──────────────┤")

        print(f"   └──────────────┴──────────────┴──────────────┘")

        # Покрытие по расстоянию
        print(f"\n📏 Распределение по РАССТОЯНИЮ:")
        for dist, count in sorted(results['distance_coverage'].items()):
            percent = count / results['found'] * 100 if results['found'] > 0 else 0
            bar = "█" * int(percent / 5)
            print(f"   • {dist:<8}: {count:2d} ({percent:5.1f}%) {bar}")

        # ОЦЕНКА И РЕКОМЕНДАЦИИ
        print(f"\n{'='*70}")
        print("ОЦЕНКА И РЕКОМЕНДАЦИИ")
        print(f"{'='*70}\n")

        issues = []
        recommendations = []

        # Проверка общего количества
        if results['found'] < 15:
            issues.append("❌ Недостаточно изображений с доской")
            recommendations.append("• Сделайте минимум 20 изображений с видимой доской")
        elif results['found'] < 20:
            issues.append("⚠️  Мало изображений для оптимальной калибровки")
            recommendations.append("• Рекомендуется 25-30 изображений для лучшей точности")
        else:
            print("✅ Достаточно изображений для калибровки")

        # Проверка покрытия углов (важно!)
        corner_zones = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_counts = [results['zone_coverage'][zone] for zone in corner_zones]
        missing_corners = sum(1 for c in corner_counts if c == 0)
        weak_corners = sum(1 for c in corner_counts if 0 < c < 2)

        if missing_corners > 0:
            issues.append(f"❌ КРИТИЧНО: Не покрыто {missing_corners} углов кадра")
            corner_names = ["Верх-Лево", "Верх-Право", "Низ-Лево", "Низ-Право"]
            missing = [corner_names[i] for i, c in enumerate(corner_counts) if c == 0]
            recommendations.append(f"• ОБЯЗАТЕЛЬНО сделайте фото в углах: {', '.join(missing)}")
        elif weak_corners > 0:
            issues.append(f"⚠️  Слабое покрытие {weak_corners} углов (меньше 2 изображений)")
            recommendations.append("• Добавьте минимум 2 изображения в каждом углу")
        else:
            print("✅ Все углы кадра покрыты")

        # Проверка покрытия краёв
        edge_zones = [(0, 1), (1, 0), (1, 2), (2, 1)]
        edge_counts = [results['zone_coverage'][zone] for zone in edge_zones]
        missing_edges = sum(1 for c in edge_counts if c == 0)

        if missing_edges > 0:
            issues.append(f"⚠️  Не покрыто {missing_edges} краев кадра")
            edge_names = ["Верх", "Лево", "Право", "Низ"]
            missing = [edge_names[i] for i, c in enumerate(edge_counts) if c == 0]
            recommendations.append(f"• Сделайте фото по краям: {', '.join(missing)}")
        else:
            print("✅ Края кадра покрыты")

        # Проверка центра
        center_count = results['zone_coverage'][(1, 1)]
        if center_count == 0:
            issues.append("❌ Нет изображений из центра кадра")
            recommendations.append("• Сделайте 3-5 изображений с доской в центре")
        elif center_count < 3:
            issues.append("⚠️  Мало изображений из центра")
            recommendations.append("• Добавьте ещё 2-3 изображения с доской в центре")
        else:
            print("✅ Центр кадра хорошо покрыт")

        # Проверка разнообразия расстояний
        dist_counts = list(results['distance_coverage'].values())
        if any(c == 0 for c in dist_counts):
            issues.append("⚠️  Не хватает разнообразия расстояний")
            if results['distance_coverage']['БЛИЗКО'] == 0:
                recommendations.append("• Сделайте несколько фото с доской БЛИЗКО к камере")
            if results['distance_coverage']['ДАЛЕКО'] == 0:
                recommendations.append("• Сделайте несколько фото с доской ДАЛЕКО от камеры")

        # Проверка качества (резкость)
        blurry_count = sum(1 for img in results['images'] if img['sharpness'] < 20)
        if blurry_count > results['found'] * 0.2:  # Больше 20% размытых
            issues.append(f"⚠️  Много размытых изображений ({blurry_count})")
            recommendations.append("• Держите камеру неподвижно во время съёмки")
            recommendations.append("• Убедитесь, что освещение достаточное")

        # Вывод проблем
        if issues:
            print(f"\n⚠️  ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
            for issue in issues:
                print(f"   {issue}")

        # Вывод рекомендаций
        if recommendations:
            print(f"\n💡 РЕКОМЕНДАЦИИ:")
            for rec in recommendations:
                print(f"   {rec}")

        # Итоговая оценка
        print(f"\n{'='*70}")
        if not issues:
            print("🎉 ОТЛИЧНО! Данные готовы для калибровки!")
            print("   Можно переходить к следующей камере или к стерео-калибровке")
        elif missing_corners > 0 or results['found'] < 15:
            print("❌ НЕДОСТАТОЧНО ДАННЫХ для качественной калибровки")
            print("   Необходимо сделать дополнительные фотографии")
        else:
            print("⚠️  Данные приемлемы, но рекомендуется улучшить покрытие")
            print("   Калибровка возможна, но точность может быть ниже оптимальной")

        print(f"{'='*70}\n")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Анализ качества калибровочных изображений')
    parser.add_argument('directory', help='Директория с изображениями')
    parser.add_argument('--board-cols', type=int, default=10, help='Количество углов по горизонтали')
    parser.add_argument('--board-rows', type=int, default=7, help='Количество углов по вертикали')
    parser.add_argument('--camera-name', default='Camera', help='Имя камеры для отчёта')

    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"❌ Директория не найдена: {args.directory}")
        return

    checker = CalibrationQualityChecker(
        images_dir=args.directory,
        board_size=(args.board_cols, args.board_rows),
        camera_name=args.camera_name
    )

    results = checker.analyze_directory()

    # Сохраняем отчёт в JSON
    if results:
        report_file = os.path.join(args.directory, "quality_report.json")
        # Упрощаем данные для JSON (убираем numpy массивы)
        json_results = {
            'total': results['total'],
            'found': results['found'],
            'not_found': results['not_found'],
            'not_found_files': results['not_found_files'],
            'zone_coverage': {f"{k[0]},{k[1]}": v for k, v in results['zone_coverage'].items()},
            'distance_coverage': results['distance_coverage']
        }
        with open(report_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"📄 Отчёт сохранён: {report_file}")


if __name__ == "__main__":
    main()
