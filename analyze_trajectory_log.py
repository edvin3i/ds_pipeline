#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор логов траектории - создает упорядоченный лог с группировкой по временным блокам.

Преобразует запутанный лог в чистую структуру с автоматическим обнаружением разрывов.
"""

import re
from collections import defaultdict


def parse_trajectory_log(log_file):
    """Парсит лог траектории и извлекает блоки."""
    blocks = []

    # Открываем с явным указанием кодировки (важно для эмодзи)
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ищем все блоки: начинаются с "TRAJECTORY DEBUG LOG" и содержат "Time span"
    # Разбиваем по блокам через "🔴 LARGE GAPS ANALYSIS" или следующему блоку
    pattern = r'TRAJECTORY DEBUG LOG:.*?Time span: \[([\d.]+), ([\d.]+)\].*?Source breakdown: ({[^}]+})'

    seen_time_spans = set()
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        source_breakdown = match.group(3)

        # Пропускаем дублирующиеся блоки
        key = (round(start_time, 2), round(end_time, 2))
        if key in seen_time_spans:
            continue

        seen_time_spans.add(key)

        # Извлекаем всё содержимое между "Source breakdown" и следующим "TRAJECTORY" или "🔴 LARGE GAPS"
        start_pos = match.end()
        rest_of_content = content[start_pos:]

        # Ищем конец блока (следующий "TRAJECTORY" или "LARGE GAPS")
        end_match = re.search(r'(TRAJECTORY DEBUG LOG|🔴 LARGE GAPS ANALYSIS)', rest_of_content)
        if end_match:
            block_content = rest_of_content[:end_match.start()]
        else:
            block_content = rest_of_content

        # Извлекаем строки с данными (начинаются с времени типа "   9.48s")
        data_lines = []
        for line in block_content.split('\n'):
            if re.match(r'^\s*[\d.]+s', line):
                data_lines.append(line.rstrip())

        # Добавляем блок только если есть данные
        if data_lines:
            block = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'source_breakdown': source_breakdown,
                'data_lines': data_lines
            }
            blocks.append(block)

    return blocks


def create_structured_log(blocks, output_file):
    """Создает структурированный лог."""

    if not blocks:
        print("❌ Блоков не найдено!")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n" + "=" * 150 + "\n")
        f.write("📊 СТРУКТУРИРОВАННЫЙ АНАЛИЗ ТРАЕКТОРИИ КАМЕРЫ\n")
        f.write("=" * 150 + "\n")
        f.write(f"Всего блоков (уникальных): {len(blocks)}\n")
        f.write(f"Временной диапазон: [{blocks[0]['start_time']:.2f}s, {blocks[-1]['end_time']:.2f}s]\n")
        f.write(f"Общая длительность: {blocks[-1]['end_time'] - blocks[0]['start_time']:.2f}s\n")
        f.write("=" * 150 + "\n\n")

        # Выводим каждый блок
        for idx, block in enumerate(blocks):
            # Заголовок блока
            duration = block['end_time'] - block['start_time']
            f.write(f"\n{'─' * 150}\n")
            f.write(f"📍 БЛОК #{idx + 1:3d} | Время: [{block['start_time']:.2f}s → {block['end_time']:.2f}s] ({duration:.2f}s)\n")
            f.write(f"{'─' * 150}\n")
            f.write(f"Состав: {block['source_breakdown']}\n")
            f.write(f"{'─' * 150}\n")

            # Выводим данные таблицы
            if block['data_lines']:
                for line in block['data_lines']:
                    f.write(line + "\n")

            # Анализируем содержимое блока
            analyze_block_content(block, f)

            # Проверяем разрыв до следующего блока
            if idx < len(blocks) - 1:
                next_block = blocks[idx + 1]
                gap = next_block['start_time'] - block['end_time']

                if gap > 0.01:  # Есть разрыв
                    f.write(f"\n⚠️  РАЗРЫВ МЕЖДУ БЛОКАМИ: {gap:.2f}s\n")
                    f.write(f"   От {block['end_time']:.2f}s до {next_block['start_time']:.2f}s\n")
                    if gap > 3.0:
                        f.write(f"   🔴 ВНИМАНИЕ: Большой разрыв > 3.0s!\n")
                    if gap > 10.0:
                        f.write(f"   🔴🔴 КРИТИЧНО: Очень большой разрыв > 10s (вероятно, потеря мяча)\n")

        # ИТОГОВЫЙ АНАЛИЗ
        write_summary(blocks, f)


def analyze_block_content(block, f):
    """Анализирует содержимое блока."""
    f.write("\n📋 Анализ блока:\n")

    # Подсчет типов
    types = defaultdict(int)
    for line in block['data_lines']:
        if 'BALL' in line:
            types['BALL'] += 1
        elif 'PLAYER_COM' in line:
            types['PLAYER_COM'] += 1
        elif 'BLEND' in line:
            types['BLEND'] += 1
        elif 'PLAYER_ONLY' in line:
            types['PLAYER_ONLY'] += 1
        elif 'INTERP' in line:
            types['INTERP'] += 1

    if types:
        summary = " + ".join([f"{k}({v})" for k, v in sorted(types.items())])
        f.write(f"  Состав: {summary}\n")

        # Определяем фазу
        if types.get('BALL', 0) > types.get('PLAYER_COM', 0) + types.get('PLAYER_ONLY', 0):
            f.write("  🎾 ФАЗА: Мяч летит (обнаружен YOLO)\n")
        elif types.get('PLAYER_COM', 0) > 0:
            f.write("  👥 ФАЗА: Мяч потерян, следим за игроками\n")
            if types.get('BLEND', 0) > 0:
                f.write("  🔄 Включает плавный переход (BLEND)\n")
        elif types.get('PLAYER_ONLY', 0) > 0:
            f.write("  👥 ФАЗА: Только игроки (мяч не был обнаружен)\n")


def write_summary(blocks, f):
    """Пишет итоговый анализ."""

    f.write("\n\n" + "=" * 150 + "\n")
    f.write("📈 ИТОГОВЫЙ АНАЛИЗ\n")
    f.write("=" * 150 + "\n\n")

    # Статистика по типам
    type_counts = defaultdict(int)
    phase_stats = {
        'ball_phase': 0,
        'player_phase': 0,
        'player_only_phase': 0,
        'transition': 0
    }

    for block in blocks:
        for line in block['data_lines']:
            if 'BALL' in line and 'INTERP' not in line:
                type_counts['BALL'] += 1
            elif 'PLAYER_COM' in line:
                type_counts['PLAYER_COM'] += 1
            elif 'BLEND' in line:
                type_counts['BLEND'] += 1
                phase_stats['transition'] += 1
            elif 'PLAYER_ONLY' in line:
                type_counts['PLAYER_ONLY'] += 1
                phase_stats['player_only_phase'] += 1
            elif 'INTERP' in line:
                type_counts['INTERP'] += 1

        # Определяем фазу блока
        types_in_block = defaultdict(int)
        for line in block['data_lines']:
            if 'BALL' in line:
                types_in_block['BALL'] += 1
            elif 'PLAYER_COM' in line or 'PLAYER_ONLY' in line:
                types_in_block['PLAYER'] += 1

        if types_in_block.get('BALL', 0) > 0:
            phase_stats['ball_phase'] += block['duration']
        else:
            phase_stats['player_phase'] += block['duration']

    f.write("Статистика по типам точек:\n")
    for point_type, count in sorted(type_counts.items()):
        f.write(f"  {point_type:<15} : {count:6d} точек\n")

    f.write("\nДлительность фаз:\n")
    total_duration = blocks[-1]['end_time'] - blocks[0]['start_time']
    f.write(f"  Мяч видим    : {phase_stats['ball_phase']:7.2f}s ({phase_stats['ball_phase']/total_duration*100:5.1f}%)\n")
    f.write(f"  Мяч потерян  : {phase_stats['player_phase']:7.2f}s ({phase_stats['player_phase']/total_duration*100:5.1f}%)\n")
    f.write(f"  Переходы     : {phase_stats['transition']:3d} блоков\n")

    # Анализ разрывов между блоками
    gaps = []
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1]['start_time'] - blocks[i]['end_time']
        if gap > 0.01:
            gaps.append({
                'blocks': f"{i + 1}-{i + 2}",
                'gap': gap,
                'from': blocks[i]['end_time'],
                'to': blocks[i + 1]['start_time']
            })

    if gaps:
        f.write(f"\n⚠️  Разрывы между блоками (найдено {len(gaps)}):\n")
        for gap_info in gaps:
            if gap_info['gap'] > 10.0:
                marker = "🔴🔴"
            elif gap_info['gap'] > 3.0:
                marker = "🔴"
            else:
                marker = "⚠️"
            f.write(f"  {marker} Блоки {gap_info['blocks']}: {gap_info['gap']:.2f}s разрыва\n")
    else:
        f.write("\n✅ Разрывов между блоками не обнаружено\n")

    # Долгие блоки
    f.write("\n⏱️  Самые длинные блоки:\n")
    sorted_blocks = sorted(enumerate(blocks), key=lambda x: x[1]['duration'], reverse=True)
    for rank, (idx, block) in enumerate(sorted_blocks[:5]):
        marker = "🔴" if block['duration'] > 3.0 else "✅"
        f.write(f"  {rank + 1}. Блок #{idx + 1}: {block['duration']:.2f}s {marker}\n")

    f.write("\n" + "=" * 150 + "\n")


if __name__ == '__main__':
    log_file = '/tmp/camera_trajectory_debug.log'
    output_file = '/tmp/camera_trajectory_structured.log'

    print("📊 Парсирование лога траектории...")
    blocks = parse_trajectory_log(log_file)
    print(f"✅ Найдено {len(blocks)} уникальных блоков")

    if blocks:
        print("📝 Создание структурированного лога...")
        create_structured_log(blocks, output_file)
        print(f"✅ Лог сохранен в {output_file}")
        print(f"\n📋 Превью первых {min(5, len(blocks))} блоков:")
        for i, block in enumerate(blocks[:5]):
            print(f"  Блок #{i + 1}: [{block['start_time']:.2f}s - {block['end_time']:.2f}s] ({block['duration']:.2f}s) - {block['source_breakdown']}")
        print(f"\n📋 Превью последних блоков (с большим разрывом 45.51s → 70.54s):")
        for i, block in enumerate(blocks[-3:]):
            idx = len(blocks) - 3 + i
            print(f"  Блок #{idx + 1}: [{block['start_time']:.2f}s - {block['end_time']:.2f}s] ({block['duration']:.2f}s) - {block['source_breakdown']}")
    else:
        print("❌ Блоков не найдено в логе!")
