#!/usr/bin/env python3
"""
Проверка синхронизации кадров в двух видео файлах
Анализирует timestamps первых кадров и общее количество кадров
"""

import sys
import subprocess
import json
import os

def get_first_frame_timestamp(video_file):
    """Получает timestamp первого кадра из видео файла"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time',
        '-of', 'json',
        '-read_intervals', '%+#1',  # Только первый кадр
        video_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"Ошибка ffprobe для {video_file}: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        if 'frames' in data and len(data['frames']) > 0:
            pts_time = data['frames'][0].get('pkt_pts_time')
            return float(pts_time) if pts_time else None
    except Exception as e:
        print(f"Ошибка при обработке {video_file}: {e}")
        return None

def get_frame_count(video_file):
    """Получает количество кадров в видео"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    return None

def get_video_info(video_file):
    """Получает базовую информацию о видео"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,r_frame_rate,codec_name',
        '-of', 'json',
        video_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                return data['streams'][0]
    except:
        pass
    return {}

def main():
    if len(sys.argv) < 3:
        print("Использование: python3 check_frame_sync.py master.mp4 slave.mp4")
        sys.exit(1)

    master_file = sys.argv[1]
    slave_file = sys.argv[2]

    # Проверяем существование файлов
    if not os.path.exists(master_file):
        print(f"❌ Файл не найден: {master_file}")
        sys.exit(1)

    if not os.path.exists(slave_file):
        print(f"❌ Файл не найден: {slave_file}")
        sys.exit(1)

    print("🔍 АНАЛИЗ СИНХРОНИЗАЦИИ ВИДЕО ФАЙЛОВ")
    print("=" * 60)
    print(f"Мастер: {master_file}")
    print(f"Слейв:  {slave_file}")
    print()

    # Получаем информацию о мастер файле
    print("[1/4] Анализ мастер файла...")
    master_info = get_video_info(master_file)
    master_first_pts = get_first_frame_timestamp(master_file)
    master_frame_count = get_frame_count(master_file)

    if master_first_pts is not None:
        print(f"  ✅ Первый кадр PTS: {master_first_pts:.6f} сек")
    else:
        print(f"  ⚠️  Не удалось получить PTS первого кадра")

    if master_frame_count:
        print(f"  ✅ Количество кадров: {master_frame_count}")

    if 'duration' in master_info:
        print(f"  ✅ Длительность: {float(master_info['duration']):.2f} сек")

    if 'r_frame_rate' in master_info:
        fps_parts = master_info['r_frame_rate'].split('/')
        fps = int(fps_parts[0]) / int(fps_parts[1])
        print(f"  ✅ FPS: {fps:.2f}")

    print()

    # Получаем информацию о слейв файле
    print("[2/4] Анализ слейв файла...")
    slave_info = get_video_info(slave_file)
    slave_first_pts = get_first_frame_timestamp(slave_file)
    slave_frame_count = get_frame_count(slave_file)

    if slave_first_pts is not None:
        print(f"  ✅ Первый кадр PTS: {slave_first_pts:.6f} сек")
    else:
        print(f"  ⚠️  Не удалось получить PTS первого кадра")

    if slave_frame_count:
        print(f"  ✅ Количество кадров: {slave_frame_count}")

    if 'duration' in slave_info:
        print(f"  ✅ Длительность: {float(slave_info['duration']):.2f} сек")

    if 'r_frame_rate' in slave_info:
        fps_parts = slave_info['r_frame_rate'].split('/')
        fps = int(fps_parts[0]) / int(fps_parts[1])
        print(f"  ✅ FPS: {fps:.2f}")

    print()

    # Анализ синхронизации
    print("[3/4] Анализ синхронизации...")

    # Разница в PTS первого кадра
    if master_first_pts is not None and slave_first_pts is not None:
        pts_diff_ms = abs(master_first_pts - slave_first_pts) * 1000

        # Определяем FPS для расчета разницы в кадрах
        fps = 30.0  # по умолчанию
        if 'r_frame_rate' in master_info:
            fps_parts = master_info['r_frame_rate'].split('/')
            fps = int(fps_parts[0]) / int(fps_parts[1])

        frame_diff = pts_diff_ms / (1000.0 / fps)

        print(f"  📊 Разница PTS первого кадра: {pts_diff_ms:.2f} мс")
        print(f"  📊 Разница в кадрах: ~{frame_diff:.2f} кадров @ {fps:.0f}fps")

        if frame_diff < 0.5:
            print(f"  ✅ ОТЛИЧНО! Практически идеальная синхронизация")
        elif frame_diff < 1.0:
            print(f"  ✅ ОТЛИЧНО! Разница меньше 1 кадра")
        elif frame_diff < 2.0:
            print(f"  ✅ ХОРОШО! Разница меньше 2 кадров")
        elif frame_diff < 5.0:
            print(f"  ⚠️  ПРИЕМЛЕМО. Разница меньше 5 кадров")
        else:
            print(f"  ❌ ПЛОХО! Разница больше 5 кадров")
    else:
        print(f"  ❌ Не удалось сравнить PTS")

    print()

    # Разница в количестве кадров
    if master_frame_count and slave_frame_count:
        frame_count_diff = abs(master_frame_count - slave_frame_count)
        print(f"  📊 Разница в количестве кадров: {frame_count_diff}")

        if frame_count_diff == 0:
            print(f"  ✅ ИДЕАЛЬНО! Одинаковое количество кадров")
        elif frame_count_diff <= 2:
            print(f"  ✅ ОТЛИЧНО! Разница не более 2 кадров")
        elif frame_count_diff <= 5:
            print(f"  ⚠️  ПРИЕМЛЕМО. Разница не более 5 кадров")
        else:
            print(f"  ❌ ПЛОХО! Большая разница в количестве кадров")

    print()

    # Итоговая оценка
    print("[4/4] Итоговая оценка синхронизации...")

    sync_quality = "НЕИЗВЕСТНО"

    if master_first_pts is not None and slave_first_pts is not None:
        pts_diff_ms = abs(master_first_pts - slave_first_pts) * 1000
        fps = 30.0
        if 'r_frame_rate' in master_info:
            fps_parts = master_info['r_frame_rate'].split('/')
            fps = int(fps_parts[0]) / int(fps_parts[1])
        frame_diff = pts_diff_ms / (1000.0 / fps)

        if frame_diff < 1.0:
            sync_quality = "ОТЛИЧНО (< 1 кадра)"
        elif frame_diff < 3.0:
            sync_quality = "ХОРОШО (< 3 кадров)"
        elif frame_diff < 5.0:
            sync_quality = "ПРИЕМЛЕМО (< 5 кадров)"
        else:
            sync_quality = "ПЛОХО (>= 5 кадров)"

    print(f"  🎯 Качество синхронизации: {sync_quality}")

    print()
    print("=" * 60)
    print("✅ Анализ завершен!")

    # Рекомендации
    if master_first_pts is not None and slave_first_pts is not None:
        frame_diff = abs(master_first_pts - slave_first_pts) * 1000 / (1000.0 / 30)

        if frame_diff >= 1.0:
            print()
            print("💡 Рекомендации для улучшения синхронизации:")

            if frame_diff >= 5.0:
                print("  • Проверьте физическое подключение sync провода")
                print("  • Убедитесь что V4L2 controls настроены (operation_mode, synchronizing_function)")
            elif frame_diff >= 3.0:
                print("  • Убедитесь что sync провод правильно подключен")
                print("  • Проверьте что обе камеры используют одинаковый sensor-mode")
            elif frame_diff >= 1.0:
                print("  • Результат хороший, но можно улучшить с hardware sync")
                print("  • Проверьте качество sync провода")

if __name__ == "__main__":
    main()
