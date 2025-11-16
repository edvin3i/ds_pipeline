#!/usr/bin/env python3
"""
Тест диапазона движения виртуальной камеры
Автоматически двигает мяч по всей панораме и проверяет границы
"""

import sys
import os

# Параметры панорамы
LON_MIN = -90.0
LON_MAX = +90.0
LAT_MIN = -32.0
LAT_MAX = +22.0

PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

FOV_MIN = 40.0
FOV_MAX = 68.0
ASPECT_RATIO = 16.0 / 9.0

def calculate_pitch_from_y(y):
    """Расчет pitch по формуле из C++ кода"""
    norm_y = y / (PANORAMA_HEIGHT - 1)
    pitch = LAT_MAX - norm_y * (LAT_MAX - LAT_MIN)
    return pitch

def calculate_yaw_from_x(x):
    """Расчет yaw по формуле из C++ кода"""
    norm_x = x / (PANORAMA_WIDTH - 1)
    yaw = LON_MIN + norm_x * (LON_MAX - LON_MIN)
    return yaw

print("=" * 80)
print("ТЕСТ ДИАПАЗОНА ДВИЖЕНИЯ ВИРТУАЛЬНОЙ КАМЕРЫ")
print("=" * 80)

print(f"\n📐 Параметры панорамы:")
print(f"   Размер: {PANORAMA_WIDTH} × {PANORAMA_HEIGHT} пикселей")
print(f"   Горизонталь (YAW): {LON_MIN}° до {LON_MAX}° (покрытие: {LON_MAX - LON_MIN}°)")
print(f"   Вертикаль (PITCH): {LAT_MIN}° до {LAT_MAX}° (покрытие: {LAT_MAX - LAT_MIN}°)")

print(f"\n🎥 FOV диапазон: {FOV_MIN}° - {FOV_MAX}°")
print(f"   Aspect ratio: {ASPECT_RATIO:.2f} (16:9)")

print("\n" + "=" * 80)
print("ТЕСТ 1: ВЕРТИКАЛЬНОЕ ДВИЖЕНИЕ (PITCH)")
print("=" * 80)

test_y_positions = [
    (0, "Верх изображения"),
    (475, "Верхняя четверть"),
    (950, "Центр изображения"),
    (1425, "Нижняя четверть"),
    (1800, "Почти низ"),
    (1899, "Низ изображения"),
]

print(f"\n{'Y позиция':<15} {'Описание':<25} {'Pitch':<10} {'Ожидаемое поведение':<30}")
print("-" * 90)

for y, description in test_y_positions:
    pitch = calculate_pitch_from_y(y)

    # Определяем поведение
    if pitch > LAT_MAX:
        behavior = "⚠️ ЗА ГРАНИЦЕЙ! (выше панорамы)"
    elif pitch < LAT_MIN:
        behavior = "⚠️ ЗА ГРАНИЦЕЙ! (ниже панорамы)"
    elif pitch > 10:
        behavior = "✅ Смотрит вверх"
    elif pitch > -10:
        behavior = "✅ Центральная зона"
    else:
        behavior = "✅ Смотрит вниз"

    print(f"{y:<15} {description:<25} {pitch:>6.1f}°   {behavior:<30}")

print("\n" + "=" * 80)
print("ТЕСТ 2: ГОРИЗОНТАЛЬНОЕ ДВИЖЕНИЕ (YAW)")
print("=" * 80)

test_x_positions = [
    (0, "Левый край"),
    (1425, "Левая четверть"),
    (2850, "Центр изображения"),
    (4275, "Правая четверть"),
    (5699, "Правый край"),
]

print(f"\n{'X позиция':<15} {'Описание':<25} {'Yaw':<10} {'Ожидаемое поведение':<30}")
print("-" * 90)

for x, description in test_x_positions:
    yaw = calculate_yaw_from_x(x)

    # Определяем поведение
    if yaw < LON_MIN:
        behavior = "⚠️ ЗА ГРАНИЦЕЙ! (левее панорамы)"
    elif yaw > LON_MAX:
        behavior = "⚠️ ЗА ГРАНИЦЕЙ! (правее панорамы)"
    elif yaw < -60:
        behavior = "✅ Смотрит влево"
    elif yaw < -20:
        behavior = "✅ Левее центра"
    elif yaw < 20:
        behavior = "✅ Центральная зона"
    elif yaw < 60:
        behavior = "✅ Правее центра"
    else:
        behavior = "✅ Смотрит вправо"

    print(f"{x:<15} {description:<25} {yaw:>6.1f}°   {behavior:<30}")

print("\n" + "=" * 80)
print("ТЕСТ 3: ПРОВЕРКА ГРАНИЦ ПРИ РАЗНЫХ FOV")
print("=" * 80)

for fov in [FOV_MIN, 50.0, 60.0, FOV_MAX]:
    half_fov = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    # Безопасные границы для центра камеры
    pitch_min_safe = LAT_MIN + half_fov
    pitch_max_safe = LAT_MAX - half_fov
    yaw_min_safe = LON_MIN + half_fov_h
    yaw_max_safe = LON_MAX - half_fov_h

    freedom_v = pitch_max_safe - pitch_min_safe
    freedom_h = yaw_max_safe - yaw_min_safe

    print(f"\n🎯 FOV = {fov}°:")
    print(f"   Вертикальный half_fov = {half_fov:.1f}°")
    print(f"   Безопасные границы pitch: {pitch_min_safe:.1f}° до {pitch_max_safe:.1f}° (свобода: {freedom_v:.1f}°)")

    if freedom_v < 0:
        print(f"   ⚠️ ПРОБЛЕМА: FOV больше высоты панорамы! Камера застрянет!")
        print(f"      Решение: Расширить EFFECTIVE_LAT_MIN/MAX на {abs(freedom_v/2):.1f}° с каждой стороны")
    elif freedom_v < 5:
        print(f"   ⚠️ ВНИМАНИЕ: Очень мало свободы по вертикали!")
    else:
        print(f"   ✅ Достаточно свободы для движения")

    print(f"\n   Горизонтальный half_fov = {half_fov_h:.1f}°")
    print(f"   Безопасные границы yaw: {yaw_min_safe:.1f}° до {yaw_max_safe:.1f}° (свобода: {freedom_h:.1f}°)")

    if freedom_h < 0:
        print(f"   ⚠️ ПРОБЛЕМА: FOV больше ширины панорамы! Камера застрянет!")
        print(f"      Решение: Расширить EFFECTIVE_LON_MIN/MAX на {abs(freedom_h/2):.1f}° с каждой стороны")
    elif freedom_h < 20:
        print(f"   ⚠️ ВНИМАНИЕ: Мало свободы по горизонтали!")
    else:
        print(f"   ✅ Достаточно свободы для движения")

print("\n" + "=" * 80)
print("ТЕСТ 4: ПРОВЕРКА ТЕКУЩИХ ГРАНИЦ В C++ КОДЕ")
print("=" * 80)

# Читаем текущие границы из исходников
try:
    cpp_file = "/home/nvidia/deep_cv_football/my_virt_cam/src/gstnvdsvirtualcam.cpp"

    with open(cpp_file, 'r') as f:
        content = f.read()

    # Ищем EFFECTIVE_LAT_MIN и EFFECTIVE_LAT_MAX (без _BASE)
    import re

    lat_min_match = re.search(r'const gfloat EFFECTIVE_LAT_MIN\s*=\s*([-+]?\d+\.?\d*)f?', content)
    lat_max_match = re.search(r'const gfloat EFFECTIVE_LAT_MAX\s*=\s*([-+]?\d+\.?\d*)f?', content)
    lon_min_match = re.search(r'const gfloat EFFECTIVE_LON_MIN\s*=\s*([-+]?\d+\.?\d*)f?', content)
    lon_max_match = re.search(r'const gfloat EFFECTIVE_LON_MAX\s*=\s*([-+]?\d+\.?\d*)f?', content)

    if lat_min_match and lat_max_match and lon_min_match and lon_max_match:
        eff_lat_min = float(lat_min_match.group(1))
        eff_lat_max = float(lat_max_match.group(1))
        eff_lon_min = float(lon_min_match.group(1))
        eff_lon_max = float(lon_max_match.group(1))

        print(f"\n📄 Текущие границы в gstnvdsvirtualcam.cpp:")
        print(f"   EFFECTIVE_LAT_MIN_BASE = {eff_lat_min}°  (панорама: {LAT_MIN}°)")
        print(f"   EFFECTIVE_LAT_MAX_BASE = {eff_lat_max}°  (панорама: {LAT_MAX}°)")
        print(f"   EFFECTIVE_LON_MIN = {eff_lon_min}°  (панорама: {LON_MIN}°)")
        print(f"   EFFECTIVE_LON_MAX = {eff_lon_max}°  (панорама: {LON_MAX}°)")

        # Проверяем расширение
        lat_min_ext = LAT_MIN - eff_lat_min
        lat_max_ext = eff_lat_max - LAT_MAX
        lon_min_ext = LON_MIN - eff_lon_min
        lon_max_ext = eff_lon_max - LON_MAX

        print(f"\n📊 Расширение относительно панорамы:")
        print(f"   Вниз: {lat_min_ext:+.1f}°  {'✅ Расширено' if lat_min_ext < 0 else '⚠️ Не расширено'}")
        print(f"   Вверх: {lat_max_ext:+.1f}°  {'✅ Расширено' if lat_max_ext > 0 else '⚠️ Не расширено'}")
        print(f"   Влево: {lon_min_ext:+.1f}°  {'✅ Расширено' if lon_min_ext < 0 else '⚠️ Не расширено'}")
        print(f"   Вправо: {lon_max_ext:+.1f}°  {'✅ Расширено' if lon_max_ext > 0 else '⚠️ Не расширено'}")

        # Проверяем достаточно ли расширения для FOV_MAX
        half_fov_max = FOV_MAX / 2.0
        half_fov_h_max = (FOV_MAX * ASPECT_RATIO) / 2.0

        eff_height = eff_lat_max - eff_lat_min
        eff_width = eff_lon_max - eff_lon_min

        freedom_v_max = eff_height - FOV_MAX
        freedom_h_max = eff_width - (FOV_MAX * ASPECT_RATIO)

        print(f"\n🎯 Проверка для FOV_MAX = {FOV_MAX}°:")
        print(f"   Эффективная высота: {eff_height:.1f}° (FOV: {FOV_MAX}°) → свобода: {freedom_v_max:.1f}°")
        if freedom_v_max >= 0:
            print(f"   ✅ Достаточно места по вертикали (±{freedom_v_max/2:.1f}°)")
        else:
            print(f"   ⚠️ НЕ ХВАТАЕТ места! Нужно расширить на {abs(freedom_v_max/2):.1f}° с каждой стороны")

        print(f"\n   Эффективная ширина: {eff_width:.1f}° (FOV_h: {FOV_MAX * ASPECT_RATIO:.1f}°) → свобода: {freedom_h_max:.1f}°")
        if freedom_h_max >= 0:
            print(f"   ✅ Достаточно места по горизонтали (±{freedom_h_max/2:.1f}°)")
        else:
            print(f"   ⚠️ НЕ ХВАТАЕТ места! Нужно расширить на {abs(freedom_h_max/2):.1f}° с каждой стороны")
    else:
        print("⚠️ Не удалось найти границы в файле")

except Exception as e:
    print(f"⚠️ Ошибка чтения файла: {e}")

print("\n" + "=" * 80)
print("ИТОГОВАЯ РЕКОМЕНДАЦИЯ")
print("=" * 80)

print(f"""
При FOV={FOV_MAX}° (максимум):
- Вертикальный FOV: {FOV_MAX}° (панорама: {LAT_MAX - LAT_MIN}°)
- Горизонтальный FOV: {FOV_MAX * ASPECT_RATIO:.1f}° (панорама: {LON_MAX - LON_MIN}°)

{'⚠️ FOV больше высоты панорамы!' if FOV_MAX > (LAT_MAX - LAT_MIN) else '✅ FOV помещается в панораму'}
{'⚠️ Горизонтальный FOV больше ширины панорамы!' if (FOV_MAX * ASPECT_RATIO) > (LON_MAX - LON_MIN) else '✅ Горизонтальный FOV помещается в панораму'}

Для обеспечения свободы движения при FOV={FOV_MAX}° рекомендуется:
""")

# Расчет рекомендуемых границ
if FOV_MAX > (LAT_MAX - LAT_MIN):
    needed_ext_v = (FOV_MAX - (LAT_MAX - LAT_MIN)) / 2 + 2  # +2° запас
    print(f"✅ Расширить вертикальные границы на {needed_ext_v:.0f}° с каждой стороны:")
    print(f"   EFFECTIVE_LAT_MIN_BASE = {LAT_MIN - needed_ext_v:.1f}°")
    print(f"   EFFECTIVE_LAT_MAX_BASE = {LAT_MAX + needed_ext_v:.1f}°")
else:
    print(f"✅ Вертикальные границы можно оставить точными:")
    print(f"   EFFECTIVE_LAT_MIN_BASE = {LAT_MIN}°")
    print(f"   EFFECTIVE_LAT_MAX_BASE = {LAT_MAX}°")

if (FOV_MAX * ASPECT_RATIO) > (LON_MAX - LON_MIN):
    needed_ext_h = ((FOV_MAX * ASPECT_RATIO) - (LON_MAX - LON_MIN)) / 2 + 5  # +5° запас
    print(f"\n✅ Расширить горизонтальные границы на {needed_ext_h:.0f}° с каждой стороны:")
    print(f"   EFFECTIVE_LON_MIN = {LON_MIN - needed_ext_h:.1f}°")
    print(f"   EFFECTIVE_LON_MAX = {LON_MAX + needed_ext_h:.1f}°")
else:
    print(f"\n✅ Горизонтальные границы можно оставить точными:")
    print(f"   EFFECTIVE_LON_MIN = {LON_MIN}°")
    print(f"   EFFECTIVE_LON_MAX = {LON_MAX}°")

print("\n" + "=" * 80)
