#!/usr/bin/env python3
"""
Тест реальных границ на сферической проекции
Найдем при каких pitch/yaw появляются черные полосы
"""
import math

# Параметры панорамы
LON_MIN = -90.0
LON_MAX = +90.0
LAT_MIN = -32.0
LAT_MAX = +22.0

PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900
ASPECT_RATIO = 16.0 / 9.0

def spherical_to_cartesian(yaw, pitch):
    """Конвертирует сферические координаты в 3D точку на единичной сфере"""
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    x = math.cos(pitch_rad) * math.sin(yaw_rad)
    y = math.sin(pitch_rad)
    z = math.cos(pitch_rad) * math.cos(yaw_rad)

    return (x, y, z)

def angle_between_vectors(v1, v2):
    """Угол между двумя 3D векторами"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(a**2 for a in v2))

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))

def test_camera_position(camera_pitch, camera_yaw, fov):
    """
    Проверяет, выходит ли камера за границы панорамы
    Возвращает: (has_black_bars, details)
    """
    half_fov_v = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    # Простой расчет краев камеры (линейное приближение)
    camera_top = camera_pitch + half_fov_v
    camera_bottom = camera_pitch - half_fov_v
    camera_left = camera_yaw - half_fov_h
    camera_right = camera_yaw + half_fov_h

    # Проверяем выход за границы
    issues = []

    if camera_top > LAT_MAX:
        issues.append(f"ВЕРХ: {camera_top:.1f}° > {LAT_MAX}° (выход на {camera_top - LAT_MAX:.1f}°)")

    if camera_bottom < LAT_MIN:
        issues.append(f"НИЗ: {camera_bottom:.1f}° < {LAT_MIN}° (выход на {LAT_MIN - camera_bottom:.1f}°)")

    if camera_left < LON_MIN:
        issues.append(f"ЛЕВО: {camera_left:.1f}° < {LON_MIN}° (выход на {LON_MIN - camera_left:.1f}°)")

    if camera_right > LON_MAX:
        issues.append(f"ПРАВО: {camera_right:.1f}° > {LON_MAX}° (выход на {camera_right - LON_MAX:.1f}°)")

    return len(issues) > 0, issues, {
        'top': camera_top,
        'bottom': camera_bottom,
        'left': camera_left,
        'right': camera_right
    }

def find_safe_boundaries_for_fov(fov):
    """
    Находит безопасные границы для заданного FOV
    Возвращает: (pitch_min, pitch_max, yaw_min, yaw_max)
    """
    half_fov_v = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    # Безопасные границы = границы панорамы минус half_fov
    pitch_min_safe = LAT_MIN + half_fov_v
    pitch_max_safe = LAT_MAX - half_fov_v
    yaw_min_safe = LON_MIN + half_fov_h
    yaw_max_safe = LON_MAX - half_fov_h

    return pitch_min_safe, pitch_max_safe, yaw_min_safe, yaw_max_safe

print("=" * 100)
print("ТЕСТ: РЕАЛЬНЫЕ ГРАНИЦЫ НА СФЕРИЧЕСКОЙ ПРОЕКЦИИ")
print("=" * 100)

print(f"\n📐 Параметры панорамы:")
print(f"   Вертикаль (LAT): {LAT_MIN}° до {LAT_MAX}° (покрытие: {LAT_MAX - LAT_MIN}°)")
print(f"   Горизонталь (LON): {LON_MIN}° до {LON_MAX}° (покрытие: {LON_MAX - LON_MIN}°)")

print("\n" + "=" * 100)
print("АНАЛИЗ ДЛЯ КАЖДОГО FOV")
print("=" * 100)

for fov in [40.0, 50.0, 60.0, 68.0]:
    print(f"\n{'='*100}")
    print(f"FOV = {fov}°")
    print(f"{'='*100}")

    half_fov_v = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    print(f"\n📊 Параметры:")
    print(f"   Вертикальный FOV: {fov}° (половина: ±{half_fov_v}°)")
    print(f"   Горизонтальный FOV: {horizontal_fov:.1f}° (половина: ±{half_fov_h:.1f}°)")

    # Вычисляем теоретические безопасные границы
    pitch_min_safe, pitch_max_safe, yaw_min_safe, yaw_max_safe = find_safe_boundaries_for_fov(fov)

    print(f"\n🎯 Теоретические безопасные границы (линейный расчет):")
    print(f"   Pitch: {pitch_min_safe:+.1f}° до {pitch_max_safe:+.1f}° (свобода: {pitch_max_safe - pitch_min_safe:.1f}°)")
    print(f"   Yaw: {yaw_min_safe:+.1f}° до {yaw_max_safe:+.1f}° (свобода: {yaw_max_safe - yaw_min_safe:.1f}°)")

    if pitch_max_safe < pitch_min_safe:
        print(f"\n   ⚠️ ВНИМАНИЕ: pitch_max < pitch_min! FOV больше высоты панорамы!")
        print(f"   Это означает что ЛЮБАЯ позиция камеры будет иметь черные полосы")

    # Тестируем разные позиции
    test_positions = [
        (LAT_MAX, 0.0, "ВЕРХ ЦЕНТР"),
        (LAT_MIN, 0.0, "НИЗ ЦЕНТР"),
        (0.0, LON_MIN, "ЦЕНТР ЛЕВО"),
        (0.0, LON_MAX, "ЦЕНТР ПРАВО"),
        ((LAT_MIN + LAT_MAX) / 2, 0.0, "ЦЕНТР ПАНОРАМЫ"),
        (LAT_MAX, LON_MIN, "ВЕРХ ЛЕВО (угол)"),
        (LAT_MAX, LON_MAX, "ВЕРХ ПРАВО (угол)"),
        (LAT_MIN, LON_MIN, "НИЗ ЛЕВО (угол)"),
        (LAT_MIN, LON_MAX, "НИЗ ПРАВО (угол)"),
    ]

    print(f"\n📍 Тестирование позиций:")
    print("-" * 100)

    positions_with_black_bars = 0

    for test_pitch, test_yaw, name in test_positions:
        has_black_bars, issues, bounds = test_camera_position(test_pitch, test_yaw, fov)

        if has_black_bars:
            positions_with_black_bars += 1
            status = "❌"
        else:
            status = "✅"

        print(f"\n{status} {name:25} pitch={test_pitch:+6.1f}° yaw={test_yaw:+7.1f}°")
        print(f"   📷 Края кадра: верх={bounds['top']:+6.1f}° низ={bounds['bottom']:+6.1f}° лево={bounds['left']:+7.1f}° право={bounds['right']:+7.1f}°")

        if has_black_bars:
            for issue in issues:
                print(f"      ⚠️ {issue}")

    print(f"\n{'='*100}")
    print(f"📊 Итого для FOV={fov}°:")
    print(f"   Позиций с черными полосами: {positions_with_black_bars} из {len(test_positions)}")

    if positions_with_black_bars == 0:
        print(f"   ✅ ОТЛИЧНО: Нет черных полос ни в одной позиции!")
    elif positions_with_black_bars == len(test_positions):
        print(f"   ❌ КРИТИЧНО: Черные полосы ВЕЗДЕ (FOV > панорама)")
    else:
        print(f"   ⚠️ ЧАСТИЧНО: Черные полосы в {positions_with_black_bars}/{len(test_positions)} позициях")

print("\n" + "=" * 100)
print("ВЫВОД: КАК НАСТРОИТЬ ГРАНИЦЫ")
print("=" * 100)

print("""
🎯 Два подхода:

1️⃣ **ДИНАМИЧЕСКИЕ ГРАНИЦЫ (зависят от FOV)**

   Для каждого FOV вычисляем безопасные границы:

   ```cpp
   gfloat half_fov_v = fov / 2.0f;
   gfloat horizontal_fov = fov * 16.0f / 9.0f;
   gfloat half_fov_h = horizontal_fov / 2.0f;

   gfloat pitch_min = LAT_MIN + half_fov_v;  // -32 + half_fov
   gfloat pitch_max = LAT_MAX - half_fov_v;  // +22 - half_fov
   gfloat yaw_min = LON_MIN + half_fov_h;    // -90 + half_fov_h
   gfloat yaw_max = LON_MAX - half_fov_h;    // +90 - half_fov_h
   ```

   ✅ Плюсы: НЕТ черных полос (гарантированно)
   ❌ Минусы: При FOV > 54° (больше высоты панорамы) камера застревает

2️⃣ **СТАТИЧЕСКИЕ ГРАНИЦЫ (фиксированные для всех FOV)**

   Просто ограничиваем центр камеры границами панорамы:

   ```cpp
   gfloat pitch_min = LAT_MIN;  // -32°
   gfloat pitch_max = LAT_MAX;  // +22°
   gfloat yaw_min = LON_MIN;    // -90°
   gfloat yaw_max = LON_MAX;    // +90°
   ```

   ✅ Плюсы: Камера ВСЕГДА может двигаться (даже при FOV=68°)
   ❌ Минусы: БУДУТ черные полосы когда камера у краев панорамы

3️⃣ **ГИБРИДНЫЙ ПОДХОД (рекомендуется)**

   Используем динамические границы, но только если они валидны:

   ```cpp
   gfloat half_fov_v = fov / 2.0f;
   gfloat horizontal_fov = fov * 16.0f / 9.0f;
   gfloat half_fov_h = horizontal_fov / 2.0f;

   // Пытаемся вычислить безопасные границы
   gfloat pitch_min = LAT_MIN + half_fov_v;
   gfloat pitch_max = LAT_MAX - half_fov_v;

   // Если границы невалидны (FOV слишком большой)
   if (pitch_min >= pitch_max) {
       // Используем статические границы (будут черные полосы)
       pitch_min = LAT_MIN;
       pitch_max = LAT_MAX;
   }

   // То же самое для yaw
   gfloat yaw_min = LON_MIN + half_fov_h;
   gfloat yaw_max = LON_MAX - half_fov_h;

   if (yaw_min >= yaw_max) {
       yaw_min = LON_MIN;
       yaw_max = LON_MAX;
   }
   ```

   ✅ Плюсы: Нет черных полос при малых FOV (40-50°)
   ✅ Плюсы: Камера не застревает при больших FOV (60-68°)
   ⚠️ Минусы: При FOV > 54° появляются черные полосы у краев

📌 РЕКОМЕНДАЦИЯ: Используй **ГИБРИДНЫЙ ПОДХОД** (вариант 3)
   - При FOV=40-50° (обычная игра): нет черных полос
   - При FOV=60-68° (сильный зум): черные полосы есть, но камера работает
""")

print("=" * 100)
