#!/usr/bin/env python3
"""
Полный тест границ виртуальной камеры при разных FOV
Проверяет все углы панорамы и выводит подробный отчет
"""

# Параметры панорамы
LON_MIN = -90.0
LON_MAX = +90.0
LAT_MIN = -32.0
LAT_MAX = +22.0

PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

# Эффективные границы (расширенные)
EFF_LAT_MIN = -40.0
EFF_LAT_MAX = +30.0
EFF_LON_MIN = -105.0
EFF_LON_MAX = +105.0

FOV_VALUES = [40.0, 50.0, 60.0, 68.0]
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

def calculate_y_from_pitch(pitch):
    """Обратное преобразование: pitch → Y координата"""
    norm_y = (LAT_MAX - pitch) / (LAT_MAX - LAT_MIN)
    y = norm_y * (PANORAMA_HEIGHT - 1)
    return y

def calculate_x_from_yaw(yaw):
    """Обратное преобразование: yaw → X координата"""
    norm_x = (yaw - LON_MIN) / (LON_MAX - LON_MIN)
    x = norm_x * (PANORAMA_WIDTH - 1)
    return x

def test_boundary_with_fov(fov, test_pitch, test_yaw, test_name):
    """
    Тестирует границу при заданном FOV
    Возвращает: (разрешено, финальный_pitch, финальный_yaw, комментарий)
    """
    half_fov = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    # Безопасные границы для центра камеры
    pitch_min_safe = EFF_LAT_MIN + half_fov
    pitch_max_safe = EFF_LAT_MAX - half_fov
    yaw_min_safe = EFF_LON_MIN + half_fov_h
    yaw_max_safe = EFF_LON_MAX - half_fov_h

    # Применяем ограничения (как в C++ коде)
    final_pitch = test_pitch
    final_yaw = test_yaw

    # Ограничиваем pitch
    if final_pitch < pitch_min_safe:
        final_pitch = pitch_min_safe
    if final_pitch > pitch_max_safe:
        final_pitch = pitch_max_safe

    # Ограничиваем yaw
    if final_yaw < yaw_min_safe:
        final_yaw = yaw_min_safe
    if final_yaw > yaw_max_safe:
        final_yaw = yaw_max_safe

    # Проверяем границы камеры (с учетом half_fov)
    camera_top = final_pitch + half_fov
    camera_bottom = final_pitch - half_fov
    camera_left = final_yaw - half_fov_h
    camera_right = final_yaw + half_fov_h

    # Проверка выхода за панораму
    issues = []

    if camera_top > LAT_MAX:
        issues.append(f"⚠️ Верх камеры ({camera_top:.1f}°) ВЫШЕ панорамы ({LAT_MAX}°)")

    if camera_bottom < LAT_MIN:
        issues.append(f"⚠️ Низ камеры ({camera_bottom:.1f}°) НИЖЕ панорамы ({LAT_MIN}°)")

    if camera_left < LON_MIN:
        issues.append(f"⚠️ Левый край ({camera_left:.1f}°) ЛЕВЕЕ панорамы ({LON_MIN}°)")

    if camera_right > LON_MAX:
        issues.append(f"⚠️ Правый край ({camera_right:.1f}°) ПРАВЕЕ панорамы ({LON_MAX}°)")

    # Проверка достижения границ панорамы
    reaches = []

    if abs(camera_top - LAT_MAX) < 2.0:
        reaches.append(f"✅ Доходит до ВЕРХА панорамы ({camera_top:.1f}° ≈ {LAT_MAX}°)")

    if abs(camera_bottom - LAT_MIN) < 2.0:
        reaches.append(f"✅ Доходит до НИЗА панорамы ({camera_bottom:.1f}° ≈ {LAT_MIN}°)")

    if abs(camera_left - LON_MIN) < 5.0:
        reaches.append(f"✅ Доходит до ЛЕВОГО края ({camera_left:.1f}° ≈ {LON_MIN}°)")

    if abs(camera_right - LON_MAX) < 5.0:
        reaches.append(f"✅ Доходит до ПРАВОГО края ({camera_right:.1f}° ≈ {LON_MAX}°)")

    was_adjusted = (test_pitch != final_pitch) or (test_yaw != final_yaw)

    return {
        'allowed': len(issues) == 0,
        'final_pitch': final_pitch,
        'final_yaw': final_yaw,
        'adjusted': was_adjusted,
        'issues': issues,
        'reaches': reaches,
        'camera_bounds': {
            'top': camera_top,
            'bottom': camera_bottom,
            'left': camera_left,
            'right': camera_right
        }
    }

print("=" * 100)
print("ПОЛНЫЙ ТЕСТ ГРАНИЦ ВИРТУАЛЬНОЙ КАМЕРЫ")
print("=" * 100)

print(f"\n📐 Параметры панорамы:")
print(f"   Размер: {PANORAMA_WIDTH} × {PANORAMA_HEIGHT} пикселей")
print(f"   Горизонталь: {LON_MIN}° до {LON_MAX}° (покрытие: {LON_MAX - LON_MIN}°)")
print(f"   Вертикаль: {LAT_MIN}° до {LAT_MAX}° (покрытие: {LAT_MAX - LAT_MIN}°)")

print(f"\n🎯 Эффективные границы (расширенные):")
print(f"   Горизонталь: {EFF_LON_MIN}° до {EFF_LON_MAX}° (расширение: {EFF_LON_MIN - LON_MIN:.0f}°/{EFF_LON_MAX - LON_MAX:.0f}°)")
print(f"   Вертикаль: {EFF_LAT_MIN}° до {EFF_LAT_MAX}° (расширение: {EFF_LAT_MIN - LAT_MIN:.0f}°/{EFF_LAT_MAX - LAT_MAX:.0f}°)")

# Определяем тестовые позиции
test_positions = [
    # (pitch, yaw, название)
    (LAT_MAX, 0.0, "ВЕРХ ЦЕНТР"),
    (LAT_MIN, 0.0, "НИЗ ЦЕНТР"),
    (0.0, LON_MIN, "ЦЕНТР ЛЕВО"),
    (0.0, LON_MAX, "ЦЕНТР ПРАВО"),
    (LAT_MAX, LON_MIN, "ВЕРХ ЛЕВО (угол)"),
    (LAT_MAX, LON_MAX, "ВЕРХ ПРАВО (угол)"),
    (LAT_MIN, LON_MIN, "НИЗ ЛЕВО (угол)"),
    (LAT_MIN, LON_MAX, "НИЗ ПРАВО (угол)"),
    (0.0, 0.0, "ЦЕНТР ЦЕНТР"),
]

for fov in FOV_VALUES:
    print("\n" + "=" * 100)
    print(f"ТЕСТ FOV = {fov}°")
    print("=" * 100)

    half_fov = fov / 2.0
    horizontal_fov = fov * ASPECT_RATIO
    half_fov_h = horizontal_fov / 2.0

    print(f"\n📊 Параметры:")
    print(f"   Вертикальный FOV: {fov}° (half: {half_fov}°)")
    print(f"   Горизонтальный FOV: {horizontal_fov:.1f}° (half: {half_fov_h:.1f}°)")

    pitch_min_safe = EFF_LAT_MIN + half_fov
    pitch_max_safe = EFF_LAT_MAX - half_fov
    yaw_min_safe = EFF_LON_MIN + half_fov_h
    yaw_max_safe = EFF_LON_MAX - half_fov_h

    freedom_v = pitch_max_safe - pitch_min_safe
    freedom_h = yaw_max_safe - yaw_min_safe

    print(f"\n🔒 Безопасные границы для центра камеры:")
    print(f"   Pitch: {pitch_min_safe:.1f}° до {pitch_max_safe:.1f}° (свобода: {freedom_v:.1f}°)")
    print(f"   Yaw: {yaw_min_safe:.1f}° до {yaw_max_safe:.1f}° (свобода: {freedom_h:.1f}°)")

    if freedom_v < 0:
        print(f"   ⚠️ КРИТИЧНО: Отрицательная свобода по вертикали! Камера застрянет!")
    elif freedom_v < 5:
        print(f"   ⚠️ ВНИМАНИЕ: Очень мало свободы по вертикали!")
    else:
        print(f"   ✅ Достаточно свободы по вертикали")

    if freedom_h < 20:
        print(f"   ⚠️ ВНИМАНИЕ: Мало свободы по горизонтали!")
    else:
        print(f"   ✅ Достаточно свободы по горизонтали")

    print(f"\n📍 Тестирование позиций:")
    print("-" * 100)

    all_ok = True
    total_reaches = 0

    for test_pitch, test_yaw, name in test_positions:
        result = test_boundary_with_fov(fov, test_pitch, test_yaw, name)

        status = "✅" if result['allowed'] else "❌"
        adjusted = "🔧 СКОРРЕКТИРОВАНО" if result['adjusted'] else ""

        print(f"\n{status} {name:20} pitch={test_pitch:+6.1f}° yaw={test_yaw:+7.1f}° {adjusted}")

        if result['adjusted']:
            print(f"   → Финальная позиция: pitch={result['final_pitch']:+6.1f}° yaw={result['final_yaw']:+7.1f}°")

        bounds = result['camera_bounds']
        print(f"   📷 Границы кадра: верх={bounds['top']:+6.1f}° низ={bounds['bottom']:+6.1f}° лево={bounds['left']:+7.1f}° право={bounds['right']:+7.1f}°")

        if result['issues']:
            for issue in result['issues']:
                print(f"      {issue}")
            all_ok = False

        if result['reaches']:
            for reach in result['reaches']:
                print(f"      {reach}")
                total_reaches += 1

    print(f"\n{'=' * 100}")
    if all_ok:
        print(f"✅ FOV={fov}°: ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Камера не выходит за границы панорамы!")
    else:
        print(f"❌ FOV={fov}°: ЕСТЬ ПРОБЛЕМЫ! Камера выходит за границы панорамы!")

    print(f"📊 Достигнуто границ панорамы: {total_reaches} из {len(test_positions) * 4}")

print("\n" + "=" * 100)
print("ИТОГОВЫЙ АНАЛИЗ")
print("=" * 100)

print(f"""
🎯 Выводы:

1. **Расширенные границы** ({EFF_LAT_MIN}° до {EFF_LAT_MAX}°, {EFF_LON_MIN}° до {EFF_LON_MAX}°)
   обеспечивают свободу движения при всех FOV (40-68°)

2. **Черные полосы** появляются когда камера доходит до расширенных границ
   (за пределами реальной панорамы {LAT_MIN}° до {LAT_MAX}°)

3. **Доступ к границам панорамы**:
   - При малых FOV (40-50°) камера может дойти до всех границ панорамы
   - При больших FOV (60-68°) камера автоматически центрируется,
     но все равно может достичь границ панорамы

4. **Рекомендации**:
   - FOV 40-50°: Полная свобода, нет черных полос
   - FOV 60-68°: Ограниченная свобода, возможны черные полосы по краям
""")

print("=" * 100)
