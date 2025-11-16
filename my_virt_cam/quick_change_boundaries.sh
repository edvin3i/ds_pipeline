#!/bin/bash
# Быстрое изменение границ панорамы для тестирования
# Использование: ./quick_change_boundaries.sh LAT_MIN LAT_MAX
# Пример: ./quick_change_boundaries.sh -35.0 25.0

if [ $# -ne 2 ]; then
    echo "❌ Использование: $0 LAT_MIN LAT_MAX"
    echo "Пример: $0 -35.0 25.0"
    exit 1
fi

LAT_MIN=$1
LAT_MAX=$2

# Вычислить высоту
HEIGHT=$(echo "$LAT_MAX - ($LAT_MIN)" | bc)

echo "════════════════════════════════════════════════════════"
echo "📐 Изменение границ панорамы"
echo "════════════════════════════════════════════════════════"
echo "LAT_MIN = $LAT_MIN°"
echo "LAT_MAX = $LAT_MAX°"
echo "Высота = $HEIGHT°"
echo ""

# Создать временный файл с новыми границами
cd "$(dirname "$0")"

# Сохранить backup
cp src/nvdsvirtualcam_config.h src/nvdsvirtualcam_config.h.backup

# Изменить границы используя sed
sed -i "s/constexpr float LAT_MIN = -[0-9.]*f;/constexpr float LAT_MIN = ${LAT_MIN}f;/" src/nvdsvirtualcam_config.h
sed -i "s/constexpr float LAT_MAX = +[0-9.]*f;/constexpr float LAT_MAX = +${LAT_MAX#-}f;/" src/nvdsvirtualcam_config.h

# Обновить комментарий с высотой
sed -i "s/~[0-9]*°/~${HEIGHT%.*}°/" src/nvdsvirtualcam_config.h

echo "✅ Конфиг обновлён!"
echo ""
echo "🔨 Пересборка плагина..."
cd src
make clean > /dev/null 2>&1
make 2>&1 | grep -E "(✓|✅|ERROR|error)" || echo "Сборка завершена"

if [ -f libnvdsvirtualcam.so ]; then
    echo "✅ Плагин успешно пересобран!"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "🚀 Запустите тест:"
    echo "   python3 test_virtual_camera_keyboard.py ../new_week/left.mp4 ../new_week/right.mp4"
    echo "════════════════════════════════════════════════════════"
else
    echo "❌ Ошибка сборки!"
    echo "Восстанавливаем backup..."
    cp nvdsvirtualcam_config.h.backup nvdsvirtualcam_config.h
    exit 1
fi
