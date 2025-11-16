#!/bin/bash
# Быстрое тестирование разных значений ball_y
# Использование: ./quick_test_ball_y.sh BALL_Y PITCH
# Пример: ./quick_test_ball_y.sh 665 5

if [ $# -ne 2 ]; then
    echo "❌ Использование: $0 BALL_Y PITCH"
    echo "Примеры:"
    echo "  $0 665 5     # ball_y=665px, pitch=+5°"
    echo "  $0 760 2     # ball_y=760px, pitch=+2°"
    echo "  $0 809.9 0   # ball_y=810px, pitch=0° (горизонт)"
    exit 1
fi

BALL_Y=$1
PITCH=$2

cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════"
echo "🎯 Изменение позиции мяча"
echo "════════════════════════════════════════════════════════"
echo "ball_y = ${BALL_Y}px"
echo "pitch = ${PITCH}°"
echo ""

# Изменить ball_y в Python файле
sed -i "s/self\.ball_y = [0-9.]*.*$/self.ball_y = ${BALL_Y}  # pitch ≈ ${PITCH}°/" test_virtual_camera_keyboard.py

# Изменить начальный pitch в pipeline
sed -i "s/pitch=[0-9-]*/pitch=${PITCH}/" test_virtual_camera_keyboard.py

echo "✅ Файл обновлён!"
echo ""
echo "════════════════════════════════════════════════════════"
echo "🚀 Запустите тест:"
echo "   python3 test_virtual_camera_keyboard.py ../new_week/left.mp4 ../new_week/right.mp4"
echo "════════════════════════════════════════════════════════"
