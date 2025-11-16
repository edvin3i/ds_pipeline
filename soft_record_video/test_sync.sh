#!/bin/bash
# Тестовый скрипт для проверки hardware sync камер

echo "=========================================="
echo "ТЕСТ СИНХРОНИЗАЦИИ КАМЕР"
echo "=========================================="
echo ""

# Проверяем наличие камер
echo "[1] Проверка камер..."
if [ ! -e /dev/video0 ]; then
    echo "❌ /dev/video0 не найден"
    exit 1
fi

if [ ! -e /dev/video1 ]; then
    echo "❌ /dev/video1 не найден"
    exit 1
fi

echo "✅ Обе камеры найдены: /dev/video0 и /dev/video1"
echo ""

# Проверяем v4l2-ctl
echo "[2] Проверка v4l2-ctl..."
if ! command -v v4l2-ctl &> /dev/null; then
    echo "❌ v4l2-ctl не найден. Установите: sudo apt install v4l-utils"
    exit 1
fi
echo "✅ v4l2-ctl найден"
echo ""

# Показываем текущие параметры
echo "[3] Текущие параметры камер:"
echo "--- /dev/video0 ---"
v4l2-ctl -d /dev/video0 --list-ctrls 2>&1 | grep -i "operation\|sync" || echo "Параметры sync не найдены"
echo ""
echo "--- /dev/video1 ---"
v4l2-ctl -d /dev/video1 --list-ctrls 2>&1 | grep -i "operation\|sync" || echo "Параметры sync не найдены"
echo ""

# Настраиваем hardware sync
echo "[4] Настройка hardware sync..."
echo "Настраиваем /dev/video0 как MASTER (operation_mode=0, synchronizing_function=1)"
v4l2-ctl -d /dev/video0 -c operation_mode=0 -c synchronizing_function=1 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Мастер настроен"
else
    echo "⚠️  Не удалось настроить мастер (возможно, камеры не поддерживают эти параметры)"
fi

echo ""
echo "Настраиваем /dev/video1 как SLAVE (operation_mode=1, synchronizing_function=2)"
v4l2-ctl -d /dev/video1 -c operation_mode=1 -c synchronizing_function=2 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Слейв настроен"
else
    echo "⚠️  Не удалось настроить слейв (возможно, камеры не поддерживают эти параметры)"
fi

echo ""
echo "=========================================="
echo "Тест завершен!"
echo "=========================================="
echo ""
echo "Теперь можно запустить запись:"
echo "  python3 synced_dual_record.py"
echo ""
echo "Или с дополнительными опциями:"
echo "  python3 synced_dual_record.py --bitrate 35 --codec h265"
echo ""
