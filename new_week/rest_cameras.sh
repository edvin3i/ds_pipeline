#!/bin/bash
# reset_cameras.sh

echo "🔧 ЭКСТРЕННЫЙ СБРОС КАМЕР И ОСТАНОВКА СТРИМА"
echo "============================================="

# 1. Останавливаем скрипт автоперезапуска
echo "📛 Останавливаем автоперезапуск..."
pkill -f auto_restart_safe.sh
pkill -f auto_restart.sh
sleep 1

# 2. Останавливаем основную программу
echo "⏹️  Останавливаем все version_masr*.py..."
pkill -f "python3.*version_masr"
sleep 2

# 3. Останавливаем все GStreamer процессы
echo "🎬 Останавливаем GStreamer..."
pkill -f nvarguscamerasrc
pkill -f gstreamer
pkill -f nvdsvirtualcam
sleep 1

# 4. Проверяем, что всё остановлено
if pgrep -f "python3.*version_masr" > /dev/null; then
    echo "⚠️  Форсированная остановка Python..."
    pkill -9 -f "python3.*version_masr"
    sleep 1
fi

if pgrep -f auto_restart > /dev/null; then
    echo "⚠️  Форсированная остановка скриптов..."
    pkill -9 -f auto_restart
    sleep 1
fi

# 5. Проверяем занятость камер
echo "🔍 Проверка камер..."
if lsof /dev/video* 2>/dev/null | grep -E "python|gst"; then
    echo "⚠️  Камеры всё ещё заняты, освобождаем..."
    lsof /dev/video* 2>/dev/null | grep -E "python|gst" | awk '{print $2}' | xargs -r kill -9
    sleep 2
fi

# 6. Сброс nvargus-daemon (если есть права)
if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo "🔄 Перезапуск nvargus-daemon..."
    sudo systemctl restart nvargus-daemon
    sleep 3
else
    echo "⚠️  Нет прав для перезапуска nvargus-daemon"
    echo "   Выполните: sudo systemctl restart nvargus-daemon"
fi

# 7. Финальная проверка
echo ""
echo "📊 СТАТУС:"
echo "----------"

if pgrep -f auto_restart > /dev/null; then
    echo "❌ Скрипт автоперезапуска ВСЁ ЕЩЁ работает!"
else
    echo "✅ Скрипт автоперезапуска остановлен"
fi

if pgrep -f "python3.*version_masr" > /dev/null; then
    echo "❌ version_masr*.py ВСЁ ЕЩЁ работает!"
else
    echo "✅ version_masr*.py остановлен"
fi

if lsof /dev/video* 2>/dev/null | grep -E "python|gst" > /dev/null; then
    echo "❌ Камеры ВСЁ ЕЩЁ заняты!"
else
    echo "✅ Камеры свободны"
fi

echo ""
echo "🎯 Готово! Теперь можно запускать заново."
echo ""
echo "Для запуска используйте:"
echo "  ./auto_restart_safe.sh"