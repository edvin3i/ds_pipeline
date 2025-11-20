#!/bin/bash
# install_to_system.sh - Устанавливает библиотеки из lib_copy в системную папку GStreamer

set -e  # Выход при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Установка плагинов в систему${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Определяем директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_PLUGIN_DIR="$HOME/.local/share/gstreamer-1.0/plugins"

echo -e "${YELLOW}📂 Источник:${NC} $SCRIPT_DIR"
echo -e "${YELLOW}📂 Целевая папка:${NC} $SYSTEM_PLUGIN_DIR"
echo ""

# Проверяем наличие .so файлов в текущей папке
if ! ls "$SCRIPT_DIR"/*.so 1> /dev/null 2>&1; then
    echo -e "${RED}❌ Ошибка: В папке $SCRIPT_DIR нет .so файлов${NC}"
    echo -e "${YELLOW}Сначала запустите:${NC} ./collect_libs.sh"
    exit 1
fi

# Создаём целевую директорию если её нет
mkdir -p "$SYSTEM_PLUGIN_DIR"
echo -e "${GREEN}✓${NC} Целевая папка готова"
echo ""

# Счётчики
INSTALLED=0
FAILED=0

echo -e "${BLUE}🔧 Установка библиотек...${NC}"
echo ""

# Копируем все .so файлы
for lib_path in "$SCRIPT_DIR"/*.so; do
    lib_name=$(basename "$lib_path")

    # Проверяем существует ли файл в системе
    if [ -f "$SYSTEM_PLUGIN_DIR/$lib_name" ]; then
        echo -e "${YELLOW}⚠${NC} Заменяем: $lib_name"
    else
        echo -e "${GREEN}+${NC} Новый файл: $lib_name"
    fi

    # Копируем файл
    if cp "$lib_path" "$SYSTEM_PLUGIN_DIR/"; then
        # Получаем размер файла
        size=$(ls -lh "$SYSTEM_PLUGIN_DIR/$lib_name" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} Установлено ${YELLOW}($size)${NC}"
        INSTALLED=$((INSTALLED + 1))
    else
        echo -e "  ${RED}✗${NC} Ошибка копирования"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Итоговая статистика
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Установлено:${NC} $INSTALLED библиотек"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}✗ Ошибок:${NC} $FAILED"
fi
echo -e "${BLUE}========================================${NC}"

# Показываем содержимое системной папки
echo ""
echo -e "${BLUE}📦 Установленные плагины в системе:${NC}"
ls -lh "$SYSTEM_PLUGIN_DIR"/*.so 2>/dev/null || echo "  (нет .so файлов)"

echo ""
echo -e "${GREEN}✓ Установка завершена!${NC}"
echo ""
echo -e "${BLUE}📝 Проверка плагинов:${NC}"
echo "  gst-inspect-1.0 nvtilebatcher"
echo "  gst-inspect-1.0 nvdsstitch"
echo "  gst-inspect-1.0 nvdsvirtualcam"
echo ""
echo -e "${YELLOW}💡 Совет:${NC} Перезапустите приложение чтобы изменения вступили в силу"
