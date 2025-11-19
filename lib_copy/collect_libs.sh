#!/bin/bash
# collect_libs.sh - Собирает библиотеки плагинов из исходных папок в lib_copy

set -e  # Выход при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Сбор библиотек плагинов DeepStream${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Определяем базовую директорию проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIB_COPY_DIR="$SCRIPT_DIR"

echo -e "${YELLOW}📂 Директория проекта:${NC} $PROJECT_DIR"
echo -e "${YELLOW}📂 Целевая папка:${NC} $LIB_COPY_DIR"
echo ""

# Список плагинов для копирования
# Формат: "путь_к_плагину:имя_файла"
PLUGINS=(
    "$PROJECT_DIR/my_tile_batcher:libnvtilebatcher.so"
    "$PROJECT_DIR/my_steach:libnvdsstitch.so"
    "$PROJECT_DIR/my_virt_cam:libnvdsvirtualcam.so"
)

# Счётчики
COPIED=0
FAILED=0

echo -e "${BLUE}🔍 Поиск и копирование библиотек...${NC}"
echo ""

for plugin_info in "${PLUGINS[@]}"; do
    # Разделяем путь и имя файла
    plugin_dir="${plugin_info%%:*}"
    lib_name="${plugin_info##*:}"

    # Ищем библиотеку в папке и подпапках
    lib_path=""

    # Сначала проверяем в корне папки плагина
    if [ -f "$plugin_dir/$lib_name" ]; then
        lib_path="$plugin_dir/$lib_name"
    # Потом в подпапке src/
    elif [ -f "$plugin_dir/src/$lib_name" ]; then
        lib_path="$plugin_dir/src/$lib_name"
    fi

    if [ -n "$lib_path" ]; then
        # Копируем библиотеку
        cp "$lib_path" "$LIB_COPY_DIR/"

        # Получаем размер файла
        size=$(ls -lh "$lib_path" | awk '{print $5}')

        echo -e "${GREEN}✓${NC} Скопировано: $lib_name ${YELLOW}($size)${NC}"
        echo -e "  Источник: $lib_path"
        COPIED=$((COPIED + 1))
    else
        echo -e "${YELLOW}⚠${NC} Не найдено: $lib_name"
        echo -e "  Искали в: $plugin_dir/"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Итоговая статистика
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Скопировано:${NC} $COPIED библиотек"
if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}⚠ Не найдено:${NC} $FAILED библиотек"
fi
echo -e "${BLUE}========================================${NC}"

# Показываем содержимое папки
echo ""
echo -e "${BLUE}📦 Содержимое $LIB_COPY_DIR:${NC}"
ls -lh "$LIB_COPY_DIR"/*.so 2>/dev/null || echo "  (нет .so файлов)"

echo ""
echo -e "${GREEN}✓ Готово!${NC}"
echo -e "${YELLOW}Для установки в систему запустите:${NC} ./install_to_system.sh"
