#!/bin/bash
# Скрипт для удаления плохих калибровочных изображений
# Основан на анализе calibration_cleanup_report.json

echo "========================================================================"
echo "УДАЛЕНИЕ ПЛОХИХ КАЛИБРОВОЧНЫХ ИЗОБРАЖЕНИЙ"
echo "========================================================================"
echo ""
echo "Этот скрипт удалит 26 изображений с критическими проблемами:"
echo "  - Камера 0: 21 файл (слишком маленькая доска в кадре)"
echo "  - Камера 1: 5 файлов (доска не найдена или высокая ошибка)"
echo ""
read -p "Продолжить? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Отменено."
    exit 1
fi

echo ""
echo "Удаление файлов..."

# Камера 0 (левая) - 21 файл
rm -v calibration/calibration_result/left/cam0_00062.jpg
rm -v calibration/calibration_result/left/cam0_00064.jpg
rm -v calibration/calibration_result/left/cam0_00065.jpg
rm -v calibration/calibration_result/left/cam0_00066.jpg
rm -v calibration/calibration_result/left/cam0_00087.jpg
rm -v calibration/calibration_result/left/cam0_00100.jpg
rm -v calibration/calibration_result/left/cam0_00127.jpg
rm -v calibration/calibration_result/left/cam0_00130.jpg
rm -v calibration/calibration_result/left/cam0_00145.jpg
rm -v calibration/calibration_result/left/cam0_00147.jpg
rm -v calibration/calibration_result/left/cam0_00160.jpg
rm -v calibration/calibration_result/left/cam0_00183.jpg
rm -v calibration/calibration_result/left/cam0_00189.jpg
rm -v calibration/calibration_result/left/cam0_00193.jpg
rm -v calibration/calibration_result/left/cam0_00205.jpg
rm -v calibration/calibration_result/left/cam0_00207.jpg
rm -v calibration/calibration_result/left/cam0_00211.jpg
rm -v calibration/calibration_result/left/cam0_00213.jpg
rm -v calibration/calibration_result/left/cam0_00216.jpg
rm -v calibration/calibration_result/left/cam0_00225.jpg
rm -v calibration/calibration_result/left/cam0_00226.jpg

echo ""
echo "Камера 0: удалено 21 файл"
echo ""

# Камера 1 (правая) - 5 файлов
rm -v calibration/calibration_result/right/cam1_00140.jpg
rm -v calibration/calibration_result/right/cam1_00256.jpg
rm -v calibration/calibration_result/right/cam1_00166.jpg
rm -v calibration/calibration_result/right/cam1_00211.jpg
rm -v calibration/calibration_result/right/cam1_00213.jpg

echo ""
echo "Камера 1: удалено 5 файлов"
echo ""

echo "========================================================================"
echo "ГОТОВО!"
echo "========================================================================"
echo ""
echo "Осталось изображений:"
echo "  Камера 0: $(ls calibration/calibration_result/left/*.jpg 2>/dev/null | wc -l) файлов"
echo "  Камера 1: $(ls calibration/calibration_result/right/*.jpg 2>/dev/null | wc -l) файлов"
echo ""
echo "Следующий шаг:"
echo "  Запустите скрипт калибровки заново с очищенным набором данных."
echo "  Ожидаемый RMS должен быть < 1.0 пикселя для обеих камер."
echo ""
