#!/bin/bash
# Тест границ камеры для разных FOV

export GST_PLUGIN_PATH=/home/nvidia/deep_cv_football/my_virt_cam/src:$GST_PLUGIN_PATH
export GST_DEBUG=nvdsvirtualcam:3

echo "========================================="
echo "  ТЕСТ ГРАНИЦ ПРИ РАЗНЫХ FOV"
echo "========================================="
echo ""
echo "Этот тест покажет, как работают границы"
echo "камеры при разных значениях FOV."
echo ""
echo "Запустите test_virtual_camera_keyboard.py"
echo "и попробуйте перемещаться по краям при"
echo "разных значениях зума (клавиши Z/X)."
echo ""
echo "При правильных границах:"
echo "  ✅ Не должно быть черных полос"
echo "  ✅ Камера должна автоматически"
echo "     ограничиваться краями панорамы"
echo ""
echo "FOV диапазон: 40° - 68°"
echo ""
echo "========================================="
