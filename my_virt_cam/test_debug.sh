#!/bin/bash
# Test script to capture debug output

cd /home/nvidia/deep_cv_football/my_virt_cam

echo "🔍 Starting virtual camera test with debug logging..."
echo "Press W key to move ball UP"
echo "Press Ctrl+C to stop"
echo ""

python3 test_virtual_camera_keyboard.py ../new_week/left.mp4 ../new_week/right.mp4 2>&1 | grep -E "(XY→ANGLE|TARGET_PITCH|BEFORE|AFTER|PITCH:)" || true
