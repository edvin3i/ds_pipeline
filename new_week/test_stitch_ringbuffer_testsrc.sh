#!/bin/bash

# Тест stitch + ringbuffer с videotestsrc (без LUT файлов)
# Используем простую цветокоррекцию вместо панорамного стичинга

cd /home/nvidia/deep_cv_football/new_week

export GST_PLUGIN_PATH=/home/nvidia/deep_cv_football/my_steach:/home/nvidia/deep_cv_football/my_ring_buffer
export GST_DEBUG=ringbuffer:4,nvdsstitch:2

echo "========================================="
echo "🧪 ТЕСТ: stitch + ringbuffer (videotestsrc)"
echo "========================================="
echo ""
echo "📹 Источник: videotestsrc (2 паттерна)"
echo "🎨 Stitch: без LUT (простая склейка)"
echo "🔀 TEE: 2 ветки для проверки PTS"
echo ""
echo "⏱️  Цель: проверить работу ringbuffer"
echo "========================================="
echo ""

# Создаём временную папку для пустых LUT
mkdir -p /tmp/empty_warps
cd /tmp/empty_warps

# Создаём dummy LUT файлы нужного размера (4096x2048 = 32 MB float)
dd if=/dev/zero of=lut_left_x.bin bs=1M count=32 2>/dev/null
dd if=/dev/zero of=lut_left_y.bin bs=1M count=32 2>/dev/null
dd if=/dev/zero of=lut_right_x.bin bs=1M count=32 2>/dev/null
dd if=/dev/zero of=lut_right_y.bin bs=1M count=32 2>/dev/null
dd if=/dev/zero of=weight_left.bin bs=1M count=32 2>/dev/null
dd if=/dev/zero of=weight_right.bin bs=1M count=32 2>/dev/null

echo '{"width": 4096, "height": 2048}' > metadata.json

cd /home/nvidia/deep_cv_football/new_week

echo "✅ Dummy LUT файлы созданы в /tmp/empty_warps"
echo ""

# Запускаем пайплайн
GST_PLUGIN_PATH=/home/nvidia/deep_cv_football/my_steach:/home/nvidia/deep_cv_football/my_ring_buffer \
gst-launch-1.0 -v \
  videotestsrc pattern=smpte num-buffers=600 ! \
    video/x-raw,width=3840,height=2160,framerate=30/1 ! \
    nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_0 \
  videotestsrc pattern=ball num-buffers=600 ! \
    video/x-raw,width=3840,height=2160,framerate=30/1 ! \
    nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_1 \
  nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
  nvdsstitch name=stitch \
    left-source-id=0 \
    right-source-id=1 \
    panorama-width=4096 \
    panorama-height=2048 ! \
  tee name=t \
  \
  t. ! queue max-size-buffers=3 leaky=downstream ! \
     identity name=realtime silent=false ! \
     fakesink sync=false \
  \
  t. ! queue max-size-buffers=3 ! \
     ringbuffer max-buffers=210 delay-seconds=7.0 ! \
     identity name=delayed silent=false ! \
     fakesink sync=false \
  2>&1 | grep -E "Ring|identity|Warming|Sent|PTS" | head -100

echo ""
echo "✅ Тест завершён"
