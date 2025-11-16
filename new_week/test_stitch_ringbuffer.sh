#!/bin/bash

# Полный тест: nvdsstitch + ringbuffer с реальными камерами
# Показывает PTS в обеих ветках для проверки задержки

cd /home/nvidia/deep_cv_football/new_week

export GST_PLUGIN_PATH=/home/nvidia/deep_cv_football/my_steach:/home/nvidia/deep_cv_football/my_ring_buffer
export GST_DEBUG=ringbuffer:4,nvdsstitch:3

echo "========================================="
echo "🧪 ПОЛНЫЙ ТЕСТ: stitch + ringbuffer"
echo "========================================="
echo ""
echo "📹 Источник: left.mp4 + right.mp4"
echo "🎨 Stitch: панорама 4096x2048 с LUT"
echo "🔀 TEE: 2 ветки"
echo "   ├─ АНАЛИЗ (realtime PTS)"
echo "   └─ DISPLAY (delayed PTS ~7 сек)"
echo ""
echo "⏱️  Смотрим разницу PTS между ветками"
echo "========================================="
echo ""

gst-launch-1.0 -v \
  filesrc location=left.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
    nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160' ! mux.sink_0 \
  filesrc location=right.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
    nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160' ! mux.sink_1 \
  nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
  nvdsstitch name=stitch \
    left-source-id=0 \
    right-source-id=1 \
    panorama-width=4096 \
    panorama-height=2048 ! \
  tee name=t \
  \
  t. ! queue name=analysis_q max-size-buffers=3 leaky=downstream ! \
     identity name=analysis_pts silent=false ! \
     fakesink name=analysis_sink sync=false \
  \
  t. ! queue name=display_q max-size-buffers=3 ! \
     ringbuffer name=rb max-buffers=210 delay-seconds=7.0 ! \
     identity name=display_pts silent=false ! \
     fakesink name=display_sink sync=false

echo ""
echo "========================================="
echo "✅ Тест завершён"
echo ""
echo "Для анализа PTS смотрите на строки:"
echo "  analysis_pts:  PTS в реальном времени"
echo "  display_pts:   PTS с задержкой ~7 сек"
echo "========================================="
