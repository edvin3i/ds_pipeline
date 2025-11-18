#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PipelineBuilder class for main analysis pipeline creation.

Extracted from PanoramaWithVirtualCamera to improve modularity and testability.
Handles:
- Source → Stitch → Tee → Analysis pipeline
- MIPI camera and file sources
- nvtilebatcher and nvinfer configuration
- Analysis probe attachment
"""

import time
import logging
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logger = logging.getLogger(__name__)

# Import global constants
# These should be imported from the main module or passed as configuration
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900
TILE_OFFSET_Y = 434
TILE_OFFSET_X = 192
TILES_COUNT = 6
TILE_WIDTH = 1024
TILE_HEIGHT = 1024

TILE_POSITIONS = [
    (TILE_OFFSET_X,                   TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 0
    (TILE_OFFSET_X + TILE_WIDTH,      TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 1
    (TILE_OFFSET_X + TILE_WIDTH * 2,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 2
    (TILE_OFFSET_X + TILE_WIDTH * 3,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 3
    (TILE_OFFSET_X + TILE_WIDTH * 4,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 4
    (TILE_OFFSET_X + TILE_WIDTH * 5,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 5
]


class PipelineBuilder:
    """Builds the main analysis pipeline with source, stitching, and inference."""

    def __init__(self,
                 source_type="files",
                 video1="left1.mp4",
                 video2="right1.mp4",
                 config_path="config_infer.txt",
                 framerate=30,
                 buffer_duration=5.0,
                 enable_display=True,
                 enable_analysis=True,
                 analysis_skip_interval=5,
                 panorama_width=PANORAMA_WIDTH,
                 panorama_height=PANORAMA_HEIGHT):
        """
        Initialize PipelineBuilder.

        Args:
            source_type: "cameras" or "files"
            video1: Path to left video or camera sensor ID
            video2: Path to right video or camera sensor ID
            config_path: Path to nvinfer config file
            framerate: Video framerate (default: 30)
            buffer_duration: Buffer duration in seconds (default: 5.0)
            enable_display: Enable display sink (default: True)
            enable_analysis: Enable analysis branch (default: True)
            analysis_skip_interval: Skip frames for analysis (default: 5)
            panorama_width: Width of stitched panorama (default: 5700)
            panorama_height: Height of stitched panorama (default: 1900)
        """
        self.source_type = source_type
        self.video1 = video1
        self.video2 = video2
        self.config_path = config_path
        self.framerate = framerate
        self.buffer_duration = buffer_duration
        self.enable_display = enable_display
        self.enable_analysis = enable_analysis
        self.analysis_skip_interval = max(1, int(analysis_skip_interval))
        self.panorama_width = panorama_width
        self.panorama_height = panorama_height

        # Audio device detection
        self.audio_device = None

        # Pipeline elements (to be populated)
        self.pipeline = None
        self.appsink = None
        self.audio_appsink = None

    def find_usb_audio_device(self):
        """Проверка доступности аудио захвата через PulseAudio."""
        try:
            # Используем PulseAudio вместо прямого доступа к hw:0,0
            test_pipe = """
                pulsesrc !
                audioconvert !
                audio/x-raw,format=S16LE,rate=44100,channels=2 !
                fakesink
            """
            test = Gst.parse_launch(test_pipe)
            test.set_state(Gst.State.PLAYING)
            time.sleep(0.2)
            state = test.get_state(0.1)
            test.set_state(Gst.State.NULL)

            if state[0] == Gst.StateChangeReturn.SUCCESS:
                self.audio_device = "pulse"  # Изменено с hw:0,0 на pulse
                logger.info("🎤 Микрофон готов через PulseAudio")
                return True

            logger.warning("⚠️ Микрофон не доступен")
            self.audio_device = None
            return False

        except Exception as e:
            logger.error(f"Ошибка проверки микрофона: {e}")
            self.audio_device = None
            return False

    def create_pipeline(self, on_new_sample_callback=None, on_new_audio_sample_callback=None,
                       frame_skip_probe_callback=None, analysis_probe_callback=None):
        """
        Создание основного pipeline с поддержкой камер и файлов.

        Args:
            on_new_sample_callback: Callback for video appsink new-sample signal
            on_new_audio_sample_callback: Callback for audio appsink new-sample signal
            frame_skip_probe_callback: Callback for frame skip probe
            analysis_probe_callback: Callback for analysis probe

        Returns:
            dict: Dictionary with 'pipeline', 'appsink', 'audio_appsink' keys, or None on error
        """
        try:
            buffer_size = int(self.framerate * self.buffer_duration)
            buffer_time_ns = int(self.buffer_duration * 1e9)

            # Определяем источники в зависимости от типа
            if self.source_type == "cameras":
                # Используем nvarguscamerasrc для камер
                left_cam = int(self.video1)
                right_cam = int(self.video2)

                logger.info(f"📷 Используем камеры: левая={left_cam}, правая={right_cam}")

                sources_str = f"""
                    nvarguscamerasrc sensor-id={left_cam} !
                    video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_0

                    nvarguscamerasrc sensor-id={right_cam} !
                    video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12 !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_1
                """

                # Для камер используем live-source=1
                mux_config = """
                    nvstreammux name=mux
                        batch-size=2
                        width=3840
                        height=2160
                        live-source=1
                        batched-push-timeout=33333 !
                """
            else:
                # Используем filesrc для файлов
                logger.info(f"📁 Используем файлы: {self.video1}, {self.video2}")

                sources_str = f"""
                    filesrc location={self.video1} !
                    decodebin !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_0

                    filesrc location={self.video2} !
                    decodebin !
                    nvvideoconvert !
                    video/x-raw(memory:NVMM),format=RGBA,width=3840,height=2160 !
                    queue max-size-buffers=4 leaky=downstream !
                    mux.sink_1
                """

                # Для файлов используем live-source=0
                mux_config = """
                    nvstreammux name=mux
                        batch-size=2
                        width=3840
                        height=2160
                        live-source=0
                        batched-push-timeout=40000 !
                """

            # Общая часть pipeline
            common_str = f"""
                nvdsstitch
                    left-source-id=0
                    right-source-id=1
                    gpu-id=0
                    use-egl=true
                    panorama-width={self.panorama_width}
                    panorama-height={self.panorama_height} !

                tee name=main_tee
            """

            # Базовый pipeline
            pipeline_str = sources_str + mux_config + common_str

            # Ветка дисплея с буферизацией для обоих режимов
            # NVMM ZERO-COPY: Buffers stay in GPU memory throughout (no CPU conversion)
            if self.enable_display:
                pipeline_str += f"""
                    main_tee. !
                    queue name=display_queue
                        max-size-buffers={buffer_size}
                        max-size-time={buffer_time_ns}
                        leaky=0 !
                    identity name=display_passthrough !
                    capsfilter caps="video/x-raw(memory:NVMM),format=RGBA,width={self.panorama_width},height={self.panorama_height}" !
                    appsink name=display_sink emit-signals=true sync=false drop=false max-buffers=60 wait-on-eos=true
                """

                # ДОБАВЛЯЕМ ЗАХВАТ АУДИО
                # Проверяем наличие USB микрофона
                if self.find_usb_audio_device():
                    # Используем pulsesrc вместо alsasrc
                    pipeline_str += f"""
                        pulsesrc name=audio_source !
                        audioconvert !
                        audioamplify amplification=2.0 !
                        audioresample !
                        audio/x-raw,format=S16LE,rate=44100,channels=2 !
                        queue name=audio_queue
                            max-size-buffers={buffer_size}
                            max-size-time={buffer_time_ns}
                            leaky=0 !
                        appsink name=audio_sink
                            emit-signals=true
                            sync=false
                            drop=false
                            max-buffers={buffer_size}
                    """
                    logger.info("🎤 Добавлен захват аудио через PulseAudio")
                else:
                    logger.warning("⚠️ Аудио устройство не найдено, стрим будет без звука")

            # Ветка анализа
            if self.enable_analysis:
                pipeline_str += """
                    main_tee. !
                    queue name=analysis_queue max-size-buffers=2 leaky=downstream !
                    tee name=tiles_tee
                """

            logger.info(f"Создаём основной pipeline для источника: {self.source_type}")
            self.pipeline = Gst.parse_launch(pipeline_str)

            # CRITICAL: Configure buffer pools for NVMM zero-copy buffering
            # Required to accommodate 7-second buffer (210 frames @ 30fps)
            nvdsstitch = self.pipeline.get_by_name("nvdsstitch")
            if nvdsstitch and nvdsstitch.find_property("num-extra-surfaces"):
                nvdsstitch.set_property("num-extra-surfaces", 64)
                logger.info("[NVMM-BUFFER-POOL] nvdsstitch: added 64 extra surfaces")

            display_queue = self.pipeline.get_by_name("display_queue")
            if display_queue:
                # Increase queue buffer capacity for 7s buffering
                display_queue.set_property("max-size-buffers", 250)  # 7s @ 30fps + margin
                logger.info("[NVMM-BUFFER-POOL] display_queue: max-size-buffers=250")

            # Подключаем video appsink для буферизации
            if self.enable_display:
                self.appsink = self.pipeline.get_by_name("display_sink")
                if self.appsink and on_new_sample_callback:
                    self.appsink.set_property("emit-signals", True)
                    self.appsink.connect("new-sample", on_new_sample_callback)
                    logger.info("✅ Video appsink подключен")

                # Подключаем audio appsink если есть
                self.audio_appsink = self.pipeline.get_by_name("audio_sink")
                if self.audio_appsink and on_new_audio_sample_callback:
                    self.audio_appsink.set_property("emit-signals", True)
                    self.audio_appsink.connect("new-sample", on_new_audio_sample_callback)
                    logger.info("✅ Audio appsink подключен")

            # Создаем тайлы для анализа если нужно
            if self.enable_analysis:
                self._create_analysis_tiles(frame_skip_probe_callback, analysis_probe_callback)

            logger.info("✅ Основной pipeline создан успешно")

            return {
                'pipeline': self.pipeline,
                'appsink': self.appsink,
                'audio_appsink': self.audio_appsink,
                'audio_device': self.audio_device
            }

        except Exception as e:
            logger.error(f"❌ Ошибка create_pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_analysis_tiles(self, frame_skip_probe_callback=None, analysis_probe_callback=None):
        """
        Создание 6 тайлов для анализа.

        Args:
            frame_skip_probe_callback: Callback for frame skip probe
            analysis_probe_callback: Callback for analysis probe after inference
        """
        tiles_tee = self.pipeline.get_by_name("tiles_tee")
        if not tiles_tee:
            logger.error("tiles_tee не найден")
            return

        # Identity для пропуска кадров
        frame_filter = Gst.ElementFactory.make("identity", "frame-filter")
        frame_filter.set_property("sync", False)
        self.pipeline.add(frame_filter)

        tee_src = tiles_tee.request_pad_simple("src_%u")
        filter_sink = frame_filter.get_static_pad("sink")
        tee_src.link(filter_sink)

        filter_src = frame_filter.get_static_pad("src")
        if frame_skip_probe_callback:
            filter_src.add_probe(Gst.PadProbeType.BUFFER, frame_skip_probe_callback, 0)
            logger.info(f"Добавлен frame_skip_probe (каждый {self.analysis_skip_interval}-й кадр)")

        # ============================================================
        # НОВЫЙ КОД: nvtilebatcher вместо filtered_tee + 6×crop + mux
        # ============================================================

        # Создаем nvtilebatcher плагин
        tilebatcher = Gst.ElementFactory.make("nvtilebatcher", "tilebatcher")
        if not tilebatcher:
            logger.error("❌ nvtilebatcher плагин не найден!")
            logger.error("Установите плагин: cd /home/nvidia/deep_cv_football/my_tile_batcher/src && make install")
            return

        # Настройки плагина
        tilebatcher.set_property("gpu-id", 0)
        tilebatcher.set_property("panorama-width", self.panorama_width)
        tilebatcher.set_property("panorama-height", self.panorama_height)
        tilebatcher.set_property("tile-offset-y", TILE_OFFSET_Y)  # Вертикальный offset из field_mask.png
        # 6 тайлов БЕЗ ПРОПУСКОВ, вырезаются на основе field_mask.png
        # Y позиция: не симметричное центрирование, а рассчитанное из маски поля!

        self.pipeline.add(tilebatcher)

        # Связываем: frame_filter → tilebatcher
        frame_filter.link(tilebatcher)

        logger.info(f"✅ nvtilebatcher создан ({TILES_COUNT} тайлов БЕЗ пропусков из центра)")
        logger.info(f"   Координаты тайлов (отступ по бокам {TILE_OFFSET_X}px, вертикальный {TILE_OFFSET_Y}px):")
        for tile_id, (x, y, w, h) in enumerate(TILE_POSITIONS):
            logger.info(f"   Тайл {tile_id}: x={x}, y={y}, size={w}×{h}")

        # ============================================================
        # nvinfer напрямую после tilebatcher (БЕЗ nvstreammux!)
        # ============================================================

        pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
        pgie.set_property("config-file-path", self.config_path)
        pgie.set_property("batch-size", 6)  # ВАЖНО: должно соответствовать TILES_PER_BATCH
        pgie.set_property("gpu-id", 0)
        self.pipeline.add(pgie)

        # Связываем: tilebatcher → nvinfer
        tilebatcher.link(pgie)

        logger.info("✅ nvinfer подключен после nvtilebatcher")

        # fakesink
        sink_inf = Gst.ElementFactory.make("fakesink", "sink-infer")
        sink_inf.set_property("sync", False)
        sink_inf.set_property("async", False)
        self.pipeline.add(sink_inf)
        pgie.link(sink_inf)

        # Probe после инференса
        pgie_src = pgie.get_static_pad("src")
        if pgie_src and analysis_probe_callback:
            pgie_src.add_probe(Gst.PadProbeType.BUFFER, analysis_probe_callback, 0)
            logger.info("Добавлен analysis_probe")
