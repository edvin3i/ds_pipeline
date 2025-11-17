#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlaybackPipelineBuilder class for playback pipeline creation.

Extracted from PanoramaWithVirtualCamera to improve modularity and testability.
Handles:
- Mode-specific playback pipelines (panorama/virtualcam/stream/record)
- Audio pipeline creation
- Encoding and output sinks
- Display probe and vcam probe attachment
"""

import logging
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logger = logging.getLogger(__name__)

# Import global constants
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900


class PlaybackPipelineBuilder:
    """Builds playback pipelines for different display modes."""

    def __init__(self,
                 display_mode="panorama",
                 bitrate=6000000,
                 stream_url=None,
                 stream_key=None,
                 output_file=None,
                 audio_device=None,
                 audio_appsink=None,
                 panorama_width=PANORAMA_WIDTH,
                 panorama_height=PANORAMA_HEIGHT):
        """
        Initialize PlaybackPipelineBuilder.

        Args:
            display_mode: "panorama", "virtualcam", "stream", or "record"
            bitrate: Video bitrate in bps (default: 6000000)
            stream_url: RTMP stream URL (for stream mode)
            stream_key: RTMP stream key (for stream mode)
            output_file: Output file path (for record/stream modes)
            audio_device: Audio device name (e.g., "pulse")
            audio_appsink: Audio appsink element from analysis pipeline
            panorama_width: Width of panorama (default: 5700)
            panorama_height: Height of panorama (default: 1900)
        """
        self.display_mode = display_mode
        self.bitrate = bitrate
        self.stream_url = stream_url
        self.stream_key = stream_key
        self.output_file = output_file
        self.audio_device = audio_device
        self.audio_appsink = audio_appsink
        self.panorama_width = panorama_width
        self.panorama_height = panorama_height

        # Pipeline elements (to be populated)
        self.playback_pipeline = None
        self.appsrc = None
        self.audio_appsrc = None
        self.vcam = None

    def create_playback_pipeline(self, on_appsrc_need_data_callback=None,
                                 vcam_update_probe_callback=None,
                                 playback_draw_probe_callback=None):
        """
        Создание playback pipeline.

        Args:
            on_appsrc_need_data_callback: Callback for appsrc need-data signal
            vcam_update_probe_callback: Callback for vcam update probe
            playback_draw_probe_callback: Callback for panorama draw probe

        Returns:
            dict: Dictionary with 'pipeline', 'appsrc', 'audio_appsrc', 'vcam' keys, or None on error
        """
        try:
            if self.display_mode == "stream":
                # Режим стриминга: с записью или без
                # ВАЖНО: FLV/RTMP требует H.264, H.265 не поддерживается
                pipeline_str = f"""
                appsrc name=src format=time is-live=true do-timestamp=true !
                video/x-raw,format=RGB !
                nvvideoconvert compute-hw=1 !
                video/x-raw(memory:NVMM),format=RGBA !
                nvdsvirtualcam name=vcam
                    output-width=1920
                    output-height=1080
                    panorama-width={self.panorama_width}
                    panorama-height={self.panorama_height}
                    yaw=0 pitch=10 roll=0 fov=68
                    auto-follow=true
                    smooth-factor=0.15 !
                video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080 !
                nvvideoconvert compute-hw=1 !
                video/x-raw(memory:NVMM),format=NV12 !
                nvv4l2h264enc
                    bitrate={self.bitrate}
                    preset-level=2
                    insert-sps-pps=1
                    iframeinterval=50
                    maxperf-enable=true !
                h264parse !
                """

                # Если нужна запись - добавляем tee для разделения потока
                if self.output_file:
                    pipeline_str += f"""
                tee name=t !
                queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
                flvmux name=flvmux streamable=true !
                rtmpsink
                    location={self.stream_url}{self.stream_key}
                    sync=false
                    async=false

                t. !
                queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
                flvmux streamable=true !
                filesink location={self.output_file} sync=false async=false
                """
                    logger.info(f"💾 Запись в FLV включена: {self.output_file}")
                else:
                    # Только стриминг без записи
                    pipeline_str += f"""
                queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
                flvmux name=flvmux streamable=true !
                rtmpsink
                    location={self.stream_url}{self.stream_key}
                    sync=false
                    async=false
                """

                # Добавляем аудио в зависимости от наличия микрофона
                if False and self.audio_device and self.audio_appsink:  # Проверяем что аудио было захвачено
                    # Используем буферизированное аудио
                    pipeline_str += """
                    appsrc name=audio_src
                        format=time
                        is-live=true
                        do-timestamp=false
                        block=false !
                    audio/x-raw,rate=44100,channels=2,format=S16LE,layout=interleaved !
                    audioconvert !
                    audioresample !
                    voaacenc bitrate=128000 !
                    aacparse !
                    queue max-size-buffers=100 !
                    flvmux.
                    """
                    logger.info("🎤 Используем буферизированное аудио")
                else:
                    # Если нет микрофона - тишина
                    pipeline_str += """
                    audiotestsrc wave=silence is-live=true !
                    audio/x-raw,rate=44100,channels=2 !
                    audioconvert !
                    voaacenc bitrate=128000 !
                    aacparse !
                    queue !
                    flvmux.
                    """
                    logger.warning("🔇 Микрофон не найден, используем тишину")

            elif self.display_mode == "record":
                # Режим только записи (без окна, без стрима)
                # Используем виртуальную камеру с кодированием напрямую в файл
                # УЛУЧШЕННЫЕ параметры качества (как у stream режима)

                # Выбор формата по расширению:
                # .flv = FLV (рекомендуется, как у YouTube)
                # .mkv = Matroska
                # .mp4 = MP4
                use_flv = self.output_file.endswith('.flv')
                use_mp4 = self.output_file.endswith('.mp4')

                # Выбираем мультиплексор
                if use_flv:
                    muxer = "flvmux streamable=true"
                elif use_mp4:
                    muxer = "mp4mux"
                else:
                    muxer = 'matroskamux streamable=false writing-app="DeepStream Football Tracker"'

                pipeline_str = f"""
                appsrc name=src format=time is-live=true do-timestamp=true !
                video/x-raw,format=RGB !
                nvvideoconvert compute-hw=1 !
                video/x-raw(memory:NVMM),format=RGBA !
                nvdsvirtualcam name=vcam
                    output-width=1920
                    output-height=1080
                    panorama-width={self.panorama_width}
                    panorama-height={self.panorama_height}
                    yaw=0 pitch=10 roll=0 fov=68
                    auto-follow=true
                    smooth-factor=0.15 !
                video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080 !
                nvvideoconvert compute-hw=1 !
                video/x-raw(memory:NVMM),format=NV12 !
                nvv4l2h265enc
                    bitrate={self.bitrate}
                    preset-level=2
                    insert-sps-pps=1
                    iframeinterval=50
                    maxperf-enable=true !
                h265parse !
                queue max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 !
                {muxer} !
                filesink location={self.output_file} sync=false async=false
                """
                bitrate_mbps = self.bitrate / 1000000.0
                logger.info(f"💾 Режим записи H.265 (HEVC): {self.output_file}")
                logger.info(f"⚡ Параметры: bitrate={bitrate_mbps:.1f}Mbps, preset=2, iframe=50")
                if use_flv:
                    logger.info(f"📦 Формат: FLV (рекомендуется, как у YouTube)")
                elif use_mp4:
                    logger.info(f"📦 Формат: MP4")
                else:
                    logger.info(f"📦 Формат: Matroska (MKV)")

            elif self.display_mode == "virtualcam":
                # Виртуальная камера для просмотра
                pipeline_str = f"""
                    appsrc name=src format=time is-live=true do-timestamp=true !
                    video/x-raw,format=RGB !
                    nvvideoconvert name=nvconv-pre compute-hw=1 !
                    video/x-raw(memory:NVMM),format=RGBA !
                    nvdsvirtualcam name=vcam
                        output-width=1920
                        output-height=1080
                        panorama-width={self.panorama_width}
                        panorama-height={self.panorama_height}
                        yaw=0 pitch=15 roll=0 fov=68
                        auto-follow=true
                        smooth-factor=0.15 !
                    nvvideoconvert !
                    video/x-raw,format=RGBA !
                    videoconvert !
                    xvimagesink sync=false
                """
            else:
                # Панорама с nvdsosd
                pipeline_str = """
                    appsrc name=src format=time is-live=true do-timestamp=true !
                    video/x-raw,format=RGB !
                    nvvideoconvert name=nvconv-pre compute-hw=1 !
                    video/x-raw(memory:NVMM),format=RGBA !
                    nvdsosd name=nvdsosd process-mode=0 !
                    nvvideoconvert name=nvconv-display compute-hw=1 nvbuf-memory-type=0 !
                    nveglglessink sync=false async=false enable-last-sample=false name=eglsink
                """

            # Создаем pipeline
            self.playback_pipeline = Gst.parse_launch(pipeline_str)

            # Настройка video appsrc
            self.appsrc = self.playback_pipeline.get_by_name("src")
            if self.appsrc:
                self.appsrc.set_property("is-live", True)
                self.appsrc.set_property("do-timestamp", True)
                self.appsrc.set_property("format", Gst.Format.TIME)
                if on_appsrc_need_data_callback:
                    self.appsrc.connect("need-data", on_appsrc_need_data_callback)
                logger.info("✅ Video appsrc настроен")

            # Настраиваем audio appsrc если есть (только для stream режима)
            if self.display_mode == "stream" and self.audio_device and self.audio_appsink:
                self.audio_appsrc = self.playback_pipeline.get_by_name("audio_src")
                if self.audio_appsrc:
                    # ВАЖНО: устанавливаем caps сразу
                    audio_caps = Gst.Caps.from_string(
                        "audio/x-raw,rate=44100,channels=2,format=S16LE,layout=interleaved"
                    )
                    self.audio_appsrc.set_property("caps", audio_caps)
                    self.audio_appsrc.set_property("is-live", True)
                    self.audio_appsrc.set_property("format", Gst.Format.TIME)
                    self.audio_appsrc.set_property("block", False)

                    # НЕ подключаем need-data, будем пушить из основного потока
                    logger.info("✅ Audio appsrc настроен")

            # Настройка виртуальной камеры или nvdsosd
            if self.display_mode in ["virtualcam", "stream", "record"]:
                self.vcam = self.playback_pipeline.get_by_name("vcam")
                if self.vcam and vcam_update_probe_callback:
                    sink_pad = self.vcam.get_static_pad("sink")
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, vcam_update_probe_callback, 0)
                    logger.info("✅ Добавлен vcam_update_probe")
            else:
                nvdsosd = self.playback_pipeline.get_by_name("nvdsosd")
                if nvdsosd and playback_draw_probe_callback:
                    sink_pad = nvdsosd.get_static_pad("sink")
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, playback_draw_probe_callback, 0)
                    logger.info("✅ Добавлен playback_draw_probe")

            # Логирование конфигурации
            if self.display_mode == "stream":
                bitrate_mbps = self.bitrate / 1000000.0
                logger.info(f"🚀 Playback pipeline создан для стриминга")
                logger.info(f"📡 URL: {self.stream_url}")
                if self.stream_key:
                    logger.info(f"🔑 Ключ: {self.stream_key[:4]}...{self.stream_key[-4:]}")
                logger.info(f"⚡ Качество видео: {bitrate_mbps:.1f} Mbps")
                if self.audio_device:
                    logger.info(f"🎤 Аудио: {self.audio_device}")
                else:
                    logger.info(f"🔇 Аудио: тишина")
            else:
                logger.info(f"✅ Playback pipeline создан для режима: {self.display_mode}")

            return {
                'pipeline': self.playback_pipeline,
                'appsrc': self.appsrc,
                'audio_appsrc': self.audio_appsrc,
                'vcam': self.vcam
            }

        except Exception as e:
            logger.error(f"❌ create_playback_pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None
