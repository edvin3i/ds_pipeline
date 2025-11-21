#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЭКСПЕРИМЕНТАЛЬНАЯ ВЕРСИЯ С МУЛЬТИКЛАССОВОЙ ДЕТЕКЦИЕЙ - REFACTORED

Панорама с двумя режимами отображения и записью:
- panorama: полная панорама с отрисовкой bbox через nvdsosd
- virtualcam: виртуальная камера, следящая за мячом (с возможностью записи)
- stream: стриминг на stream
Поддержка источников: файлы или камеры MIPI CSI

=== МУЛЬТИКЛАССОВЫЕ ВОЗМОЖНОСТИ ===
1. Детекция 5 классов: ball, player, staff, side_referee, main_referee
2. Хранение истории игроков для расчёта центра масс
3. Fallback: при потере мяча камера центрируется на игроках
4. Визуализация в panorama режиме:
   - Мяч: КРАСНЫЙ цвет (border=3)
   - Игроки: ЗЕЛЁНЫЙ цвет (border=2)
   - Лимит отрисовки: 16 объектов (ограничение nvdsosd на Jetson)
   - Тайлы ОТКЛЮЧЕНЫ для экономии слотов

ПРИОРИТЕТ ОТРИСОВКИ: мяч → игроки (персонал и судьи отключены)

=== REFACTORED VERSION ===
This version uses modular architecture with delegation:
- utils: FieldMaskBinary, CSV logging, NMS
- core: HistoryManager (replaces BallDetectionHistory), PlayersHistory
- processing: TensorProcessor, AnalysisProbeHandler
- rendering: VirtualCameraProbeHandler, DisplayProbeHandler
- pipeline: ConfigBuilder, PipelineBuilder, PlaybackPipelineBuilder, BufferManager
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
import ctypes
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
import logging
import time
import math
import threading
import csv
import cv2

# ============================================================
# Import all extracted modules
# ============================================================
# Utilities
from utils import FieldMaskBinary, save_detection_to_csv, apply_nms

# Core detection and history management
from core import HistoryManager, PlayersHistory

# Processing (YOLO inference and analysis)
from processing import TensorProcessor, AnalysisProbeHandler

# Rendering (virtual camera and display)
from rendering import VirtualCameraProbeHandler, DisplayProbeHandler

# Pipeline builders and buffer management
from pipeline import ConfigBuilder, PipelineBuilder, PlaybackPipelineBuilder, BufferManager

# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("panorama-virtualcam")

# ============================================================
# Инициализация GStreamer
# ============================================================
# Плагины устанавливаются в ~/.local/share/gstreamer-1.0/plugins/
# GStreamer найдёт их автоматически при инициализации
Gst.init(None)


# ============================================================
# КОНСТАНТЫ КОНФИГУРАЦИИ ПАНОРАМЫ
# ============================================================
# Размеры панорамы (обновлено с 1632 на 1800 для поддержки FOV до 75°)
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

# Параметры тайлов для nvtilebatcher
TILE_WIDTH = 1024
TILE_HEIGHT = 1024
TILES_COUNT = 6

# Вертикальный отступ тайлов: рассчитан АВТОМАТИЧЕСКИ на основе маски поля
# Логика: находим центр поля (field_top + field_bottom)/2, вычитаем половину тайла (512)
# Field bounds: top=438, bottom=1454 → center=946 → offset = 946 - 512 = 434
TILE_OFFSET_Y = 434  # Рассчитано из field_mask.png (было: 304 симметричное)
TILE_OFFSET_X = 192  # Горизонтальный margin для 6 тайлов (6×1024=6144, margin=(6528-6144)/2)

# Координаты тайлов (автоматически вычисляются)
TILE_POSITIONS = [
    (TILE_OFFSET_X,                   TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 0
    (TILE_OFFSET_X + TILE_WIDTH,      TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 1
    (TILE_OFFSET_X + TILE_WIDTH * 2,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 2
    (TILE_OFFSET_X + TILE_WIDTH * 3,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 3
    (TILE_OFFSET_X + TILE_WIDTH * 4,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 4
    (TILE_OFFSET_X + TILE_WIDTH * 5,  TILE_OFFSET_Y, TILE_WIDTH, TILE_HEIGHT),  # Tile 5
]


# ============================================================
# MAIN ORCHESTRATOR CLASS (REFACTORED)
# ============================================================

class PanoramaWithVirtualCamera:
    """
    Панорама с двумя режимами отображения, записью и единой буферизацией.

    REFACTORED VERSION:
    This class now focuses on orchestration and delegates functionality to specialized handlers:

    Delegation Map:
    ---------------
    ConfigBuilder: Inference config generation
    PipelineBuilder: Analysis pipeline construction
    PlaybackPipelineBuilder: Playback pipeline construction
    BufferManager: Frame/audio buffering and playback management
    AnalysisProbeHandler: YOLO tensor processing and detection
    VirtualCameraProbeHandler: Virtual camera control and ball tracking
    DisplayProbeHandler: Panorama rendering with bboxes
    HistoryManager: Ball detection history (replaces BallDetectionHistory)
    PlayersHistory: Player detection history for fallback
    TensorProcessor: YOLO output tensor processing
    FieldMaskBinary: Field mask validation
    """

    def __init__(self,
                # Источники видео
                source_type: str = "files",
                video1: str = "left1.mp4",
                video2: str = "right1.mp4",
                config_path: str = None,
                buffer_duration: float = 5.0,
                enable_display: bool = True,
                display_mode: str = "panorama",  # "panorama", "virtualcam", "stream", "record"
                enable_analysis: bool = True,
                analysis_skip_interval: int = 5,
                confidence_threshold: float = 0.35,
                auto_zoom: bool = True,
                stream_key: str = None,
                stream_url: str = None,
                output_file: str = None,
                bitrate: int = 6000000):

        # Store configuration
        self.source_type = source_type
        self.video1 = video1
        self.video2 = video2
        self.buffer_duration = float(buffer_duration)
        self.enable_display = enable_display
        self.display_mode = display_mode
        self.enable_analysis = enable_analysis
        self.confidence_threshold = confidence_threshold
        self.auto_zoom = auto_zoom
        self.stream_key = stream_key
        self.stream_url = stream_url
        self.output_file = output_file
        self.bitrate = bitrate

        # Panorama dimensions (global constants)
        self.panorama_width = PANORAMA_WIDTH
        self.panorama_height = PANORAMA_HEIGHT

        # ROI configuration - use pre-calculated tile positions
        self.roi_configs = TILE_POSITIONS

        # ============================================================
        # DELEGATED: Field mask validation → FieldMaskBinary (utils)
        # ============================================================
        self.field_mask = FieldMaskBinary(
            mask_path='field_mask.png',
            panorama_width=self.panorama_width,
            panorama_height=self.panorama_height
        )

        # ============================================================
        # DELEGATED: Ball detection history → HistoryManager (core)
        # Replaces: BallDetectionHistory
        # ============================================================
        self.history = HistoryManager(history_duration=10.0, cleanup_interval=1000)

        # ============================================================
        # DELEGATED: Players history → PlayersHistory (core)
        # ============================================================
        self.players_history = PlayersHistory(history_duration=10.0)

        # ============================================================
        # DELEGATED: Tensor processing → TensorProcessor (processing)
        # ============================================================
        self.tensor_processor = TensorProcessor(conf_thresh=confidence_threshold)

        # Adaptive filter state (kept in main class for now)
        self.last_ball_position = None
        self.frames_without_reliable_detection = 0

        # All detections storage for rendering (synced by timestamp)
        self.all_detections_history = {}  # {timestamp: {'ball': [...], 'player': [...], ...}}

        # EMA smoothing for player center of mass
        self.players_center_mass_smoothed = None  # (x, y) - smoothed position
        self.players_center_mass_alpha = 0.18  # Smoothing coefficient

        # Raw position buffer for detecting back-and-forth patterns
        self.players_center_mass_history = []  # [(x, y), ...] last 10 raw positions
        self.players_center_mass_history_max = 10

        # Virtual camera element reference
        self.vcam = None

        # Statistics
        self.display_frame_count = 0
        self.analysis_frame_count = 0
        self.analysis_skip_counter = 0
        self.analysis_skip_interval = max(1, int(analysis_skip_interval))
        self.analysis_actual_frame = 0
        self.detection_count = 0
        self.start_time = None
        self.current_fps = 0.0

        # Timestamp for backward interpolation
        self.current_display_timestamp = 0.0  # Current playback timestamp

        # ============================================================
        # DELEGATED: Buffer management → BufferManager (pipeline)
        # ============================================================
        self.framerate = 30
        self.buffer_manager = BufferManager(
            buffer_duration=buffer_duration,
            framerate=self.framerate
        )

        # Keep references for compatibility (delegated to BufferManager)
        self.appsink = None
        self.appsrc = None
        self.audio_appsrc = None
        self.audio_appsink_analysis = None  # Audio appsink from analysis pipeline
        self.audio_device = None  # Audio device (e.g., "pulse")
        self.playback_pipeline = None

        # Pipelines
        self.pipeline = None
        self.loop = GLib.MainLoop()

        # ============================================================
        # DELEGATED: Config generation → ConfigBuilder (pipeline)
        # ============================================================
        self.config_builder = ConfigBuilder()
        self.config_path = config_path or self.config_builder.create_inference_config()

        # Speed zoom settings
        self.speed_zoom_enabled = True
        self.speed_history = deque(maxlen=5)
        self.last_speed_calc_time = 0
        self.last_speed_calc_pos = None
        self.current_smooth_speed = 0.0
        self.speed_zoom_factor = 1.6

        # Ball radius interpolation for smooth zoom
        self.smooth_ball_radius = 20.0
        self.radius_smooth_factor = 0.3

        # Ball loss behavior parameters
        self.ball_lost = False
        self.ball_lost_frames = 0
        self.last_known_position = None
        self.lost_ball_fov_rate = 2.0
        self.max_search_fov = 90.0
        self.ball_recovery_frames = 6

        # Speed thresholds (pixels/sec)
        self.speed_low_threshold = 300.0
        self.speed_high_threshold = 1200.0
        self.speed_zoom_max_factor = 3.0
        self.speed_smoothing = 0.3

        # ============================================================
        # DELEGATED: Display rendering → DisplayProbeHandler (rendering)
        # ============================================================
        self.display_probe_handler = DisplayProbeHandler(
            ball_history=self.history,
            players_history=self.players_history,
            all_detections_history=self.all_detections_history,
            display_mode=self.display_mode
        )

        # ============================================================
        # DELEGATED: Virtual camera → VirtualCameraProbeHandler (rendering)
        # Will be initialized after pipeline creation when vcam element is available
        # ============================================================
        self.vcam_probe_handler = None

        # ============================================================
        # DELEGATED: Analysis probe → AnalysisProbeHandler (processing)
        # Will be initialized after we have all required references
        # ============================================================
        self.analysis_probe_handler = None

        # Clean up old log files
        for log_file in ['ball_events.tsv', 'ball_raw_future.csv', 'ball_display_used.csv']:
            if os.path.exists(log_file):
                os.remove(log_file)
                logger.info(f"Удален старый лог: {log_file}")

    def frame_skip_probe(self, pad, info, u_data):
        """
        Frame skip probe for analysis pipeline.

        KEPT IN MAIN CLASS: Simple frame counting logic for skip interval.
        """
        self.analysis_skip_counter += 1
        if self.analysis_skip_counter % self.analysis_skip_interval != 0:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def create_pipeline(self) -> bool:
        """
        Create the main analysis pipeline.

        DELEGATED TO: PipelineBuilder (pipeline module)
        """
        # Initialize pipeline builder (roi_configs removed - handled internally by builder)
        pipeline_builder = PipelineBuilder(
            source_type=self.source_type,
            video1=self.video1,
            video2=self.video2,
            config_path=self.config_path,
            panorama_width=self.panorama_width,
            panorama_height=self.panorama_height,
            buffer_duration=self.buffer_duration,
            framerate=self.framerate
        )

        # Build pipeline - use correct method name
        result = pipeline_builder.create_pipeline()
        if not result:
            return False

        # Extract pipeline and elements
        self.pipeline = result['pipeline']
        self.appsink = result['appsink']

        # Store audio_appsink if available
        self.audio_appsink_analysis = result.get('audio_appsink')
        self.audio_device = result.get('audio_device')

        # ============================================================
        # DELEGATED: Analysis probe → AnalysisProbeHandler (processing)
        # ============================================================
        # Now we can initialize the analysis probe handler with all dependencies
        self.analysis_probe_handler = AnalysisProbeHandler(
            ball_history=self.history,
            players_history=self.players_history,
            field_mask=self.field_mask,
            tensor_processor=self.tensor_processor,
            roi_configs=self.roi_configs,
            all_detections_history=self.all_detections_history,
            panorama_width=self.panorama_width,
            panorama_height=self.panorama_height
        )

        # Connect frame skip probe - get element by name
        frame_filter = self.pipeline.get_by_name("frame-filter")
        if frame_filter:
            filter_src_pad = frame_filter.get_static_pad("src")
            if filter_src_pad:
                filter_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.frame_skip_probe, None)
                logger.info("✓ Frame skip probe connected")

        # Connect analysis probe - get nvinfer by name
        nvinfer = self.pipeline.get_by_name("primary-infer")
        if nvinfer:
            nvinfer_src_pad = nvinfer.get_static_pad("src")
            if nvinfer_src_pad:
                nvinfer_src_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self.analysis_probe_handler.handle_analysis_probe,
                    None
                )
                logger.info("✓ Analysis probe connected")

        # ============================================================
        # DELEGATED: Buffer sink → BufferManager (pipeline)
        # ============================================================
        # Connect appsink callback to buffer manager
        if self.appsink:
            self.appsink.set_property("emit-signals", True)
            self.appsink.connect("new-sample", self.buffer_manager.on_new_sample)
            logger.info("✓ Video appsink connected to buffer manager")

        logger.info("✓ Analysis pipeline created successfully")
        return True

    def create_playback_pipeline(self) -> bool:
        """
        Create the playback pipeline for delayed display.

        DELEGATED TO: PlaybackPipelineBuilder (pipeline module)
        """
        # Initialize playback builder (framerate and auto_zoom removed - not in __init__)
        # Pass audio_device and audio_appsink from analysis pipeline
        playback_builder = PlaybackPipelineBuilder(
            display_mode=self.display_mode,
            panorama_width=self.panorama_width,
            panorama_height=self.panorama_height,
            stream_url=self.stream_url,
            stream_key=self.stream_key,
            output_file=self.output_file,
            bitrate=self.bitrate,
            audio_device=self.audio_device,
            audio_appsink=self.audio_appsink_analysis
        )

        # Build pipeline - use correct method name
        result = playback_builder.create_playback_pipeline()
        if not result:
            return False

        # Extract pipeline and elements
        self.playback_pipeline = result['pipeline']
        self.appsrc = result['appsrc']
        self.audio_appsrc = result.get('audio_appsrc')
        self.vcam = result.get('vcam')

        # ============================================================
        # DELEGATED: Virtual camera control → VirtualCameraProbeHandler (rendering)
        # ============================================================
        if self.vcam and self.display_mode in ['virtualcam', 'stream', 'record']:
            self.vcam_probe_handler = VirtualCameraProbeHandler(
                ball_history=self.history,
                players_history=self.players_history,
                all_detections_history=self.all_detections_history,
                vcam=self.vcam
            )

            # Connect vcam probe
            vcam_sink_pad = self.vcam.get_static_pad("sink")
            if vcam_sink_pad:
                vcam_sink_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self.vcam_probe_handler.handle_vcam_update_probe,
                    None
                )
                logger.info("✓ Virtual camera probe connected")

        # ============================================================
        # DELEGATED: Display probe → DisplayProbeHandler (rendering)
        # ============================================================
        # Connect display probe for panorama rendering (get osd by name)
        if self.display_mode == 'panorama':
            osd = self.playback_pipeline.get_by_name("nvdsosd")
            if osd:
                osd_sink_pad = osd.get_static_pad("sink")
                if osd_sink_pad:
                    osd_sink_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self.display_probe_handler.handle_playback_draw_probe,
                        None
                    )
                    logger.info("✓ Display probe connected to nvdsosd")

        # ============================================================
        # DELEGATED: appsrc callbacks → BufferManager (pipeline)
        # ============================================================
        if self.appsrc:
            self.appsrc.set_property("emit-signals", True)
            self.appsrc.connect("need-data", self.buffer_manager._on_appsrc_need_data)
            logger.info("✓ Video appsrc connected to buffer manager")

        if self.audio_appsrc:
            self.audio_appsrc.set_property("emit-signals", True)
            self.audio_appsrc.connect("need-data", self.buffer_manager._on_audio_appsrc_need_data)
            logger.info("✓ Audio appsrc connected to buffer manager")

        # Store references in buffer manager
        self.buffer_manager.set_elements(
            appsrc=self.appsrc,
            audio_appsrc=self.audio_appsrc,
            playback_pipeline=self.playback_pipeline
        )

        logger.info("✓ Playback pipeline created successfully")
        return True

    def _on_bus_message(self, bus, message):
        """
        Handle GStreamer bus messages.

        KEPT IN MAIN CLASS: Core orchestration logic for error handling and EOS.
        """
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[BUS] ERROR: {err}; debug: {debug}")
            self.stop()
        elif t == Gst.MessageType.EOS:
            logger.info("[BUS] EOS")
            self.stop()
        return True

    def run(self) -> bool:
        """
        Start the application.

        KEPT IN MAIN CLASS: Core orchestration - creates pipelines, starts threads, runs main loop.
        """
        if not self.create_pipeline():
            return False

        if not self.create_playback_pipeline():
            return False

        if self.display_mode == "stream":
            logger.info(f"🚀 Запуск stream стриминга с виртуальной камерой")
            logger.info(f"🔑 Ключ: {self.stream_key[:4]}...{self.stream_key[-4:]}")
            logger.info(f"📺 URL: {self.stream_url}")
            logger.info(f"📷 Виртуальная камера будет следить за мячом")
        else:
            logger.info(f"Запуск основного пайплайна в режиме {self.display_mode}…")

        # Connect bus handlers
        main_bus = self.pipeline.get_bus()
        main_bus.add_signal_watch()
        main_bus.connect("message", self._on_bus_message)

        pb_bus = self.playback_pipeline.get_bus()
        pb_bus.add_signal_watch()
        pb_bus.connect("message", self._on_bus_message)

        # Start pipelines
        logger.info(f"Запуск основного пайплайна в режиме {self.display_mode}…")
        self.pipeline.set_state(Gst.State.PLAYING)

        self.start_time = time.time()

        # ============================================================
        # DELEGATED: Buffer loop → BufferManager (pipeline)
        # ============================================================
        self.buffer_manager.start_buffer_thread()

        try:
            logger.info("Главный цикл запущен. Нажмите Ctrl+C для выхода.")
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Остановлено пользователем (Ctrl+C).")
        finally:
            self.stop()

        return True

    def stop(self):
        """
        Clean shutdown.

        KEPT IN MAIN CLASS: Core orchestration for cleanup.
        DELEGATED: Buffer thread management → BufferManager
        """
        # ============================================================
        # DELEGATED: Stop buffer thread → BufferManager (pipeline)
        # ============================================================
        self.buffer_manager.stop_buffer_thread()

        # Send EOS to appsrc if available
        try:
            if self.appsrc:
                self.appsrc.emit("end-of-stream")
        except:
            pass

        # Stop playback pipeline
        try:
            if self.playback_pipeline:
                self.playback_pipeline.set_state(Gst.State.NULL)
        except:
            pass

        # Stop analysis pipeline
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
        except:
            pass

        # Stop main loop
        try:
            if self.loop.is_running():
                self.loop.quit()
        except:
            pass

        logger.info(f"[STATS] recv={self.buffer_manager.frames_received}, sent={self.buffer_manager.frames_sent}")
        logger.info("Остановлено.")


# =========================
# MAIN FUNCTION
# =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Панорама с виртуальной камерой и записью')

    # Выбор источника
    parser.add_argument('--source', choices=['files', 'cameras'], default='files',
                       help='Источник видео: files (видеофайлы) или cameras (MIPI CSI камеры)')

    # Источники
    parser.add_argument('--video1', default="left.mp4",
                       help="Левое видео (путь к файлу или ID камеры)")
    parser.add_argument('--video2', default="right.mp4",
                       help="Правое видео (путь к файлу или ID камеры)")

    parser.add_argument('--config', default=None, help="Путь к конфигу nvinfer")
    parser.add_argument('--buffer', type=float, default=5.0, help="Длительность буфера (сек)")

    parser.add_argument('--mode', choices=['panorama', 'virtualcam', 'stream', 'record'],
                       default='virtualcam',
                       help='Режим: panorama=окно панорамы, virtualcam=окно камеры, stream=стрим на YouTube, record=только запись в файл')

    parser.add_argument('--output', type=str, default=None,
                       help='Путь к файлу для записи (работает только в режимах stream и record)')

    parser.add_argument('--stream-url', default='rtmp://a.rtmp.youtube.com/live2/',
                       help='RTMP URL для стриминга (например: rtmp://live.twitch.tv/live)')
    parser.add_argument('--stream-key', default='eub1-0rce-quc6-c1xm-d72s',
                       help='Ключ стрима stream')
    parser.add_argument('--bitrate', type=int, default=6000000,
                       help='Битрейт видео в bps (3500000=3.5Mbps для слабого 4G, 4500000=4.5Mbps для среднего 4G, 6000000=6Mbps для хорошего WiFi/4G)')
    parser.add_argument('--skip-interval', type=int, default=15,
                       help='Анализировать каждый N-й кадр')
    parser.add_argument('--confidence', type=float, default=0.35,
                       help='Порог уверенности детекции')
    parser.add_argument('--no-zoom', action='store_true',
                       help='Отключить автозум в режиме virtualcam')
    parser.add_argument('--disable-display', action='store_true',
                       help='Отключить отображение')
    parser.add_argument('--disable-analysis', action='store_true',
                       help='Отключить анализ')

    args = parser.parse_args()

    # Валидация параметров записи
    if args.output:
        # --output работает только с режимами stream и record
        if args.mode not in ['stream', 'record']:
            logger.error("Ошибка: --output работает только с режимами 'stream' или 'record'")
            logger.error(f"Текущий режим: {args.mode}")
            return 1

        # Для режима record параметр --output обязателен
        if args.mode == 'record' and not args.output:
            logger.error("Ошибка: для режима 'record' необходимо указать --output <файл>")
            return 1

    # Проверка режима record без --output
    if args.mode == 'record' and not args.output:
        logger.error("Ошибка: для режима 'record' необходимо указать --output <файл>")
        return 1

    # Автоматический выбор оптимального битрейта для режима record
    # Если пользователь НЕ указал --bitrate явно, используем максимум для записи
    if args.mode == 'record' and args.bitrate == 6000000:  # Значение по умолчанию
        args.bitrate = 8000000  # 8 Mbps для максимального качества записи
        logger.info("📹 Режим записи: автоматически установлен битрейт 8 Mbps (максимальное качество)")
        logger.info("   Для изменения используйте: --bitrate <значение>")

    # Проверка источников
    if args.source == "files":
        # Проверяем существование файлов
        for vf in [args.video1, args.video2]:
            if not os.path.exists(vf):
                logger.error(f"Файл не найден: {vf}")
                return 1
    else:
        # Для камер преобразуем в числа
        try:
            cam1 = int(args.video1)
            cam2 = int(args.video2)
            args.video1 = str(cam1)
            args.video2 = str(cam2)
            logger.info(f"Используем камеры: {cam1} и {cam2}")
        except ValueError:
            logger.error("Для камер укажите числовые ID (например: --video1 0 --video2 1)")
            return 1

    app = PanoramaWithVirtualCamera(
        source_type=args.source,
        video1=args.video1,
        video2=args.video2,
        config_path=args.config,
        buffer_duration=args.buffer,
        enable_display=not args.disable_display,
        display_mode=args.mode,
        enable_analysis=not args.disable_analysis,
        analysis_skip_interval=args.skip_interval,
        confidence_threshold=args.confidence,
        auto_zoom=not args.no_zoom,
        stream_url=args.stream_url,
        stream_key=args.stream_key,
        output_file=args.output,
        bitrate=args.bitrate
    )

    ok = app.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
