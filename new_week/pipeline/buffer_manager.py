#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVMM Buffer Manager for Video and Audio Frame Buffering.

Manages video and audio frame buffering using GPU memory (NVMM) with:
- 7 second window buffering (configurable)
- Zero-copy buffer reference management (no deep copies)
- Python automatic memory management (no manual ref/unref)
- Timestamp synchronization between analysis and display
- Thread-safe buffer access with locks
- Buffer statistics tracking (duration, frame count)
- Both video and audio data stream handling
- Background buffering thread management

CRITICAL: Video buffers stay in NVMM (GPU memory) throughout.
Python's GC manages buffer lifetime automatically via GObject introspection.
Stores Gst.Sample objects - Python references keep NVMM buffers alive.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import time
import logging
from collections import deque
from typing import Optional, Dict, Any

logger = logging.getLogger("buffer-manager")


class NVMMBufferManager:
    """
    Manages video and audio frame buffering for GStreamer pipelines using NVMM.

    This class handles:
    - Receiving video/audio frames from appsink callbacks
    - Buffering Gst.Sample REFERENCES (not copies) in NVMM with configurable duration
    - Python automatic memory management (no manual ref/unref needed)
    - Pushing frames to playback pipeline via appsrc callbacks
    - Timestamp synchronization between video and audio
    - Cleanup of old frames to maintain buffer window
    - Background thread for buffer management

    CRITICAL: Video buffers stay in GPU memory (NVMM) throughout.
    No CPU copies or deep copies - only Python references to Gst.Sample objects.
    Python's GC automatically handles buffer lifetime via GObject introspection.
    """

    def __init__(self,
                 buffer_duration: float = 7.0,
                 framerate: int = 30,
                 audio_chunks_per_sec: int = 100,
                 appsrc: Optional[Gst.Element] = None,
                 audio_appsrc: Optional[Gst.Element] = None,
                 playback_pipeline: Optional[Gst.Element] = None):
        """
        Initialize the BufferManager.

        Args:
            buffer_duration: Duration of buffer in seconds (default: 7.0)
            framerate: Video framerate (default: 30)
            audio_chunks_per_sec: Number of audio chunks per second (default: 100)
            appsrc: Video appsrc element for playback pipeline
            audio_appsrc: Audio appsrc element for playback pipeline
            playback_pipeline: Playback pipeline element
        """
        self.buffer_duration = float(buffer_duration)
        self.framerate = framerate
        self.audio_chunks_per_sec = audio_chunks_per_sec

        # GStreamer elements
        self.appsrc = appsrc
        self.audio_appsrc = audio_appsrc
        self.playback_pipeline = playback_pipeline

        # Frame buffers
        self.frame_buffer = deque(maxlen=int(self.buffer_duration * self.framerate))
        self.audio_buffer = deque(maxlen=int(self.buffer_duration * self.audio_chunks_per_sec))

        # Audio caps storage
        self.audio_caps = None

        # Thread safety
        self.buffer_lock = threading.RLock()

        # Statistics
        self.frames_received = 0
        self.frames_sent = 0

        # Playback state
        self.current_playback_time = None
        self.last_send_time = 0.0
        self.last_frame_sent_time = 0.0
        self.send_interval = 1.0 / self.framerate
        self.display_buffer_duration = 0.0

        # Background thread
        self.buffer_thread = None
        self.buffer_thread_running = False

        # Emergency shutdown callback
        self.emergency_shutdown_callback = None

    def set_elements(self,
                     appsrc: Optional[Gst.Element] = None,
                     audio_appsrc: Optional[Gst.Element] = None,
                     playback_pipeline: Optional[Gst.Element] = None):
        """
        Set or update GStreamer elements after initialization.

        Args:
            appsrc: Video appsrc element
            audio_appsrc: Audio appsrc element
            playback_pipeline: Playback pipeline element
        """
        if appsrc is not None:
            self.appsrc = appsrc
        if audio_appsrc is not None:
            self.audio_appsrc = audio_appsrc
        if playback_pipeline is not None:
            self.playback_pipeline = playback_pipeline

    def set_emergency_shutdown_callback(self, callback):
        """
        Set callback function for emergency shutdown.

        Args:
            callback: Function to call on emergency shutdown
        """
        self.emergency_shutdown_callback = callback

    def on_new_sample(self, sink):
        """
        Receive NVMM buffers from appsink (zero-copy).

        CRITICAL: Buffer stays in NVMM, only store Python reference.
        Python's GC manages buffer lifetime automatically (no manual ref/unref needed).

        GStreamer callback for receiving video frames.

        Args:
            sink: GStreamer appsink element

        Returns:
            Gst.FlowReturn status
        """
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.OK

            # Get timestamp
            timestamp = (float(buffer.pts) / float(Gst.SECOND)
                        if buffer.pts != Gst.CLOCK_TIME_NONE
                        else time.time())

            with self.buffer_lock:
                # CRITICAL: Store sample object - Python's GC keeps buffer alive!
                # No need for .ref()/.unref() - Python GI handles refcounting automatically
                # Buffer stays in NVMM (zero-copy), we just hold Python reference
                self.frame_buffer.append({
                    'timestamp': timestamp,
                    'sample': sample  # Python reference keeps buffer alive
                })

                self.frames_received += 1

                # Log every 300 frames
                if self.frames_received % 300 == 0:
                    logger.info(
                        f"[NVMM-BUFFER] recv={self.frames_received}, "
                        f"buf={len(self.frame_buffer)}/{self.frame_buffer.maxlen}"
                    )

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"on_new_sample error: {e}", exc_info=True)
            return Gst.FlowReturn.ERROR

    def on_new_audio_sample(self, sink):
        """
        Получаем аудио сэмплы и буферизируем их.

        GStreamer callback for receiving audio samples.

        Args:
            sink: GStreamer audio appsink element

        Returns:
            Gst.FlowReturn status
        """
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.OK

            # Используем тот же timestamp что и для видео
            timestamp = float(buffer.pts) / float(Gst.SECOND) if buffer.pts != Gst.CLOCK_TIME_NONE else time.time()

            with self.buffer_lock:
                buffer_copy = buffer.copy_deep() if hasattr(buffer, 'copy_deep') else buffer.copy()

                # Сохраняем caps только для первого буфера
                caps_copy = sample.get_caps() if not self.audio_caps else None
                if caps_copy and not self.audio_caps:
                    self.audio_caps = caps_copy

                self.audio_buffer.append({
                    'timestamp': timestamp,
                    'buffer': buffer_copy,
                    'caps': caps_copy
                })

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"on_new_audio_sample error: {e}")
            return Gst.FlowReturn.ERROR

    def _on_appsrc_need_data(self, src, length):
        """
        Подаём кадры в playback пайплайн.

        GStreamer callback for pushing video frames to playback.

        Args:
            src: GStreamer appsrc element
            length: Requested buffer length
        """
        try:
            if not self.frame_buffer:
                return

            with self.buffer_lock:
                if len(self.frame_buffer) == 0:
                    return

                if self.current_playback_time is None:
                    self.current_playback_time = self.frame_buffer[0]['timestamp']

                frame_to_send = None
                for frame in self.frame_buffer:
                    if frame['timestamp'] >= self.current_playback_time:
                        frame_to_send = frame
                        break

                if frame_to_send is None:
                    return

                self.current_playback_time = frame_to_send['timestamp']
                self._remove_old_frames_locked()

                if len(self.frame_buffer) >= 2:
                    newest_ts = self.frame_buffer[-1]['timestamp']
                    self.display_buffer_duration = max(0.0, newest_ts - self.current_playback_time)

            # Get buffer from sample (buffer stays in NVMM)
            buffer = frame_to_send['sample'].get_buffer()

            # Используем оригинальные временные метки
            buffer.pts = int(frame_to_send['timestamp'] * Gst.SECOND)
            buffer.dts = buffer.pts
            buffer.duration = int((1.0 / self.framerate) * Gst.SECOND)

            # NOTE: Caps are set in playback_builder.py before pipeline starts
            # No need to set caps here - appsrc already has NVMM caps configured

            result = src.emit("push-buffer", buffer)
            if result == Gst.FlowReturn.OK:
                self.frames_sent += 1
                self.last_send_time = time.time()
                self.last_frame_sent_time = self.last_send_time

                if self.audio_appsrc and self.audio_buffer:
                    self._push_audio_for_timestamp(self.current_playback_time)

                if self.frames_sent % 300 == 0:
                    logger.info(
                        f"[NVMM-PLAYBACK] sent={self.frames_sent}, "
                        f"delay={self.display_buffer_duration:.2f}s"
                    )

        except Exception as e:
            logger.error(f"_on_appsrc_need_data error: {e}")

    def _on_audio_appsrc_need_data(self, src, length):
        """
        Подаём аудио в playback pipeline синхронно с видео.

        GStreamer callback for pushing audio samples synchronized with video.

        Args:
            src: GStreamer audio appsrc element
            length: Requested buffer length
        """
        try:
            if not self.audio_buffer or not self.current_playback_time:
                return

            with self.buffer_lock:
                # Ищем аудио chunk с нужным timestamp
                audio_to_send = None

                for audio_chunk in self.audio_buffer:
                    if audio_chunk['timestamp'] >= self.current_playback_time - 0.05:  # 50ms tolerance
                        audio_to_send = audio_chunk
                        break

                if not audio_to_send:
                    # Если нет точного совпадения, берём ближайший
                    if self.audio_buffer:
                        audio_to_send = self.audio_buffer[0]
                    else:
                        return

            # Подготавливаем буфер
            buffer = audio_to_send['buffer']

            # ВАЖНО: используем тот же timestamp что и видео!
            buffer.pts = int(self.current_playback_time * Gst.SECOND)
            buffer.dts = buffer.pts

            # Устанавливаем caps при первой отправке
            if audio_to_send.get('caps') and self.audio_caps:
                src.set_property("caps", audio_to_send['caps'])

            # Отправляем
            result = src.emit("push-buffer", buffer)

            # Удаляем старые аудио chunks
            self._remove_old_audio_chunks()

        except Exception as e:
            logger.error(f"_on_audio_appsrc_need_data error: {e}")

    def _push_audio_for_timestamp(self, video_timestamp):
        """
        Отправляет аудио chunk соответствующий видео timestamp.

        Synchronizes audio with video by timestamp.

        Args:
            video_timestamp: Video timestamp to sync audio with
        """
        try:
            if not self.audio_buffer or not self.audio_appsrc:
                return

            with self.buffer_lock:
                # Ищем ближайший аудио chunk
                best_audio = None
                min_diff = float('inf')

                for audio_chunk in list(self.audio_buffer):
                    diff = abs(audio_chunk['timestamp'] - video_timestamp)
                    if diff < min_diff and diff < 0.1:  # В пределах 100ms
                        min_diff = diff
                        best_audio = audio_chunk

                if best_audio:
                    # Копируем буфер
                    audio_buf = best_audio['buffer'].copy()
                    audio_buf.pts = int(video_timestamp * Gst.SECOND)
                    audio_buf.dts = audio_buf.pts
                    audio_buf.duration = int(0.02 * Gst.SECOND)  # ~20ms chunk

                    # Отправляем
                    ret = self.audio_appsrc.emit("push-buffer", audio_buf)

                    # Удаляем старые аудио chunks
                    cutoff = video_timestamp - 1.0
                    while self.audio_buffer and self.audio_buffer[0]['timestamp'] < cutoff:
                        self.audio_buffer.popleft()

        except Exception as e:
            logger.debug(f"Audio push error: {e}")

    def _remove_old_audio_chunks(self):
        """
        Удаляем старые аудио chunks из буфера.

        Cleanup old audio chunks from buffer to maintain buffer window.
        """
        if not self.current_playback_time or not self.audio_buffer:
            return

        threshold = self.current_playback_time - 1.0  # Храним 1 секунду истории

        with self.buffer_lock:
            while self.audio_buffer and self.audio_buffer[0]['timestamp'] < threshold:
                chunk = self.audio_buffer.popleft()
                chunk['buffer'] = None
                chunk['caps'] = None

    def _remove_old_frames_locked(self):
        """
        Remove old frames (CRITICAL for pool recycling).

        Must be called with buffer_lock held.

        Python's GC automatically handles buffer cleanup when samples
        go out of scope - no manual unref() needed.
        """
        if self.current_playback_time is None or not self.frame_buffer:
            return

        threshold = self.current_playback_time - 0.5  # Keep 0.5s history
        removed_count = 0

        while (len(self.frame_buffer) > 1 and
               self.frame_buffer[0]['timestamp'] < threshold):
            old_frame = self.frame_buffer.popleft()

            # Clear references to help Python's GC
            # When sample goes out of scope, GI layer unrefs GStreamer buffer
            old_frame['sample'] = None
            removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"[NVMM-BUFFER] Released {removed_count} sample refs (Python GC)"
            )

    def _buffer_loop(self):
        """
        Фоновый поток буферизации.

        Background thread for buffer management.
        Waits for buffer to fill, then starts playback pipeline.
        Monitors for playback stalls and triggers emergency shutdown if needed.
        """
        logger.info("[BUFFER] поток запущен")
        threshold = int(self.frame_buffer.maxlen * 0.3)
        wait_steps = 0

        while self.buffer_thread_running and len(self.frame_buffer) < threshold and wait_steps < 100:
            # Логируем только каждую секунду (каждые 10 итераций)
            if wait_steps % 10 == 0:
                logger.info(f"[BUFFER] ожидание: {len(self.frame_buffer)}/{threshold}")
            time.sleep(0.1)
            wait_steps += 1

        if not self.buffer_thread_running:
            return

        logger.info(f"[BUFFER] достаточно кадров в RAM, старт playback")
        if self.playback_pipeline:
            self.playback_pipeline.set_state(Gst.State.PLAYING)

        if self.appsrc:
            self.appsrc.emit("need-data", 0)

        first_sent = False
        start_t = time.time()

        while self.buffer_thread_running:
            if not first_sent and self.frames_sent > 0:
                first_sent = True
                logger.info(f"[BUFFER] первый кадр отправлен через {time.time() - start_t:.2f}s")

            # Проверка зависания
            if first_sent and (time.time() - self.last_frame_sent_time) > 5.0:
                logger.error("🔴 КРИТИЧНО: Нет отправки в playback >5с! Экстренное завершение!")
                self.buffer_thread_running = False
                if self.emergency_shutdown_callback:
                    GLib.idle_add(self.emergency_shutdown_callback)
                return

            time.sleep(0.2)

        logger.info("[BUFFER] поток завершён")

    def start_buffer_thread(self):
        """
        Start the background buffering thread.
        """
        if not self.buffer_thread_running:
            self.buffer_thread_running = True
            self.buffer_thread = threading.Thread(target=self._buffer_loop, daemon=True)
            self.buffer_thread.start()
            logger.info("[BUFFER] Background thread started")

    def stop_buffer_thread(self):
        """
        Stop the background buffering thread.
        """
        self.buffer_thread_running = False
        if self.buffer_thread:
            self.buffer_thread.join(timeout=1.0)
            logger.info("[BUFFER] Background thread stopped")

    def clear_buffers(self):
        """
        Clear all buffers and reset state.

        Python's GC automatically cleans up samples when they go out of scope.
        """
        with self.buffer_lock:
            # Clear all references - Python's GC will clean up automatically
            for frame in self.frame_buffer:
                frame['sample'] = None
                frame['caps'] = None

            self.frame_buffer.clear()
            self.audio_buffer.clear()
            self.frames_received = 0
            self.frames_sent = 0
            self.current_playback_time = None
            self.audio_caps = None

            logger.info("[NVMM-BUFFER] Buffers cleared (Python GC will cleanup)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self.buffer_lock:
            return {
                'frames_received': self.frames_received,
                'frames_sent': self.frames_sent,
                'frame_buffer_size': len(self.frame_buffer),
                'frame_buffer_capacity': self.frame_buffer.maxlen,
                'audio_buffer_size': len(self.audio_buffer),
                'audio_buffer_capacity': self.audio_buffer.maxlen,
                'display_buffer_duration': self.display_buffer_duration,
                'current_playback_time': self.current_playback_time,
                'memory_type': 'NVMM',  # Indicates zero-copy NVMM buffering
            }

    def __del__(self):
        """
        Cleanup on destruction.

        Python's GC automatically handles buffer cleanup.
        This method just clears references to help GC.
        """
        try:
            logger.info(
                f"[NVMM-BUFFER] Destructor: clearing {len(self.frame_buffer)} sample refs"
            )

            # Clear all references - Python's GC will clean up automatically
            for frame in self.frame_buffer:
                frame['sample'] = None
                frame['caps'] = None

            self.frame_buffer.clear()

            logger.info("[NVMM-BUFFER] Destructor: all sample refs cleared (GC will cleanup)")

        except Exception as e:
            logger.error(f"[NVMM-BUFFER] Destructor error: {e}", exc_info=True)


# Backward compatibility alias
BufferManager = NVMMBufferManager
