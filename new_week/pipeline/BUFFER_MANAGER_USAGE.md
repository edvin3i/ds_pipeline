# BufferManager Usage Guide

## Overview

The `BufferManager` class provides video and audio frame buffering capabilities for GStreamer pipelines. It was extracted from the `PanoramaWithVirtualCamera` class to provide modular, reusable buffer management.

## Features

- **RAM Buffering**: Configurable buffer duration (default: 7 seconds)
- **Thread Safety**: All buffer operations are protected by `RLock`
- **Timestamp Synchronization**: Automatic audio-video sync
- **Automatic Cleanup**: Old frames are removed to maintain buffer window
- **Statistics**: Track buffer state, frame counts, and delays
- **Emergency Handling**: Detects playback stalls and triggers callbacks

## Installation

```python
from pipeline import BufferManager
```

## Basic Usage

### 1. Create BufferManager Instance

```python
# Initialize with configuration
buffer_mgr = BufferManager(
    buffer_duration=7.0,           # 7 second buffer window
    framerate=30,                  # Video framerate
    audio_chunks_per_sec=100       # Audio chunks per second
)
```

### 2. Set GStreamer Elements

After creating your GStreamer pipelines, connect the elements:

```python
buffer_mgr.set_elements(
    appsrc=video_appsrc,           # Video appsrc for playback
    audio_appsrc=audio_appsrc,     # Audio appsrc for playback
    playback_pipeline=playback_pipeline
)
```

### 3. Set Emergency Shutdown Callback (Optional)

```python
def emergency_shutdown():
    """Handle emergency shutdown when playback stalls."""
    logger.error("Emergency shutdown triggered!")
    # Cleanup code here

buffer_mgr.set_emergency_shutdown_callback(emergency_shutdown)
```

### 4. Connect GStreamer Callbacks

Connect the BufferManager methods to your GStreamer appsink/appsrc elements:

```python
# Analysis pipeline - receive frames
video_appsink.connect("new-sample", buffer_mgr.on_new_sample)
audio_appsink.connect("new-sample", buffer_mgr.on_new_audio_sample)

# Playback pipeline - send frames
video_appsrc.connect("need-data", buffer_mgr._on_appsrc_need_data)
audio_appsrc.connect("need-data", buffer_mgr._on_audio_appsrc_need_data)
```

### 5. Start Buffer Thread

```python
# Start background buffering thread
buffer_mgr.start_buffer_thread()
```

### 6. Monitor Buffer Statistics

```python
# Get current buffer stats
stats = buffer_mgr.get_stats()
print(f"Frames received: {stats['frames_received']}")
print(f"Frames sent: {stats['frames_sent']}")
print(f"Buffer delay: {stats['display_buffer_duration']:.2f}s")
print(f"Video buffer: {stats['frame_buffer_size']}/{stats['frame_buffer_capacity']}")
print(f"Audio buffer: {stats['audio_buffer_size']}/{stats['audio_buffer_capacity']}")
```

### 7. Cleanup

```python
# Stop buffer thread
buffer_mgr.stop_buffer_thread()

# Clear all buffers
buffer_mgr.clear_buffers()
```

## Complete Integration Example

```python
from pipeline import BufferManager
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class MyPipeline:
    def __init__(self):
        # Initialize buffer manager
        self.buffer_mgr = BufferManager(
            buffer_duration=7.0,
            framerate=30
        )

        # Set emergency callback
        self.buffer_mgr.set_emergency_shutdown_callback(self._emergency_shutdown)

    def create_pipelines(self):
        """Create analysis and playback pipelines."""
        # Create analysis pipeline with appsink
        self.video_appsink = Gst.ElementFactory.make("appsink", "video_sink")
        self.audio_appsink = Gst.ElementFactory.make("appsink", "audio_sink")

        # Connect callbacks
        self.video_appsink.connect("new-sample", self.buffer_mgr.on_new_sample)
        self.audio_appsink.connect("new-sample", self.buffer_mgr.on_new_audio_sample)

        # Create playback pipeline with appsrc
        self.video_appsrc = Gst.ElementFactory.make("appsrc", "video_src")
        self.audio_appsrc = Gst.ElementFactory.make("appsrc", "audio_src")

        # Connect callbacks
        self.video_appsrc.connect("need-data", self.buffer_mgr._on_appsrc_need_data)
        self.audio_appsrc.connect("need-data", self.buffer_mgr._on_audio_appsrc_need_data)

        # Build pipelines...
        # (Add elements, link, etc.)

        # Set elements in buffer manager
        self.buffer_mgr.set_elements(
            appsrc=self.video_appsrc,
            audio_appsrc=self.audio_appsrc,
            playback_pipeline=self.playback_pipeline
        )

    def run(self):
        """Start pipelines and buffer thread."""
        # Start analysis pipeline
        self.analysis_pipeline.set_state(Gst.State.PLAYING)

        # Start buffer thread (will start playback when ready)
        self.buffer_mgr.start_buffer_thread()

        # Run main loop
        self.loop = GLib.MainLoop()
        self.loop.run()

    def stop(self):
        """Stop everything."""
        self.buffer_mgr.stop_buffer_thread()
        self.buffer_mgr.clear_buffers()

        # Stop pipelines
        self.analysis_pipeline.set_state(Gst.State.NULL)
        self.playback_pipeline.set_state(Gst.State.NULL)

    def _emergency_shutdown(self):
        """Emergency shutdown handler."""
        logger.error("Emergency shutdown triggered!")
        self.stop()
        return False
```

## Method Reference

### Initialization

#### `__init__(buffer_duration, framerate, audio_chunks_per_sec, appsrc, audio_appsrc, playback_pipeline)`

Initialize the BufferManager with configuration.

**Parameters:**
- `buffer_duration` (float): Buffer window in seconds (default: 7.0)
- `framerate` (int): Video framerate (default: 30)
- `audio_chunks_per_sec` (int): Audio chunks per second (default: 100)
- `appsrc` (Gst.Element): Video appsrc element (optional)
- `audio_appsrc` (Gst.Element): Audio appsrc element (optional)
- `playback_pipeline` (Gst.Element): Playback pipeline (optional)

### Configuration Methods

#### `set_elements(appsrc, audio_appsrc, playback_pipeline)`

Set or update GStreamer elements after initialization.

#### `set_emergency_shutdown_callback(callback)`

Set callback function to be called on emergency shutdown.

### GStreamer Callback Methods

#### `on_new_sample(sink)`

Video frame callback for appsink. Receives and buffers video frames.

**Returns:** `Gst.FlowReturn` status

#### `on_new_audio_sample(sink)`

Audio sample callback for appsink. Receives and buffers audio samples.

**Returns:** `Gst.FlowReturn` status

#### `_on_appsrc_need_data(src, length)`

Video frame callback for appsrc. Pushes buffered video frames to playback.

#### `_on_audio_appsrc_need_data(src, length)`

Audio sample callback for appsrc. Pushes buffered audio synchronized with video.

### Thread Management

#### `start_buffer_thread()`

Start the background buffering thread.

#### `stop_buffer_thread()`

Stop the background buffering thread.

### Buffer Management

#### `clear_buffers()`

Clear all buffers and reset state.

#### `get_stats()`

Get buffer statistics.

**Returns:** Dictionary with:
- `frames_received`: Total frames received
- `frames_sent`: Total frames sent
- `frame_buffer_size`: Current video buffer size
- `frame_buffer_capacity`: Max video buffer capacity
- `audio_buffer_size`: Current audio buffer size
- `audio_buffer_capacity`: Max audio buffer capacity
- `display_buffer_duration`: Current buffer delay in seconds
- `current_playback_time`: Current playback timestamp

## Internal Methods

### `_push_audio_for_timestamp(video_timestamp)`

Synchronize audio with video by pushing audio chunk for given timestamp.

### `_remove_old_audio_chunks()`

Remove old audio chunks from buffer to maintain window.

### `_remove_old_frames_locked()`

Remove old video frames from buffer (must be called with lock held).

### `_buffer_loop()`

Background thread function that:
1. Waits for buffer to fill (30% of capacity)
2. Starts playback pipeline
3. Monitors for playback stalls
4. Triggers emergency shutdown if needed

## Thread Safety

All buffer operations are thread-safe using `threading.RLock()`:
- Multiple readers can access simultaneously
- Write operations are serialized
- Nested locking is supported

## Buffer Window Management

- **Video Buffer**: `buffer_duration * framerate` frames
- **Audio Buffer**: `buffer_duration * audio_chunks_per_sec` chunks
- **Cleanup**: Old frames beyond playback position are automatically removed
- **Threshold**: Keeps frames within 0.5s before playback position

## Timestamp Synchronization

- Video and audio use GStreamer PTS (presentation timestamp)
- Audio is synchronized within 100ms of video timestamp
- Tolerance: 50ms for audio chunk matching

## Emergency Shutdown Detection

The buffer loop monitors for playback stalls:
- Triggers if no frames sent for >5 seconds
- Calls emergency shutdown callback
- Prevents pipeline deadlock

## Performance Considerations

- **Buffer Size**: 7 seconds at 30fps = ~210 video frames
- **Audio Chunks**: 7 seconds at 100 chunks/sec = ~700 chunks
- **Memory Usage**: ~18KB code + buffer data
- **Thread Overhead**: Single background thread (daemon)

## Logging

The BufferManager logs to the `buffer-manager` logger:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buffer-manager")
```

Log messages include:
- `[SOURCE]`: Frame reception statistics
- `[PLAYBACK]`: Frame sending statistics
- `[BUFFER]`: Thread status and buffering progress

## Migration from PanoramaWithVirtualCamera

To migrate from the monolithic class:

1. Create `BufferManager` instance
2. Move buffer-related initialization to `BufferManager.__init__`
3. Replace direct method calls with `buffer_mgr.method_name()`
4. Connect GStreamer callbacks to buffer manager methods
5. Use `buffer_mgr.start_buffer_thread()` instead of inline thread creation

## See Also

- `pipeline/config_builder.py` - Configuration generation
- `pipeline/pipeline_builder.py` - Analysis pipeline builder
- `pipeline/playback_builder.py` - Playback pipeline builder
