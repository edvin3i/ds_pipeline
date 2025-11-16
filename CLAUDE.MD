# DeepStream Football Analysis Pipeline

## Project Overview

This project implements a real-time football/soccer analysis system using NVIDIA DeepStream SDK on Jetson Orin NX platform. The system captures video from dual 4K fisheye cameras, creates a panoramic view, performs object detection (ball and players), and provides an intelligent virtual camera with automatic ball tracking.

**Platform**: NVIDIA Jetson Orin NX 16GB
**Framework**: DeepStream 7.1 (JetPack 6.2)
**Performance**: 30-45 FPS end-to-end processing
**Main Pipeline**: `new_week/version_masr_multiclass.py`

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES (Dual Cameras)                 │
│                 4K Left Camera + 4K Right Camera                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PANORAMA STITCHING                            │
│                     nvdsstitch Plugin                            │
│                  Output: 5700×1900 panorama                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
                          TEE (Branch split)
                           ↓
         ┌─────────────────┴────────────────┐
         ↓                                   ↓
┌────────────────────┐            ┌──────────────────────┐
│  ANALYSIS BRANCH   │            │   DISPLAY BRANCH     │
│                    │            │                      │
│ nvtilebatcher      │            │ 7-second buffer      │
│   ↓                │            │   ↓                  │
│ nvinfer (YOLO11)   │            │ nvdsvirtualcam       │
│   ↓                │            │   ↓                  │
│ Detection storage  │            │ Output (1920×1080)   │
└────────────────────┘            └──────────────────────┘
```

### Dual-Pipeline Architecture

The system employs a **time-delayed dual-pipeline** design:

**1. Analysis Pipeline (Real-time)**:
- Captures video from 2 sources
- Creates panorama stitching
- Extracts 6 tiled regions (1024×1024 each)
- Performs object detection (YOLO11n)
- Buffers frames and detections in RAM (7 seconds default)

**2. Playback Pipeline (Delayed)**:
- Reads buffered frames with 7-second delay
- Retrieves corresponding detection results
- Applies virtual camera transformation
- Outputs to display/stream/recording

### Time Architecture

```
Analysis:  Frame 0 ──────▶ Frame 210 ──────▶ Frame 420
           (t=0s)         (t=7s)            (t=14s)

Playback:                 Frame 0 ──────▶ Frame 210
                          (t=7s)         (t=14s)

           ◄────7s delay────►
```

**Benefits**:
- Smooth playback even with detection gaps
- Time for trajectory interpolation
- Allows ball tracking with historical context
- No dropped frames during inference spikes

## Hardware Platform

### NVIDIA Jetson Orin NX 16GB

**GPU**: NVIDIA Ampere Architecture
- 1024 CUDA cores
- 32 Tensor cores
- ~918 MHz clock
- ~100 TOPS (INT8, sparse)

**CPU**: 8× ARM Cortex-A78AE
- 64-bit ARMv8.2
- Up to 2 GHz
- 4 MB L3 cache

**Memory**: 16 GB LPDDR5 SDRAM
- Unified memory (CPU and GPU share same physical RAM)
- 128-bit bus width
- 3200 MHz frequency
- ~102 GB/s peak bandwidth

**Accelerators**:
- 2× NVDLA (Deep Learning Accelerator) - ~20 TOPS each (INT8)
- NVENC/NVDEC (Hardware video encoding/decoding)
- VIC (Video Image Compositor)

### Memory Architecture

**Unified Memory Design**:
- No separate VRAM - CPU and GPU share 16GB physical RAM
- Zero-copy operations possible via NVMM (NVIDIA Memory Manager)
- Hardware coherency (I/O coherent, GPU compute capability ≥7.2)
- DMAbuf support for efficient buffer sharing

**Why Zero-Copy is Critical**:
1. **Bandwidth Limited**: ~102 GB/s shared between CPU/GPU
2. **Large Data**: 4K panorama = ~43 MB per frame
3. **High Frequency**: 30 FPS = 1.3 GB/s sustained throughput
4. **Copying Cost**: Even with unified memory, copying takes time and bandwidth

**Solution**: Use NVMM memory type throughout pipeline
```
nvarguscamerasrc (NVMM) → nvvideoconvert (NVMM) →
nvdsstitch (NVMM) → nvtilebatcher (NVMM) →
nvinfer (NVMM) → nvdsvirtualcam (NVMM) → output (NVMM)
```

All GPU operations happen in-place with EGL/CUDA interop.

## Custom Plugins

The system uses **three custom GStreamer plugins**, all optimized for Jetson with CUDA acceleration:

### 1. nvdsstitch - Panorama Stitching

**Purpose**: Combines dual 4K fisheye camera feeds into equirectangular panorama

**Location**: `my_steach/`

**Key Features**:
- LUT-based warping (pre-computed spherical projection)
- Automatic color correction in overlap zones
- VIC engine offloading for buffer management
- EGL zero-copy on Jetson

**Specifications**:
- Input: 2× 3840×2160 RGBA (NVMM batch)
- Output: 5700×1900 RGBA (NVMM)
- Performance: 40-50 FPS
- Memory: ~670 MB GPU

**Algorithm**:
1. Split batched input to intermediate buffers (VIC)
2. Load pre-computed LUT maps (6 arrays: X/Y coords + weights for each camera)
3. CUDA kernel: Bilinear sample from both cameras
4. Weighted blend based on overlap region
5. Apply color correction gains (updated every 30 frames)
6. Output equirectangular panorama

**Documentation**: See `my_steach/README.md`

### 2. nvtilebatcher - Tile Extraction

**Purpose**: Extract 6 non-overlapping 1024×1024 tiles from panorama for inference

**Location**: `my_tile_batcher/`

**Key Features**:
- Parallel CUDA tile extraction (6 tiles in one kernel launch)
- Fixed buffer pool (4 pre-allocated batches)
- DeepStream metadata integration
- EGL resource caching

**Specifications**:
- Input: 5700×1900 RGBA (NVMM)
- Output: 6× 1024×1024 RGBA (NVMM batch)
- Tile positions: X=[192, 1216, 2240, 3264, 4288, 5312], Y=434
- Performance: 500-1000 μs per frame
- Memory: ~96 MB GPU

**Algorithm**:
1. Calculate tile positions (horizontal strip across panorama)
2. Set tile pointers in CUDA constant memory
3. Launch CUDA kernel with grid (32×32×6)
4. Each thread copies one pixel from panorama to corresponding tile
5. Create DeepStream batch metadata (6 NvDsFrameMeta)
6. Attach TileRegionInfo for coordinate mapping

**Documentation**: See `my_tile_batcher/README.md`

### 3. nvdsvirtualcam - Virtual Camera

**Purpose**: Create virtual camera view from panorama with ball tracking

**Location**: `my_virt_cam/`

**Key Features**:
- Spherical coordinate transformation
- Pan (-90° to +90°), Tilt (-27° to +27°), Zoom (40-68° FOV)
- Auto-follow ball tracking with smooth transitions
- Predictive edge offset
- Speed-adaptive zoom
- LUT-based optimization

**Specifications**:
- Input: 5700×1900 RGBA (NVMM)
- Output: 1920×1080 RGBA (NVMM)
- Performance: 45-48 FPS
- Memory: ~146 MB GPU

**Algorithm**:
1. **Camera Ray Generation**: For each output pixel, compute 3D ray based on FOV
2. **Rotation Application**: Apply Roll → Pitch → Yaw transformations
3. **Spherical Projection**: Convert to spherical coordinates (λ, φ)
4. **Panorama Mapping**: Map to panorama pixel (u, v)
5. **LUT Optimization**: Pre-compute and cache mappings (update only when angles change >0.1°)
6. **Interpolation**: Nearest neighbor (real-time performance)

**Ball Tracking**:
- Converts ball pixel coordinates to yaw/pitch angles
- Adds predictive edge offset (±8° yaw, ±4° pitch)
- Auto-zoom based on ball size (closer = wider FOV)
- Exponential smoothing (default 30% per frame)

**Documentation**: See `my_virt_cam/README.md`

## Main Pipeline: version_masr_multiclass.py

**Location**: `new_week/version_masr_multiclass.py`

**Purpose**: Production-ready football analysis pipeline with multiclass detection

### Pipeline Modes

**1. Source Modes**:
- `--source cameras`: Live MIPI CSI cameras (sensor IDs 0, 1)
- `--source files`: Video files (`--video1`, `--video2`)

**2. Output Modes**:
- `--mode panorama`: Display full 5700×1900 panorama
- `--mode virtualcam`: Display 1920×1080 virtual camera
- `--mode stream`: RTMP streaming to YouTube/Twitch (with audio)
- `--mode record`: File recording (FLV/MP4/MKV)

### Configuration Files

**config_infer.txt** (nvinfer configuration):
```ini
[property]
model-engine-file=yolo11n_mixed_finetune_v9.engine
network-mode=1
batch-size=6
network-type=100
output-tensor-meta=1

[class-attrs-all]
pre-cluster-threshold=0.25  # Ball threshold
nms-iou-threshold=0.45

[class-attrs-1]
pre-cluster-threshold=0.40  # Player threshold
```

**field_mask.png**:
- Binary mask (5700×1900) defining valid field area
- Used to filter out-of-field detections
- Determines tile vertical offset (434px calculated from mask)

**labels.txt**:
```
ball
player
staff
side_referee
main_referee
```

### Detection Processing

**Multiclass Detection**:
The system detects 5 classes but prioritizes display:
1. **Ball** (red boxes, priority 1)
2. **Players** (green center-of-mass, priority 2)
3. Staff/Referees (detected but not displayed)

**Detection Flow**:
```
YOLO Raw Tensors (6 tiles)
    ↓
TensorProcessor.postprocess_yolo_output()
    ↓
Confidence filtering (0.35)
    ↓
Field mask validation
    ↓
Size filtering (8-120 pixels)
    ↓
Edge rejection (20px margin)
    ↓
Adaptive distance filtering (100-500px radius)
    ↓
Permanent ban zones (outlier removal)
    ↓
BallDetectionHistory storage (timestamped)
```

**Trajectory Interpolation**:
- **Linear**: Short gaps (<1s)
- **Parabolic**: Ball flight (>1s, distance >500px)
- **Backward**: Fill historical gaps after recovery
- **Smoothing**: EMA for ball radius and player center-of-mass

### Buffer Management

**Frame Buffering**:
- Analysis pipeline extracts frames via `appsink`
- Deep-copied to prevent data corruption
- Stored in thread-safe deque (maxlen: 210 frames = 7 seconds @ 30 FPS)
- Playback pipeline reads via `appsrc` with timestamp matching

**Audio Buffering**:
- PulseAudio integration (for streaming mode)
- ~500 audio chunks buffered
- Synchronized with video timestamps

**Detection History**:
```python
BallDetectionHistory:
    - raw_future_history: New detections from analysis
    - processed_future_history: Cleaned + interpolated
    - confirmed_history: Already-displayed detections
    - permanent_ban_zones: Outlier rejection regions
```

### Virtual Camera Integration

**Update Probe** (`vcam_update_probe`):
1. Retrieve ball detection for current timestamp
2. Fallback to players center-of-mass if no ball
3. Calculate ball velocity (pixels/second)
4. Apply speed-based zoom factor (1.0-3.0×)
5. Update virtual camera properties

**Speed-Adaptive Zoom**:
```python
if ball_speed < 300 px/s:
    zoom_factor = 1.0  # Wide view
elif ball_speed > 1200 px/s:
    zoom_factor = 3.0  # Maximum zoom
else:
    zoom_factor = interpolate(1.0, 3.0)
```

**Display Probe** (`playback_draw_probe`):
1. Retrieve all detections for current timestamp
2. Draw ball bounding boxes (real: red, interpolated: yellow)
3. Draw players center-of-mass (green 100×100 box)
4. Render FPS and buffer stats
5. Limit: 16 rectangles (nvdsosd platform constraint)

### Performance Characteristics

**Frame Rate**:
- Analysis: 30 FPS (limited by cameras or file playback)
- Playback: 30 FPS (matched to analysis)
- Frame skip: Process every 8th frame for inference (configurable)
- Effective detection rate: 3.75 FPS (sufficient for ball tracking)

**Latency**:
- Panorama stitching: ~20-25 ms
- Tile extraction: ~1 ms
- YOLO inference (6 tiles): ~30-40 ms
- Post-processing: ~5-10 ms
- Virtual camera: ~1-2 ms
- **Total analysis**: ~60-80 ms per processed frame

**Memory Usage**:
- nvdsstitch LUT maps: ~260 MB
- nvtilebatcher buffers: ~96 MB
- nvdsvirtualcam LUT: ~40 MB
- Frame buffers (210 frames): ~3 GB
- YOLO model: ~15 MB
- Total: ~3.5 GB (fits comfortably in 16 GB)

## DeepStream Integration

### GStreamer Elements Used

**Input**:
- `nvarguscamerasrc`: MIPI CSI camera capture
- `filesrc` + `qtdemux` + `h264parse`: File playback
- `nvv4l2decoder`: Hardware H.264/H.265 decoding

**Processing**:
- `nvstreammux`: Stream multiplexing (batch=2 for dual cameras)
- `nvvideoconvert`: Format conversion (NV12↔RGBA, GPU-accelerated)
- `nvdsstitch`: Custom panorama stitching
- `nvtilebatcher`: Custom tile extraction
- `nvinfer`: YOLO inference
- `nvdsvirtualcam`: Custom virtual camera
- `nvdsosd`: On-screen display overlay

**Output**:
- `nveglglessink`: EGL display (Jetson)
- `xvimagesink`: X11 display (fallback)
- `nvv4l2h264enc`: Hardware H.264 encoding
- `rtmpsink`: RTMP streaming
- `filesink`: File recording

**Audio** (streaming mode only):
- `pulsesrc`: Audio capture
- `audioconvert` + `audioresample`: Format conversion
- `voaacenc`: AAC encoding
- Muxed with video for RTMP

### Metadata Flow

**Analysis Path**:
```
nvinfer (NvDsInferTensorMeta)
    ↓
TensorProcessor (Python)
    ↓
Detection Dict {timestamp, x, y, w, h, class, confidence}
    ↓
BallDetectionHistory
```

**Display Path**:
```
BallDetectionHistory.get_detection_for_timestamp()
    ↓
NvDsDisplayMeta (nvdsosd)
    ↓
rect_params[]: bounding boxes with colors
    ↓
nvdsosd renders on frame
```

### Probe Points

| Probe Name | Element | Pad | Purpose |
|------------|---------|-----|---------|
| `frame_skip_probe` | identity | src | Drop frames (analyze every Nth) |
| `analysis_probe` | nvinfer | src | Extract YOLO detections |
| `vcam_update_probe` | nvdsvirtualcam | sink | Update camera position |
| `playback_draw_probe` | nvdsosd | sink | Draw bounding boxes |

## Usage Guide

### Installation

1. **Platform Setup**:
```bash
# Install JetPack 6.2 on Jetson Orin NX
# Includes DeepStream 7.1, CUDA 12.6, GStreamer 1.0
```

2. **Build Custom Plugins**:
```bash
# nvdsstitch
cd my_steach
make && make install

# nvtilebatcher
cd ../my_tile_batcher/src
make && make install

# nvdsvirtualcam
cd ../../my_virt_cam/src
make && make install
```

3. **Verify Installation**:
```bash
gst-inspect-1.0 nvdsstitch
gst-inspect-1.0 nvtilebatcher
gst-inspect-1.0 nvdsvirtualcam
```

### Running the Pipeline

**From Live Cameras**:
```bash
cd new_week
python3 version_masr_multiclass.py \
    --source cameras \
    --mode virtualcam \
    --skip-interval 8
```

**From Video Files**:
```bash
python3 version_masr_multiclass.py \
    --source files \
    --video1 left_camera.mp4 \
    --video2 right_camera.mp4 \
    --mode virtualcam \
    --skip-interval 8
```

**Streaming to YouTube**:
```bash
python3 version_masr_multiclass.py \
    --source cameras \
    --mode stream \
    --rtmp-url "rtmp://a.rtmp.youtube.com/live2/<stream_key>" \
    --bitrate 6000000 \
    --skip-interval 8
```

**Recording to File**:
```bash
python3 version_masr_multiclass.py \
    --source files \
    --video1 left.mp4 \
    --video2 right.mp4 \
    --mode record \
    --output-file football_match.flv \
    --bitrate 8000000 \
    --skip-interval 8
```

### Command-Line Options

```
Source Options:
  --source {cameras,files}     Input source type
  --video1 PATH                Left camera video file
  --video2 PATH                Right camera video file

Mode Options:
  --mode {panorama,virtualcam,stream,record}
                               Output mode
  --output-file PATH           Output file (record mode)
  --rtmp-url URL               RTMP URL (stream mode)

Performance Options:
  --skip-interval N            Process every Nth frame (default: 8)
  --bitrate BITRATE           Video bitrate in bps (default: 6000000)

Buffer Options:
  --buffer-duration SECONDS    Frame buffer size (default: 7)
```

### Configuration Tuning

**For Better Detection Accuracy**:
```bash
--skip-interval 4  # Process every 4th frame (higher CPU/GPU load)
```

**For Higher FPS**:
```bash
--skip-interval 16  # Process every 16th frame (lighter load)
```

**For Better Streaming Quality**:
```bash
--bitrate 8000000  # 8 Mbps (higher quality, more bandwidth)
```

**For Lower Latency**:
```bash
--buffer-duration 3  # 3-second buffer (less smoothing)
```

## Performance Optimization

### Memory Optimization

**1. Use NVMM Throughout**:
```python
# GOOD: All elements support NVMM
nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA

# BAD: Forces copy to CPU memory
videoconvert ! video/x-raw,format=RGBA
```

**2. Fixed Buffer Pools**:
- nvdsstitch: 8 pre-allocated buffers
- nvtilebatcher: 4 pre-allocated batches
- nvdsvirtualcam: 8 pre-allocated buffers

**3. EGL Resource Caching**:
All plugins cache EGL→CUDA registrations to avoid ~1-2ms overhead per frame.

### GPU Optimization

**1. VIC Offloading**:
nvdsstitch uses VIC (Video Image Compositor) for buffer copying, freeing GPU for stitching kernel.

**2. Asynchronous Processing**:
- Color correction in nvdsstitch runs in background CUDA stream
- nvtilebatcher and nvdsvirtualcam use non-blocking streams

**3. Optimal Thread Blocks**:
- nvdsstitch: 32×8 (256 threads)
- nvtilebatcher: 32×32 (1024 threads)
- nvdsvirtualcam: 32×16 (512 threads)

Empirically determined for Jetson Orin SM 8.7 architecture.

### Pipeline Optimization

**1. Parallel Branches**:
```
tee splits to:
  - Analysis branch (inference)
  - Display branch (virtual camera)
Both run in parallel without blocking each other
```

**2. Leaky Queues**:
```python
queue max-size-buffers=10 leaky=downstream
# Drops old frames if downstream is slow
```

**3. Frame Skipping**:
```python
# Analyze every 8th frame
if frame_number % 8 != 0:
    return Gst.PadProbeReturn.DROP
```

## Known Limitations

### Platform Constraints

1. **nvdsosd Rectangle Limit**: Maximum 16 rectangles per frame
   - Solution: Prioritize ball > players > referees

2. **Memory Bandwidth**: ~102 GB/s shared between CPU/GPU
   - Solution: Zero-copy NVMM, avoid CPU↔GPU transfers

3. **Thermal Throttling**: Jetson Orin can throttle under sustained load
   - Solution: Adequate cooling, power mode MAX (15W or 25W)

### Algorithm Limitations

1. **Ball Occlusion**: Cannot track ball when completely hidden
   - Mitigation: Trajectory interpolation, fallback to players

2. **Fixed Tile Positions**: 6 tiles cover field but not entire panorama
   - Trade-off: Performance vs coverage

3. **Detection Latency**: 7-second delay for smooth playback
   - Trade-off: Latency vs smoothness

### Performance Bottlenecks

1. **YOLO Inference**: ~30-40ms for 6 tiles
   - Potential: Use DLA accelerators (not implemented)

2. **Frame Buffering**: 3 GB RAM for 7-second buffer
   - Limit: Cannot extend buffer beyond ~10 seconds

3. **Encoding**: H.264 encoding adds ~5-10ms latency
   - Trade-off: Real-time display vs recording quality

## Troubleshooting

### Pipeline Doesn't Start

**Error**: "Failed to set pipeline to PLAYING state"

**Check**:
```bash
# Verify plugins are installed
gst-inspect-1.0 nvdsstitch
gst-inspect-1.0 nvtilebatcher
gst-inspect-1.0 nvdsvirtualcam

# Check DeepStream installation
gst-inspect-1.0 nvinfer
```

### Low FPS

**Symptoms**: FPS < 25, stuttering playback

**Solutions**:
1. Increase `--skip-interval` (e.g., 16)
2. Check GPU memory: `tegrastats`
3. Verify power mode: `sudo nvpmodel -q`
4. Check thermal throttling: `cat /sys/devices/virtual/thermal/thermal_zone*/temp`

### No Detections

**Symptoms**: No bounding boxes displayed

**Check**:
1. YOLO model exists: `ls -lh yolo11n_mixed_finetune_v9.engine`
2. Labels file exists: `ls -lh labels.txt`
3. Field mask exists: `ls -lh field_mask.png`
4. Inference logs: Look for "NvDsInferContext" messages

### Black Output

**Symptoms**: Display shows black screen

**Check**:
1. LUT maps exist: `ls -lh my_steach/warp_maps/`
2. Panorama dimensions match properties
3. Camera sensors detected: `v4l2-ctl --list-devices`

### Audio Sync Issues (Streaming)

**Symptoms**: Audio drifts from video

**Solutions**:
1. Use `--buffer-duration 5` (reduce buffer)
2. Check PulseAudio latency: `pacmd list-sources`
3. Restart PulseAudio: `pulseaudio -k && pulseaudio --start`

## Development Guide

### Adding New Detection Class

1. **Update labels.txt**:
```
ball
player
staff
side_referee
main_referee
new_class  # Add here
```

2. **Update TensorProcessor**:
```python
# In postprocess_yolo_output()
num_classes = 6  # Was 5
```

3. **Update Display Logic**:
```python
# In playback_draw_probe()
if obj_class == 5:  # New class
    rect.border_color = (0.0, 0.0, 1.0, 1.0)  # Blue
```

### Customizing Virtual Camera

**Modify FOV Range**:
```cpp
// In my_virt_cam/src/nvdsvirtualcam_config.h
constexpr float FOV_MIN = 40.0f;  // Change min
constexpr float FOV_MAX = 85.0f;  // Change max
```

**Change Tracking Smoothness**:
```python
# In version_masr_multiclass.py
vcam.set_property("smooth-factor", 0.5)  # 0.0-1.0
```

**Adjust Edge Offset**:
```cpp
// In my_virt_cam/src/gstnvdsvirtualcam.cpp
const float EDGE_DISTANCE = 400.0;  // Was 300
const float EDGE_OFFSET_YAW = 10.0;  // Was 8.0
```

### Custom Post-Processing

**Add Detection Filter**:
```python
# In analysis_probe()
def custom_filter(detection):
    # Filter small balls
    if detection['class'] == 0:  # Ball
        if detection['width'] < 15:
            return False
    return True

filtered_dets = [d for d in detections if custom_filter(d)]
```

**Custom Interpolation**:
```python
# In BallDetectionHistory
def custom_interpolate(self, t1, t2, target_t):
    # Cubic spline instead of parabolic
    from scipy.interpolate import CubicSpline
    # ... implementation
```

## Repository Structure

```
ds_pipeline/
├── CLAUDE.MD                          # This file
├── nvidia_jetson_orin_nx_16GB_super_arch.pdf  # Platform documentation
├── ds_doc/                            # DeepStream 7.1 documentation
│   └── 7.1/
├── new_week/
│   └── version_masr_multiclass.py    # MAIN PIPELINE
├── my_steach/                         # nvdsstitch plugin
│   ├── README.md
│   ├── src/
│   │   ├── gstnvdsstitch.cpp
│   │   ├── gstnvdsstitch.h
│   │   ├── cuda_stitch_kernel.cu
│   │   └── nvdsstitch_config.h
│   ├── libnvdsstitch.so
│   ├── Makefile
│   ├── panorama_cameras_realtime.py
│   ├── panorama_stream.py
│   └── test_fps.py
├── my_tile_batcher/                   # nvtilebatcher plugin
│   ├── README.md
│   ├── src/
│   │   ├── gstnvtilebatcher.cpp
│   │   ├── gstnvtilebatcher.h
│   │   ├── gstnvtilebatcher_allocator.cpp
│   │   ├── gstnvtilebatcher_allocator.h
│   │   └── cuda_tile_extractor.cu
│   ├── test_complete_pipeline.py
│   ├── test_performance.py
│   ├── test_simple.py
│   └── test_tilebatcher.py
├── my_virt_cam/                       # nvdsvirtualcam plugin
│   ├── README.md
│   ├── src/
│   │   ├── gstnvdsvirtualcam.cpp
│   │   ├── gstnvdsvirtualcam.h
│   │   ├── cuda_vcam_kernel.cu
│   │   └── nvdsvirtualcam_config.h
│   ├── test_full_pipeline.py
│   ├── test_virtual_camera_keyboard.py
│   ├── test_virtual_camera_sliders.py
│   ├── test_boundaries_full.py
│   └── [many other test scripts]
├── calibration/                       # Camera calibration files
├── sliser/                            # Utility scripts
└── soft_record_video/                 # Video recording tools
```

**Note**: `my_ring_buffer/` is excluded from this documentation as requested.

## References

### NVIDIA Documentation

- **DeepStream SDK**: https://developer.nvidia.com/deepstream-sdk
- **DeepStream 7.1 Guide**: `ds_doc/7.1/index.html`
- **Jetson Orin Docs**: https://developer.nvidia.com/embedded/jetson-orin
- **JetPack 6.2**: https://developer.nvidia.com/embedded/jetpack

### Technical Resources

- **GStreamer**: https://gstreamer.freedesktop.org/
- **CUDA Programming**: https://docs.nvidia.com/cuda/
- **DeepStream Python**: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
- **NVMM Memory**: CUDA for Tegra documentation (included in platform PDF)

### Papers and Algorithms

- **YOLO**: https://github.com/ultralytics/ultralytics
- **Equirectangular Projection**: https://en.wikipedia.org/wiki/Equirectangular_projection
- **Ball Tracking**: Kalman filtering, parabolic trajectory estimation

## License

Custom implementation for research and development. Uses NVIDIA proprietary SDKs (DeepStream, CUDA) which require NVIDIA hardware and software licenses.

## Contact

For questions about specific components:
- **Main Pipeline**: See `version_masr_multiclass.py` header
- **nvdsstitch**: See `my_steach/README.md`
- **nvtilebatcher**: See `my_tile_batcher/README.md`
- **nvdsvirtualcam**: See `my_virt_cam/README.md`

---

**Last Updated**: 2025-11-16
**DeepStream Version**: 7.1
**JetPack Version**: 6.2
**Platform**: NVIDIA Jetson Orin NX 16GB
