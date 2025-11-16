# DeepStream Sports Analytics Pipeline

## Project Overview

This is a **real-time AI-powered sports analytics system** built on NVIDIA DeepStream SDK 7.1, designed to run on NVIDIA Jetson Orin NX 16GB platform. The system processes dual 4K camera feeds to create 360° panoramic video, performs object detection and tracking (ball, players, referees), and provides intelligent virtual camera control with automated playback capabilities.

### Key Capabilities

- **Dual 4K Camera Stitching**: Real-time panorama generation (5700×1900px) from two Sony IMX678 cameras
- **AI-Powered Detection**: YOLOv11 multiclass object detection (ball, players, staff, referees)
- **Virtual Camera Control**: GPU-accelerated perspective extraction with intelligent ball/player tracking
- **Intelligent Buffering**: 7-second RAM buffer with timestamp synchronization for analysis and playback
- **Multi-Output Support**: Panorama view, virtual camera, RTMP streaming, and video recording

---

## Hardware Platform

### NVIDIA Jetson Orin NX 16GB Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | 1024 CUDA cores + 32 Tensor cores (Ampere architecture) @ 918 MHz |
| **CPU** | 8× ARM Cortex-A78AE @ 2.0 GHz |
| **Memory** | 16 GB LPDDR5 (unified CPU/GPU) @ 102 GB/s bandwidth |
| **AI Performance** | ~100 TOPS (INT8) |
| **DLA** | 2× NVDLA engines (~20 TOPS each) |
| **Video Decode** | 2× 4K60 HEVC/H.264/AV1 |
| **Video Encode** | 1× 4K60 HEVC/H.264 |

**Architecture Notes** (from nvidia_jetson_orin_nx_16GB_super_arch.pdf):
- Unified memory architecture: CPU and GPU share physical RAM (no separate VRAM)
- I/O coherency supported (compute capability ≥7.2)
- Memory bandwidth is shared resource - avoid unnecessary CPU↔GPU copies
- Optimal for DeepStream: all video buffers stay in GPU memory (NVMM)

### Camera System

#### Sony IMX678-AAQR1 (AR) Sensor Module

Based on camera_doc/IMX678C_Framos_Docs_documentation.pdf:

| Specification | Value |
|---------------|-------|
| **Sensor** | Sony IMX678-AAQR1 (Starvis2 technology) |
| **Resolution** | 3840×2160 (8MP / 4K) |
| **Framerate** | 72.05 FPS @ 10-bit, 60.00 FPS @ 12-bit |
| **Optical Format** | 1/1.8" |
| **Pixel Size** | 2×2 µm |
| **Shutter** | Rolling shutter (CMOS) |
| **Interface** | MIPI CSI-2 (4-lane, 2.5 Gbps/lane) |
| **Power** | Two rails: 3.8V + 1.8V (540mW max) |

#### Lens Configuration (L100A)

| Parameter | Value |
|-----------|-------|
| **Horizontal FOV** | 100° |
| **Vertical FOV** | 55° |
| **Diagonal FOV** | 114° |
| **Aperture** | F/2.7 |
| **Mount** | M12×0.5 |
| **Optical Filter** | IR cut @ 660nm |
| **Distortion** | -35.8% (F-Tan-Theta model) |
| **Operating Temp** | -30°C to +85°C |

**Dual Camera Setup**:
- 2× cameras mounted at 85° angle
- 15° downward tilt for field coverage
- ~15% overlap zone for seamless stitching
- Calibrated using 8×6 chessboard (25mm squares)

---

## System Architecture

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES (Dual 4K Cameras)                      │
│  Camera 0 (Left): 3840×2160 @ 30fps    Camera 1 (Right): 3840×2160 @ 30fps │
└──────────────┬──────────────────────────────────────┬─────────────────────────┘
               │                                      │
               ▼                                      ▼
        [nvarguscamerasrc]                   [nvarguscamerasrc]
               │                                      │
               ▼                                      ▼
        [nvvideoconvert]                     [nvvideoconvert]
         RGBA (NVMM)                          RGBA (NVMM)
               │                                      │
               └──────────────┬───────────────────────┘
                              ▼
                      [nvstreammux]
                   batch-size=2, GPU memory
                              │
                              ▼
                  ┌───────────────────────┐
                  │    MY_STEACH PLUGIN    │
                  │  Panorama Stitching    │
                  │    5700×1900 RGBA      │
                  └───────────┬────────────┘
                              │
                              ▼
                         [queue]
                     GPU buffer (NVMM)
                              │
                    ┌─────────┴─────────┐
                    │                   │
      ┌─────────────▼──────┐   ┌────────▼────────────────┐
      │  ANALYSIS BRANCH   │   │  DISPLAY BRANCH (7s lag) │
      │    (Real-time)     │   │    (Buffered playback)   │
      └─────────┬──────────┘   └───────────┬──────────────┘
                │                          │
                ▼                          ▼
      [MY_TILE_BATCHER]              [RAM Buffer]
     6×1024×1024 tiles          150-210 frames @ 30fps
                │                          │
                ▼                          ▼
         [nvinfer]                   [appsrc]
       YOLOv11n/s                   Playback
      TensorRT INT8                  pipeline
                │                          │
                ▼                          │
    [Tensor Processing]                   │
    Post-NMS multiclass:                  │
    - ball (class 0)                      │
    - player (class 1)                    │
    - staff (class 2)                     │
    - side_referee (class 3)              │
    - main_referee (class 4)              │
                │                          │
                ▼                          │
    [BallDetectionHistory]                │
     + PlayersHistory                     │
     Raw → Processed → Confirmed          │
     (10s history, interpolation)         │
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  MY_VIRT_CAM PLUGIN    │
              │ (CUDA perspective       │
              │  projection)            │
              │                         │
              │  Input: 5700×1900       │
              │  Output: 1920×1080      │
              │                         │
              │  Controls:              │
              │  - Yaw: -90° to +90°    │
              │  - Pitch: -32° to +22°  │
              │  - Roll: -28° to +28°   │
              │  - FOV: 55° to 68°      │
              │  - Auto-zoom on ball    │
              │  - Fallback to players  │
              └────────────┬────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
          [nvdsosd]              [nvdsosd]
        Panorama view         Virtual camera
        (16 objects max)         + overlays
                │                     │
                ▼                     ▼
         Output options:     Output options:
         - Display           - Display
         - RTMP stream       - RTMP stream
         - Video file        - Video file
```

### Memory Architecture

**Unified Memory Model** (GPU-resident throughout pipeline):

1. **Camera Capture**: nvarguscamerasrc → NVMM buffer pool
2. **Stitching**: my_steach → Pre-allocated 8-buffer pool (NVMM)
3. **Tile Extraction**: my_tile_batcher → Fixed 4-buffer pool (6 surfaces each)
4. **Inference**: nvinfer → TensorRT managed memory
5. **Virtual Camera**: my_virt_cam → EGL-mapped CUDA memory with LUT cache
6. **Display**: nveglglessink → Direct NVMM consumption

**No CPU copies** occur until:
- Analysis results extraction (metadata only, ~few KB)
- RAM buffering for playback (encoded H.264, ~1MB per frame)

---

## Core Components

### 1. MY_STEACH - Panorama Stitching Plugin

**Location**: `~/ds_pipeline/my_steach/`

**Function**: Stitches two 4K camera streams into seamless equirectangular panorama

#### Technical Details

- **Algorithm**: LUT-based warping with bilinear interpolation
- **Input**: 2× 3840×2160 RGBA (MIPI CSI or files)
- **Output**: 5700×1900 RGBA (configurable)
- **LUT Maps**: 6× binary files (warp_maps/*.bin):
  - lut_left_x.bin, lut_left_y.bin
  - lut_right_x.bin, lut_right_y.bin
  - weight_left.bin, weight_right.bin
- **Color Correction**: Asynchronous 2-phase system
  - Overlap zone analysis every 30 frames
  - Smoothing factor: 0.1 (10% update rate)
  - 6 coefficients: R/G/B gains per camera
- **CUDA Kernel**: `panorama_lut_kernel`
  - Block size: 32×8 threads (256 threads/block)
  - Grid: ~179×238 blocks for 5700×1900 output
  - Bandwidth: ~110 MB/frame (input + output)
- **Performance**: Real-time 30 FPS on Jetson Orin NX

#### Key Features

- Weighted blending in overlap zones
- Edge brightness boost (optional, disabled by default)
- EGL texture caching for Jetson
- Vertical/horizontal flip transformation
- Dynamic panorama size configuration

**See**: `my_steach/PLUGIN.MD` for detailed documentation

---

### 2. MY_VIRT_CAM - Virtual Camera Plugin

**Location**: `~/ds_pipeline/my_virt_cam/`

**Function**: Extracts perspective view from equirectangular panorama with intelligent tracking

#### Technical Details

- **Algorithm**: 3-stage equirectangular → perspective projection
  1. Ray generation (camera intrinsics)
  2. LUT generation (3D rotation + spherical mapping)
  3. Image remapping (nearest-neighbor sampling)

- **Input**: 6528×1632 RGBA panorama (equirectangular, 180°×54° coverage)
- **Output**: 1920×1080 RGBA (fixed Full HD)

- **CUDA Kernel**: `virtual_camera_kernel`
  - Block size: 16×16 threads (256 threads/block)
  - Grid: 8,160 blocks for 1920×1080
  - Occupancy: ~80-90% GPU utilization
  - Bandwidth: Only 1.7% of 120 GB/s capacity

#### Control Parameters

| Parameter | Range | Purpose |
|-----------|-------|---------|
| **yaw** | -90° to +90° | Horizontal pan |
| **pitch** | -32° to +22° | Vertical tilt |
| **roll** | -28° to +28° | Image rotation |
| **fov** | 55° to 68° | Zoom level |
| **smooth-factor** | 0.0 to 1.0 | Movement interpolation (default: 0.3) |

#### Auto-Zoom Formula

Based on detected ball size:
```
target_fov = BASE_FOV + (ball_radius - 20) × ZOOM_RATE
BASE_FOV = 60°
ZOOM_RATE = -0.15 (zoom in when ball is closer/larger)
```

Spherical correction for wide angles:
```
yaw_factor = cos(normalized_yaw × π/2)
effective_fov = target_fov × yaw_factor
```

#### LUT Caching System

- **Ray Cache**: 24.9 MB (FOV-dependent, 0.1° dead zone)
- **LUT Cache**: 16.6 MB (angle-dependent, 0.1° threshold)
- **EGL Mapping Cache**: 4-8 entries, <1μs lookup
- **Fixed Buffer Pool**: 8 round-robin pre-allocated buffers

**Performance**: 47.90 FPS, 20.88ms latency

**See**: `my_virt_cam/PLUGIN.MD` for detailed documentation

---

### 3. MY_TILE_BATCHER - Tile Batching Plugin

**Location**: `~/ds_pipeline/my_tile_batcher/`

**Function**: Extracts 6× 1024×1024 tiles from panorama for efficient inference batching

#### Technical Details

- **Tile Layout**: Horizontal array with 192px left offset
  ```
  Tile 0: X=192,   Y=434
  Tile 1: X=1216,  Y=434
  Tile 2: X=2240,  Y=434
  Tile 3: X=3264,  Y=434
  Tile 4: X=4288,  Y=434
  Tile 5: X=5312,  Y=434
  ```

- **Vertical Offset (Y=434)**: Calculated from field_mask.png
  - Field center: (top=438 + bottom=1454) / 2 = 946px
  - Tile center offset: 946 - 512 = 434px

- **CUDA Kernel**: `extract_tiles_kernel_multi`
  - Launch config: (32, 32, 6) blocks × (32, 32, 1) threads
  - 6,291,456 pixels processed per frame
  - Constant memory for tile positions
  - Coalesced 4-byte RGBA reads/writes

#### Batch Structure

- **Output**: NvDsBatchMeta with 6× NvDsFrameMeta
- **Memory Layout**: NVBUF_MEM_SURFACE_ARRAY (native GPU)
- **User Metadata**: TileRegionInfo attached to each frame
  ```c
  struct TileRegionInfo {
      uint tile_id;         // 0-5
      uint panorama_x;      // Source X position
      uint panorama_y;      // Source Y position
      uint tile_width;      // 1024
      uint tile_height;     // 1024
  };
  ```

#### Performance Optimizations

1. **Fixed Output Pool**: 4 pre-allocated buffers (no dynamic allocation)
2. **EGL Cache**: Hash table for input buffer registration
3. **CUDA Stream**: Asynchronous processing with event-based sync
4. **Memory Access**: 64-byte aligned pitch for GPU efficiency

**Target**: ≥30 FPS on Jetson Orin NX

**See**: `my_tile_batcher/PLUGIN.MD` for detailed documentation

---

### 4. Calibration Module

**Location**: `~/ds_pipeline/calibration/`

**Function**: Stereo camera calibration for precise panorama stitching

#### Calibration Methods

1. **Individual Camera Calibration** (recalibrate_cleaned.py)
   - Pattern: 8×6 chessboard (25mm squares)
   - Algorithm: cv2.calibrateCamera()
   - Results:
     - Left camera: RMS = 0.180 px (49 images)
     - Right camera: RMS = 0.198 px (63 images)

2. **Essential Matrix Method** (stereo_essential_matrix.py) **[PRIMARY]**
   - For wide-angle stereo (85° camera separation)
   - Algorithm: cv2.findEssentialMat() with RANSAC
   - Input: 42 synchronized image pairs
   - Results:
     - Successful pairs: 38 (90.5%)
     - RANSAC inliers: 54 points (3.0% - expected for wide-angle)
     - Measured angle: 85.19° (target: 85°)
     - Rotation: Yaw=-85.82°, Pitch=0.38°, Roll=-21.29°

3. **Standard Stereo Calibration** (stereo_calibration.py)
   - Algorithm: cv2.stereoCalibrate() with CALIB_FIX_INTRINSIC
   - Limited to narrow baselines (<20°)

#### Output Files

- **calibration_results_cleaned.json**: Individual camera parameters (K, D)
- **stereo_essential_matrix_results.json**: Stereo parameters (R, T, E, F)
- **calibration_result_standard.pkl**: Combined binary for stitching pipeline

#### Integration with Stitching

The stitcher (test_stiching.py) uses calibration data:
1. Undistortion with cv2.undistort() and optimal camera matrix
2. Feature detection (SIFT with 2,000 keypoints)
3. Feature matching (FLANN with Lowe's ratio test 0.7)
4. Homography computation (RANSAC threshold 5.0px)
5. Cylindrical or planar projection

**See**: `calibration/CALIBRATION.MD` for detailed documentation

---

### 5. MASR Inference Pipeline

**Location**: `~/ds_pipeline/new_week/version_masr_multiclass.py`

**Function**: Multi-class object detection and intelligent tracking

#### Model Configuration

- **Model**: YOLOv11n/s FP16
- **Engine**: TensorRT (yolo11n_mixed_finetune_v9.engine)
- **Batch Size**: 6 (matching tile count)
- **Input Size**: 1024×1024 per tile
- **Network Mode**: FP16 (mode=2)
- **Classes**: 5 multiclass detection
  - Class 0: ball
  - Class 1: player
  - Class 2: staff (ignore for now)
  - Class 3: side_referee (ignore for now)
  - Class 4: main_referee (ignore for now)

#### Detection Thresholds

```python
confidence_threshold = 0.25  # Pre-clustering
nms_iou_threshold = 0.45     # Inter-class overlap
topk = 100                   # Max detections per class
```

**Class-specific thresholds**:
- Ball (class 0): 0.25
- Players/Staff/Refs (classes 1-4): 0.40

#### Post-Processing Pipeline

1. **Tensor Extraction**: Extract output0 blob (21504×9 for 1024×1024 input)
2. **Multiclass Parsing**:
   ```python
   bbox_data = tensor[:, :4]        # x, y, w, h
   class_scores = tensor[:, 4:9]    # 5 class probabilities
   class_ids = argmax(class_scores, axis=1)
   confidences = max(class_scores, axis=1)
   ```
3. **Filtering**:
   - Confidence > threshold
   - Size: 8 ≤ w/h ≤ 120 pixels
   - Edge exclusion: 20px border
4. **Coordinate Transformation**: Tile-local → panorama-global
5. **NMS**: IoU threshold 0.5 (inter-tile deduplication)
6. **Field Mask Filtering**: Binary mask validation (field_mask.png)

#### Detection History System

**BallDetectionHistory** (10-second temporal buffer):

- **Raw Future History**: Incoming detections from analysis branch
- **Processed Future History**: Cleaned + interpolated trajectory
- **Confirmed History**: Detections already displayed (7s ago)

**Key Methods**:
- `add_detection()`: Stores raw detection with duplicate filtering (≤2px threshold)
- `get_detection_for_timestamp()`: Interpolates between points for smooth playback
- `_interpolate_between_points()`: Parabolic trajectory for flight (gap > 1s)
- `detect_and_remove_false_trajectories()`: Outlier removal with permanent blacklist
- `interpolate_history_gaps()`: Fills missing frames (max 10s gap)

**PlayersHistory** (center-of-mass fallback):
- Stores player positions for each timestamp
- EMA smoothing: α = 0.18 (smooth camera movement)
- Fallback target when ball lost >3s

#### Intelligent Camera Control

**Ball Tracking**:
- Speed-based auto-zoom: 300-1200 px/s → FOV adjustment
- Smooth factor: 0.3 (30% new position per frame)
- Radius smoothing: α = 0.3 for zoom stability

**Ball Loss Recovery**:
- Lost threshold: No detection for 6 frames (0.2s)
- FOV expansion: 2°/second up to max 90°
- Recovery: 6-frame confirmation before relock

**Backward Interpolation**:
- When ball reappears after long gap
- Generates synthetic trajectory for smooth camera movement
- 30 points/second linear interpolation

**See**: `new_week/INFERENCE.MD` for detailed documentation

---

## Pipeline Modes

### 1. Panorama Mode
- Full 5700×1900 view with bbox overlays
- Max 16 objects rendered (nvdsosd limitation on Jetson)
- Priority: ball (red, border=3) > players (green, border=2)
- Tiles disabled to preserve render slots

### 2. Virtual Camera Mode
- Intelligent ball/player tracking
- 1920×1080 output
- Auto-zoom based on ball size/speed
- Fallback to center-of-mass when ball lost

### 3. Streaming Mode (RTMP)
- H.264 encoding @ 6 Mbps
- RTMP push to server (e.g., rtmp://live.twitch.tv/app/{stream_key})
- Low-latency configuration

### 4. Recording Mode
- H.264 @ 6-8 Mbps to MP4 file
- 7-second buffered output (includes pre-event)
- Synchronized audio via PulseAudio (if available)

---

## Performance Benchmarks

### Measured Performance (Jetson Orin NX)

| Component | FPS | Latency | GPU Load | Memory |
|-----------|-----|---------|----------|--------|
| **Camera Capture** | 30 | - | - | 2× 66 MB |
| **Stitching (my_steach)** | 30 | ~10 ms | ~15% | 43 MB out |
| **Tile Batching** | 30 | ~1 ms | ~5% | 25 MB |
| **Inference (YOLOv11n INT8)** | 30 | ~20 ms | ~40% | Variable |
| **Virtual Camera** | 47.9 | 20.9 ms | ~10% | 8 MB |
| **Overall Pipeline** | 30 | ~100 ms | ~70% | ~10 GB |

### Memory Breakdown (16 GB total)

- **System/OS**: ~2 GB
- **DeepStream SDK**: ~1 GB
- **Video Buffers**: ~4 GB (NVMM pools + buffer)
- **TensorRT Engine**: ~2 GB (model + workspace)
- **LUT Caches**: ~0.3 GB (steach + virtcam)
- **Frame Buffer (7s @ 30fps)**: ~3 GB (210 frames × ~15 MB)
- **Available Headroom**: ~3 GB

### Bottlenecks & Optimizations

**Current Bottlenecks**:
1. Inference on 6 tiles: ~20ms (can use DLA for offload)
2. RAM buffer encoding: H.264 CPU encoder (consider nvenc)
3. Memory bandwidth: Shared 102 GB/s (avoid unnecessary copies)

**Optimizations Applied**:
- All video data in NVMM (GPU-resident)
- Fixed buffer pools (no allocation overhead)
- LUT caching (ray/coordinate regeneration avoided)
- EGL mapping cache (reduced registration calls)
- Asynchronous CUDA streams (pipelined execution)

---

## Build & Deployment

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **JetPack** | 6.2+ | NVIDIA Jetson SDK |
| **DeepStream** | 7.1 | Video analytics framework |
| **CUDA** | 12.6 | GPU compute |
| **GStreamer** | 1.0 | Media pipeline |
| **TensorRT** | 10.5+ | AI inference |
| **OpenCV** | 4.5+ | Calibration, image processing |
| **Python** | 3.8+ | Pipeline orchestration |
| **PyDS (pyds)** | 1.1.11+ | DeepStream Python bindings |

### Building Custom Plugins

**my_steach**:
```bash
cd my_steach
make clean && make
make install  # Installs to ~/.local/share/gstreamer-1.0/plugins/
```

**my_virt_cam**:
```bash
cd my_virt_cam/src
make clean && make
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$(pwd)
```

**my_tile_batcher**:
```bash
cd my_tile_batcher
make clean && make
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$(pwd)/src
```

### Verify Plugin Registration

```bash
gst-inspect-1.0 nvdsstitch
gst-inspect-1.0 nvvirtualcam
gst-inspect-1.0 nvtilebatcher
```

### Running the Pipeline

**File Sources** (testing):
```bash
cd new_week
python3 version_masr_multiclass.py \
    --source-type files \
    --video1 left_camera.mp4 \
    --video2 right_camera.mp4 \
    --display-mode virtualcam \
    --buffer-duration 7.0
```

**Live Cameras** (production):
```bash
python3 version_masr_multiclass.py \
    --source-type cameras \
    --video1 0 \
    --video2 1 \
    --display-mode virtualcam \
    --enable-analysis \
    --output-file output.mp4
```

**RTMP Streaming**:
```bash
python3 version_masr_multiclass.py \
    --source-type cameras \
    --video1 0 \
    --video2 1 \
    --display-mode stream \
    --stream-url rtmp://live.twitch.tv/app \
    --stream-key YOUR_STREAM_KEY
```

---

## Known Limitations

1. **nvdsosd Rendering**: Maximum 16 objects on Jetson (hardware limit)
2. **Inference Latency**: 6-tile batch takes ~20ms (consider DLA offload)
3. **Memory Bandwidth**: Shared 102 GB/s - careful with large panoramas
4. **Calibration**: Essential matrix T vector normalized (unknown real baseline)
5. **Camera Synchronization**: Software sync only (hardware sync recommended for production)

---

## Project Structure

```
ds_pipeline/
├── calibration/                  # Stereo calibration tools
│   ├── stereo_essential_matrix.py
│   ├── test_stiching.py
│   ├── calibration_result_standard.pkl
│   └── CALIBRATION.MD
├── my_steach/                    # Panorama stitching plugin
│   ├── src/
│   │   ├── gstnvdsstitch.cpp    # GStreamer plugin
│   │   └── cuda_stitch_kernel.cu # CUDA stitching
│   ├── Makefile
│   ├── libnvdsstitch.so
│   └── PLUGIN.MD
├── my_virt_cam/                  # Virtual camera plugin
│   ├── src/
│   │   ├── gstnvvirtualcam.cpp  # GStreamer plugin
│   │   └── virtual_camera_kernel.cu # CUDA projection
│   ├── Makefile
│   └── PLUGIN.MD
├── my_tile_batcher/              # Tile extraction plugin
│   ├── src/
│   │   ├── gstnvtilebatcher.cpp
│   │   └── cuda_tile_extractor.cu
│   ├── Makefile
│   └── PLUGIN.MD
├── new_week/                     # Main inference pipeline
│   ├── version_masr_multiclass.py
│   ├── config_infer.txt
│   ├── labels.txt
│   ├── yolo11n_mixed_finetune_v9.engine
│   └── INFERENCE.MD
├── camera_doc/                   # Camera specifications
│   └── IMX678C_Framos_Docs_documentation.pdf
├── ds_doc/                       # DeepStream documentation
│   └── 7.1/                     # HTML reference
├── nvidia_jetson_orin_nx_16GB_super_arch.pdf
├── field_mask.png               # Binary field mask for filtering
└── CLAUDE.MD                    # This file
```

---

## References

1. **NVIDIA DeepStream SDK 7.1**: https://docs.nvidia.com/metropolis/deepstream/dev-guide/
2. **Jetson Orin NX Datasheet**: nvidia_jetson_orin_nx_16GB_super_arch.pdf
3. **Camera Documentation**: camera_doc/IMX678C_Framos_Docs_documentation.pdf
4. **Platform Architecture**: See embedded PDF analysis on unified memory
5. **YOLOv11 Documentation**: https://docs.ultralytics.com/models/yolo11/

---

## Contact & Contribution

This project is a production sports analytics system. For technical questions or contributions, refer to individual plugin documentation files:
- Stitching: `my_steach/PLUGIN.MD`
- Virtual Camera: `my_virt_cam/PLUGIN.MD`
- Tile Batching: `my_tile_batcher/PLUGIN.MD`
- Calibration: `calibration/CALIBRATION.MD`
- Inference Pipeline: `new_week/INFERENCE.MD`

---

**Last Updated**: 2025-11-16
**Platform**: NVIDIA Jetson Orin NX 16GB
**DeepStream**: 7.1
**CUDA**: 12.6
