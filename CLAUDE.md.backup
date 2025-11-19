# DeepStream Sports Analytics Pipeline


## CLAUDE PROJECT RULES — POLYCUBE (STRICT MODE)

*DeepStream • CUDA • TensorRT • NVIDIA Jetson Orin NX 16GB • Sports Analytics*

## 0. Mission Statement

Claude must act as a reliable junior engineer inside a highly optimized GPU-centric real-time video analytics system. All work must be:

* correct
* minimal
* stable
* deterministic
* Jetson-compatible
* aware of pipeline constraints

Claude prioritizes **precision**, **plans**, **questions**, and **architecture**.

## 1. Core Behavior Rules

* Follow workflow: **Plan → Review → Approval → Execute**.
* Never write code without approved plan.
* Ask clarification questions when needed.
* Analyze existing code before changes.
* Keep changes minimal.
* Respect all DeepStream, CUDA, TensorRT, Jetson constraints.
* Provide concise, technical responses.

If unsure — ask.

## 2. Operational Modes

### 2.1 Plan Mode (mandatory)

Before writing or modifying code:

* Enter Plan Mode
* Produce a structured plan:

  1. Goal
  2. Current state analysis
  3. Step-by-step actions
  4. Files impacted
  5. Risks
  6. Clarifying questions
* Wait for approval.

### 2.2 Thinking Levels

* **think** — local tasks
* **think hard** — multi-module tasks
* **think harder** — performance/architecture tasks
* **ultrathink** — only with explicit permission

## 3. DeepStream Pipeline Rules

Correct pipeline order:

```
1. Camera → convert → streammux
2. my_steach (stitch)
3. my_tile_batcher (6×1024 tiles)
4. nvinfer (YOLO FP16)
5. MASR history (raw → processed → confirmed)
6. virtual camera (my_virt_cam)
7. 7s playback branch
```

### 3.1 Memory Model

* Always use NVMM
* No CPU pixel copies
* CPU only for metadata & light tasks
* Respect 16GB unified RAM

### 3.2 Latency Budgets

* Stitch: ≤10ms
* Tile batcher: ≤1ms
* Inference: ≤20ms
* Virtual camera: ≤22ms
* Entire pipeline: 30 FPS, ≤100ms latency

### 3.3 Metadata Rules

* Only small metadata
* Use NvDsUserMeta / FrameMeta / BatchMeta
* Maintain timestamp consistency

### 3.4 Branch Synchronization

Do not break:

* Real-time analysis branch
* 7-second playback branch

## 4. CUDA / C++ Plugin Rules

### 4.1 CUDA Requirements

* Keep existing block/grid sizes unless justified
* Coalesced memory access
* Avoid warp divergence
* No dynamic allocations
* Maintain LUT caching (no per-frame regen)

### 4.2 GStreamer Plugin Requirements

* Respect memory ownership
* Null-check everything
* Maintain thread safety
* Avoid creating new buffer pools

## 5. Python Rules

* Python 3.8+ only
* Functions ≤60 lines
* Files ≤400 lines
* No circular imports
* No heavy CPU ops in callbacks
* No deep copies of big structures
* No blocking I/O in hot paths

## 6. YOLO / TensorRT Rules

* Inference allowed: FP16
* Never use FP32 on Jetson
* Batch size = 6
* Tile size = 1024×1024
* ONNX exported on server, engine built on Jetson

Forbidden:

* PyTorch inference on Jetson
* Tile geometry changes
* NMS structure changes

## 7. MASR Tracking Rules

Maintain:

* Triple-buffer system
* Timestamp continuity
* Interpolation consistency
* Ballistics model
* Fallback to players
* Field mask filtering
* Inter-tile NMS

## 8. File Modification Protocol

* Modify only approved files
* Provide patches in diff format:

```
File: new_week/core/history_manager.py
Patch:
+ added_line
- removed_line
```

## 9. Communication Rules

* Ask when unsure
* Respond concisely, technically
* No hallucinations

## 10. Strict Prohibitions

Claude must NOT:

* Generate code before plan approval
* Propose refactors without permission
* Break NVMM zero-copy path
* Add CPU-heavy logic in GPU hot paths
* Change pipeline topology
* Use unavailable libraries
* Output pseudocode

## 11. Jetson Constraints

* Unified RAM 16GB
* GPU <70% load
* Avoid large allocations
* Avoid Python-heavy loops
* Respect FP16 memory constraints
* NVENC/NVDEC bandwidth limited
* nvdsosd: max 16 boxes

## 12. Improvement Proposal Protocol

1. Problem (file + line)
2. Cause
3. Impact
4. Solution
5. Risks
6. Patch

## 13. Documentation Using
1. Search the documentation for every aspect in docs/ directory.
2. Explore a documentation before coding, follow the documentation strictly.
3. If you can not find some documentation in docs/ or in any projec dirertory - search on internet.
4. Create the docs/DOCs_NOTES.md if you need note or memorize something in documentation.

## 14. Conflict Resolution

If rules conflict: Architecture > Performance > Readability > Aesthetics If unsure — ask.

## 15. Allowed Creative Behavior

Claude may propose optimizations or improvements, but must not implement them without approval.



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
│                         INPUT SOURCES (Dual 4K Cameras)                     │
│  Camera 0 (Left): 3840×2160 @ 30fps    Camera 1 (Right): 3840×2160 @ 30fps  │
└──────────────┬──────────────────────────────────────┬───────────────────────┘
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
                  ┌────────────────────────┐
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
      ┌─────────────▼──────┐   ┌────────▼─────────────────┐
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
      TensorRT FP16                  pipeline
                │                          │
                ▼                          │
    [Tensor Processing]                    │
    Post-NMS multiclass:                   │
    - ball (class 0)                       │
    - player (class 1)                     │
    - staff (class 2)                      │
    - side_referee (class 3)               │
    - main_referee (class 4)               │
                │                          │
                ▼                          │
    [BallDetectionHistory]                 │
     + PlayersHistory                      │
     Raw → Processed → Confirmed           │
     (10s history, interpolation)          │
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  MY_VIRT_CAM PLUGIN     │
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

**See**: `my_steach/PLUGIN.md` for detailed documentation

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

| Parameter         | Range        | Purpose                               |
|-------------------|--------------|---------------------------------------|
| **yaw**           | -90° to +90° | Horizontal pan                        |
| **pitch**         | -32° to +22° | Vertical tilt                         |
| **roll**          | -28° to +28° | Image rotation                        |
| **fov**           | 55° to 68°   | Zoom level                            |
| **smooth-factor** | 0.0 to 1.0   | Movement interpolation (default: 0.3) |

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

**See**: `my_virt_cam/PLUGIN.md` for detailed documentation

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

**See**: `my_tile_batcher/PLUGIN.md` for detailed documentation

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

**See**: `calibration/CALIBRATION.md` for detailed documentation

---

### 5. MASR Inference Pipeline (Refactored Modular Architecture)

**Location**: `~/ds_pipeline/new_week/`

**Function**: Multi-class object detection and intelligent tracking with modular design

#### Architecture Overview

The pipeline has been **refactored into a modular architecture** with **76% code reduction** (from 3,015 to 712 lines), maintaining 100% API compatibility while improving maintainability and testability.

**Main Entry Point**: `version_masr_multiclass.py` (712 lines)

#### Modular Structure

**core/** - Detection & History Management
- `history_manager.py` (15KB) - Ball detection history with 3-tier buffering
- `players_history.py` (1.8KB) - Player center-of-mass tracking
- `detection_storage.py` (9.4KB) - Three-tier storage (raw, processed, confirmed)
- `trajectory_filter.py` (12KB) - Outlier detection and blacklisting
- `trajectory_interpolator.py` (10KB) - Smooth trajectory interpolation

**pipeline/** - GStreamer Pipeline Building
- `pipeline_builder.py` (16KB) - Analysis pipeline construction
- `playback_builder.py` (15KB) - Playback/display pipeline construction
- `config_builder.py` (3.7KB) - YOLO config generation
- `buffer_manager.py` (18KB) - 7-second RAM buffer for frame/audio

**processing/** - YOLO Analysis
- `analysis_probe.py` (27KB) - Analysis probe callback and detection aggregation
- `tensor_processor.py` (5.7KB) - YOLO tensor parsing and post-processing

**rendering/** - Display & Virtual Camera
- `display_probe.py` (20KB) - Panorama rendering with multi-class bboxes
- `virtual_camera_probe.py` (17KB) - Virtual camera control and ball tracking

**utils/** - General Utilities
- `field_mask.py` (1.4KB) - Field mask validation
- `csv_logger.py` (1.6KB) - Detection event logging
- `nms.py` (2.3KB) - Non-maximum suppression

#### Model Configuration

- **Model**: YOLOv11n FP16
- **Engine**: TensorRT (yolo11n_mixed_finetune_v9.engine - 8.1MB)
- **Batch Size**: 6 (matching tile count)
- **Input Size**: 1024×1024 per tile
- **Network Mode**: FP16 (mode=2)
- **Classes**: 5 multiclass detection
  - Class 0: ball
  - Class 1: player
  - Class 2: staff
  - Class 3: side_referee
  - Class 4: main_referee

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

**See**:
- `new_week/INFERENCE.md` - Inference pipeline documentation
- `new_week/README_REFACTORING.md` - Refactoring overview
- `new_week/refactoring_reference.md` - Detailed delegation map
- `new_week/pipeline/BUFFER_MANAGER_USAGE.md` - Buffer manager guide

---

### 6. Buffer Manager

**Location**: `~/ds_pipeline/new_week/pipeline/buffer_manager.py`

**Function**: 7-second intelligent buffering system for analysis/playback synchronization

#### Technical Details

- **Buffer Duration**: 7 seconds (configurable)
- **Frame Storage**: 210 frames @ 30fps
- **Memory**: ~3 GB (210 frames × ~15 MB H.264 encoded)
- **Audio Support**: Synchronized audio buffering via PulseAudio

#### Key Features

**Frame Buffering**:
- Stores encoded H.264 frames with timestamps
- Fixed-size circular buffer with automatic overflow handling
- Deep copy management for frame data safety

**Playback Synchronization**:
- Timestamp-based frame retrieval
- 7-second lag between analysis and display branches
- Smooth playback with frame interpolation

**Audio Integration**:
- PulseAudio source synchronization
- Audio/video timestamp alignment
- Optional audio muting in analysis branch

**Performance Notes**:
- Deep copies identified as CPU bottleneck (CODEX report)
- O(n) timestamp scans flagged for optimization
- Future optimization: Ring buffer with binary search

**See**: `new_week/pipeline/BUFFER_MANAGER_USAGE.md` for detailed API

---

### 7. Soft Record Video Module

**Location**: `~/ds_pipeline/soft_record_video/`

**Function**: Synchronized dual 4K camera recording utilities

#### Key Scripts

- **`synced_dual_record.py`** (27KB) - Primary synchronized recording
  - Hardware master/slave synchronization
  - Sony IMX678/IMX477 camera support
  - 4K @ 30fps recording
  - Synchronized start/stop

- **`fixed_exposure_recorder.py`** (20KB) - Fixed exposure recording
  - Manual exposure control
  - Consistent lighting conditions
  - Calibration-ready output

- **`check_frame_sync.py`** - Frame synchronization verification
  - Validates timestamp alignment
  - Detects frame drops
  - Reports sync drift

#### Features

- **Hardware Synchronization**: Master camera triggers slave via GPIO
- **Dual Output**: Simultaneous left/right camera recording
- **MP4 Format**: H.264/HEVC encoding with nvenc
- **Metadata**: Embedded timestamp and camera ID
- **Recovery**: Auto-restart on camera failure

**Shell Utilities**:
- `fix_mp4.sh` - Repair corrupted MP4 files
- `test_sync.sh` - Synchronization testing script

---

### 8. Sliser Module

**Location**: `~/ds_pipeline/sliser/`

**Function**: Panorama and tile extraction for dataset creation

#### Key Scripts

- **`panorama_tiles_saver.py`** (16KB) - Main extraction utility
  - Saves full panorama (5700×1900 JPEG)
  - Extracts 6× 1024×1024 tiles
  - Configurable frame interval
  - Sequential numbering system

**Output Structure**:
```
output/
├── panorama_0000.jpg
├── panorama_0001.jpg
├── tile0_0000.jpg
├── tile0_0001.jpg
├── tile1_0000.jpg
...
├── tile5_0000.jpg
└── tile5_0001.jpg
```

**Use Cases**:
- Training dataset creation
- Manual annotation preparation
- Quality assurance review
- Stitching validation

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
| **Inference (YOLOv11n)** | 30 | ~20 ms | ~40% | Variable |
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
│   ├── stereo_essential_matrix.py      # PRIMARY: Wide-angle calibration
│   ├── stereo_calibration.py           # Standard stereo calibration
│   ├── recalibrate_cleaned.py          # Individual camera calibration
│   ├── test_stiching.py                # Stitching integration test
│   ├── calibration_result_standard.pkl # Binary calibration data
│   └── CALIBRATION.md                  # Calibration documentation
│
├── my_steach/                    # Panorama stitching plugin
│   ├── src/
│   │   ├── gstnvdsstitch.cpp           # GStreamer plugin (1,427 lines)
│   │   ├── cuda_stitch_kernel.cu       # CUDA stitching (765 lines)
│   │   ├── gstnvdsstitch_allocator.cpp # Memory allocator (595 lines)
│   │   └── *.h                         # Header files
│   ├── Makefile                        # Build system
│   ├── libnvdsstitch.so               # Compiled plugin
│   ├── panorama_stream.py             # File-based test
│   ├── panorama_cameras_realtime.py   # Live camera test
│   └── PLUGIN.md                      # Plugin documentation
│
├── my_tile_batcher/              # Tile extraction plugin
│   ├── src/
│   │   ├── gstnvtilebatcher.cpp        # Plugin (1,733 lines)
│   │   ├── cuda_tile_extractor.cu      # CUDA kernel (150 lines)
│   │   ├── gstnvtilebatcher_allocator.cpp
│   │   └── *.h
│   ├── Makefile
│   ├── libnvtilebatcher.so            # Compiled plugin
│   ├── test_tilebatcher.py            # Standalone test
│   ├── test_complete_pipeline.py      # Full pipeline test
│   └── PLUGIN.md
│
├── my_virt_cam/                  # Virtual camera plugin
│   ├── src/
│   │   ├── gstnvdsvirtualcam.cpp       # Plugin (2,100+ lines)
│   │   ├── cuda_virtual_cam_kernel.cu  # CUDA kernel (350 lines)
│   │   ├── gstnvdsvirtualcam_allocator.cpp
│   │   ├── nvds_ball_meta.h            # Ball metadata
│   │   └── *.h
│   ├── Makefile
│   ├── libnvdsvirtualcam.so           # Compiled plugin
│   ├── test_virtual_camera_sliders.py # Interactive GUI test
│   ├── test_virtual_camera_keyboard.py
│   ├── calculate_safe_boundaries.py
│   └── PLUGIN.md
│
├── new_week/                     # Main inference pipeline (REFACTORED)
│   ├── version_masr_multiclass.py      # Main entry (712 lines)
│   ├── version_masr_multiclass_REFACTORED.py  # Modular version
│   ├── version_masr_multiclass_ORIGINAL_BACKUP.py  # Original (3,015 lines)
│   │
│   ├── core/                           # Detection & History
│   │   ├── history_manager.py
│   │   ├── players_history.py
│   │   ├── detection_storage.py
│   │   ├── trajectory_filter.py
│   │   └── trajectory_interpolator.py
│   │
│   ├── pipeline/                       # GStreamer Builders
│   │   ├── pipeline_builder.py
│   │   ├── playback_builder.py
│   │   ├── config_builder.py
│   │   ├── buffer_manager.py
│   │   └── BUFFER_MANAGER_USAGE.md
│   │
│   ├── processing/                     # YOLO Analysis
│   │   ├── analysis_probe.py
│   │   └── tensor_processor.py
│   │
│   ├── rendering/                      # Display & Camera
│   │   ├── display_probe.py
│   │   └── virtual_camera_probe.py
│   │
│   ├── utils/                          # Utilities
│   │   ├── field_mask.py
│   │   ├── csv_logger.py
│   │   └── nms.py
│   │
│   ├── config_infer.txt                # YOLO config
│   ├── labels.txt                      # Class labels
│   ├── auto_restart.sh                 # Auto-restart script
│   ├── INFERENCE.md                    # Pipeline docs
│   ├── README_REFACTORING.md           # Refactoring guide
│   └── refactoring_reference.md        # Detailed delegation map
│
├── models/                       # AI Models
│   └── yolo11n_mixed_finetune_v9.engine  # YOLOv11n FP16 (8.5MB)
│
├── soft_record_video/            # Dual camera recording
│   ├── synced_dual_record.py           # Hardware-synced recording
│   ├── synced_dual_record_robust.py
│   ├── fixed_exposure_recorder.py
│   ├── dual_record.py
│   ├── check_frame_sync.py
│   ├── fix_mp4.sh                      # MP4 repair utility
│   └── test_sync.sh
│
├── sliser/                       # Panorama tile saver
│   ├── panorama_tiles_saver.py         # Dataset creation
│   └── test_gstreamer_import.py
│
│
│
├── docs/                         # Documentation and reports
│   ├── reports/                        # Analysis reports
│   │    ├── CODEX_report.md            # CPU performance analysis
│   │    ├── COMPILED_CODEX_REPORT.md   # Compiled CPU analysis
│   │    ├── DEEPSTREAM_CODE_REVIEW.md  # Code review findings
│   │    └── Performance_report.md      # Performance benchmarks
│   ├── ds_doc/                         # DeepStream documentation
│   ├── 7.1/                            # HTML reference
│   ├── camera_doc/                     # Camera specifications
│   │    └── IMX678C_Framos_Docs_documentation.pdf
    └── hw_arch/                        # Platform harware specifications and documentation
         ├── nvidia_jetson_orin_nx_16GB_super_arch.pdf
         └── nvidia_jetson_orin_nx_16GB_super_arch.txt

│
├── CLAUDE.md                     # This file (main documentation)
├── architecture.md               # System architecture documentation
├── decisions.md                  # Architectural Decision Records (ADRs)
├── plan.md                       # Master project plan and roadmap
├── todo.md                       # Current tasks and backlog
├── nvidia_jetson_orin_nx_16GB_super_arch.pdf
└── nvidia_jetson_orin_nx_16GB_super_arch.txt
```

---


## Code Review & Performance Reports

The codebase has undergone comprehensive analysis and review, documented in the following reports:

### docs/reports/CODEX_report.md - CPU Performance Analysis

**Key Findings**: 6 high-CPU load paths identified

1. **BufferManager Deep Copies** (buffer_manager.py:116)
   - Issue: Deep copying ~15MB frames on every buffer add
   - Impact: Significant CPU overhead in frame buffering
   - Recommendation: Use shallow copies or zero-copy techniques

2. **Playback O(n) Scans** (buffer_manager.py:156)
   - Issue: Linear timestamp search through buffer
   - Impact: Performance degrades with buffer size
   - Recommendation: Binary search or ring buffer with indexing

3. **Heavy Python Post-Processing** (analysis_probe.py, display_probe.py)
   - Issue: Extensive Python processing in critical path
   - Impact: CPU bottleneck in detection pipeline
   - Recommendation: Move to C++ or use Numba JIT compilation

4. **Field Mask Validation** (field_mask.py)
   - Issue: Per-detection mask lookup
   - Impact: O(n) overhead for n detections
   - Recommendation: GPU-accelerated mask checking

5. **NMS Implementation** (nms.py)
   - Issue: Pure Python NMS with nested loops
   - Impact: O(n²) complexity for overlapping detections
   - Recommendation: Use DeepStream native NMS or GPU implementation

6. **CSV Logging** (csv_logger.py)
   - Issue: Synchronous file I/O in probe callbacks
   - Impact: Frame drops under high detection load
   - Recommendation: Async logging or buffered writes

**See**: `docs/reports/CODEX_report.md` for full analysis

---

### docs/reports/DEEPSTREAM_CODE_REVIEW.md - DeepStream 7.1 Compliance

**Critical Issues** (15 findings):
- Memory leak potential in probe callbacks
- Missing null checks in metadata iteration
- Improper GStreamer state management
- Unsafe buffer pool handling

**Important Issues** (8 findings):
- Suboptimal batch processing
- Missing error recovery paths
- Inconsistent metadata handling
- Thread safety concerns

**Recommendations** (12 items):
- Migrate to DeepStream 7.1 best practices
- Implement proper error handling
- Add comprehensive logging
- Use native DeepStream analytics

**See**: `docs/reports/DEEPSTREAM_CODE_REVIEW.md` for detailed findings

---

### docs/reports/Performance_report.md - Comprehensive Benchmarks

**System-Level Metrics**:
- Overall pipeline: 30 FPS @ 70% GPU load
- End-to-end latency: ~100ms (camera to display)
- Memory usage: ~10 GB / 16 GB
- Available headroom: ~3 GB

**Component Breakdown**:
- Stitching: 30 FPS, 10ms latency, 15% GPU
- Tile batching: 30 FPS, 1ms latency, 5% GPU
- Inference: 30 FPS, 20ms latency, 40% GPU
- Virtual camera: 47.9 FPS, 20.9ms latency, 10% GPU

**Optimization Opportunities**:
1. DLA offload for YOLO inference (reduce GPU load)
2. H.264 hardware encoding for buffer (reduce CPU load)
3. LUT compression for memory savings
4. Multi-stream batching for throughput

**See**: `docs/reports/Performance_report.md` for full benchmarks

---

## References

### Official Documentation

1. **NVIDIA DeepStream SDK 7.1**: https://docs.nvidia.com/metropolis/deepstream/7.1/index.html
2. **YOLOv11 Documentation**: https://docs.ultralytics.com/models/yolo11/
3. **GStreamer 1.0 Reference**: https://gstreamer.freedesktop.org/documentation/

### Hardware Specifications

4. **Jetson Orin NX Datasheet PDF**: nvidia_jetson_orin_nx_16GB_super_arch.pdf
4.5 **Jetson Orin NX Datasheet TXT**: nvidia_jetson_orin_nx_16GB_super_arch.txt
5. **Sony IMX678 Camera**: docs/camera_doc/IMX678C_Framos_Docs_documentation.pdf

### Project Documentation

6. **Main Documentation**: `CLAUDE.md` (this file)
7. **System Architecture**: `architecture.md`
8. **Architectural Decisions**: `decisions.md`
9. **Master Plan & Roadmap**: `plan.md`
10. **Current Tasks & Backlog**: `todo.md`
11. **Stitching Plugin**: `my_steach/PLUGIN.md`
12. **Virtual Camera Plugin**: `my_virt_cam/PLUGIN.md`
13. **Tile Batcher Plugin**: `my_tile_batcher/PLUGIN.md`
14. **Calibration Guide**: `calibration/CALIBRATION.md`
15. **Inference Pipeline**: `new_week/INFERENCE.md`
16. **Refactoring Guide**: `new_week/README_REFACTORING.md`
17. **Refactoring Reference**: `new_week/refactoring_reference.md`
18. **Buffer Manager**: `new_week/pipeline/BUFFER_MANAGER_USAGE.md`

### Analysis Reports

19. **CPU Performance Analysis**: `docs/reports/CODEX_report.md`
20. **Code Review Findings**: `docs/reports/DEEPSTREAM_CODE_REVIEW.md`
21. **Performance Benchmarks**: `docs/reports/Performance_report.md`

---

## Development & Testing

### Test Suite

The project includes **50+ test scripts** across all modules:

**Plugin Testing**:
- `my_steach/panorama_stream.py` - File-based stitching
- `my_steach/panorama_cameras_realtime.py` - Live camera stitching
- `my_virt_cam/test_virtual_camera_sliders.py` - Interactive GUI testing
- `my_tile_batcher/test_complete_pipeline.py` - Full pipeline validation

**Performance Testing**:
- `my_steach/test_fps.py` - FPS measurement
- `my_tile_batcher/test_performance.py` - Benchmarking
- `my_virt_cam/test_ball_positions.py` - Tracking validation

**Calibration Testing**:
- `calibration/check_calibration_quality.py` - Quality assessment
- `calibration/test_calibration_visual.py` - Visual validation
- `calibration/test_undistortion.py` - Undistortion verification

**Recording Testing**:
- `soft_record_video/check_frame_sync.py` - Sync verification
- `soft_record_video/test_sync.sh` - Synchronization testing

### Shell Utilities

**Automation**:
- `new_week/auto_restart.sh` - Automatic restart on failure
- `new_week/rest_cameras.sh` - Camera reset utility

**Maintenance**:
- `soft_record_video/fix_mp4.sh` - MP4 file repair
- `new_week/test_stitch_ringbuffer.sh` - Ring buffer testing

---

## Contact & Contribution

This project is a production sports analytics system. For technical questions or contributions, refer to individual documentation files:

**Core Documentation**:
- Main: `CLAUDE.md` (this file)
- Architecture: `architecture.md`
- Decisions: `decisions.md`
- Plan: `plan.md`
- TODO: `todo.md`
- DeepStream 7.1: 'docs/ds_doc/7.1/'
- Hardware: 'docs/hw_arch'
- Cameras: 'docs/camera_doc'

**Component Documentation**:
- Stitching: `my_steach/PLUGIN.md`
- Virtual Camera: `my_virt_cam/PLUGIN.md`
- Tile Batching: `my_tile_batcher/PLUGIN.md`
- Calibration: `calibration/CALIBRATION.md`
- Inference: `new_week/INFERENCE.md`
- Refactoring: `new_week/README_REFACTORING.md`
- Buffer Manager: `new_week/pipeline/BUFFER_MANAGER_USAGE.md`

**Analysis Reports**:
- CPU Analysis: `docs/reports/CODEX_report.md`
- Code Review: `docs/reports/DEEPSTREAM_CODE_REVIEW.md`
- Performance: `docs/reports/Performance_report.md`

---

**Last Updated**: 2025-11-17
**Platform**: NVIDIA Jetson Orin NX 16GB
**DeepStream**: 7.1
**CUDA**: 12.6
**Architecture**: Refactored modular design (76% code reduction)
