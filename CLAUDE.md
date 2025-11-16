# CLAUDE.md - AI Assistant Guide for ds_pipeline

> **Last Updated:** 2025-11-16
> **Repository:** ds_pipeline - Real-time Sports Video Analytics Pipeline
> **Target Platform:** NVIDIA Jetson AGX Orin (SM 87)
> **Primary Framework:** DeepStream 7.1 + GStreamer

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Technology Stack](#technology-stack)
5. [Development Environment](#development-environment)
6. [Key Conventions](#key-conventions)
7. [Common Workflows](#common-workflows)
8. [Testing Guidelines](#testing-guidelines)
9. [Important Gotchas](#important-gotchas)
10. [File Structure Reference](#file-structure-reference)

---

## Repository Overview

### Purpose
This repository implements a **real-time sports video analytics pipeline** for football/soccer analysis. It performs:

- **Stereo panoramic stitching** from dual 4K MIPI CSI cameras (IMX678, 100° FOV each)
- **Multi-class object detection** (ball, players, staff, referees) using YOLO11n
- **Intelligent ball tracking** with trajectory interpolation and player center-of-mass fallback
- **Virtual camera following** that automatically tracks the ball across the panorama
- **3-second replay buffer** for delayed playback (analysis on live, recording on delayed)
- **Hardware-accelerated processing** using CUDA and NVMM (GPU-resident memory)

### Pipeline Flow
```
Dual 4K Cameras (3840×2160 each)
    ↓
Panorama Stitching (5700×1900)
    ↓
[Split into 2 paths]
    ↓                           ↓
Tile Extraction (6×1024×1024)   Ring Buffer (3-second delay)
    ↓                           ↓
YOLO Detection                  Delayed Panorama
    ↓                           ↓
Ball Tracking ──────────────→ Virtual Camera (1920×1080)
                                ↓
                            H.264/H.265 Encoding
```

### Performance Targets
- **FPS:** 45-47 FPS (target: 50+ FPS) on full pipeline
- **Latency:** ~20ms for virtual camera rendering
- **Memory:** ~2-3GB GPU memory footprint
- **Detection:** Real-time YOLO inference on 6 tiles

---

## Architecture

### System Architecture

The system is built as a modular GStreamer pipeline with **4 custom plugins**:

1. **nvdsstitch** (`my_steach/`) - Panorama stitching from dual cameras
2. **nvdsvirtualcam** (`my_virt_cam/`) - Virtual camera extraction from panorama
3. **nvdsringbuf** (`my_ring_buffer/`) - GPU-resident 3-second delay buffer
4. **nvtilebatcher** (`my_tile_batcher/`) - Tile extraction for YOLO detection

### Design Principles

1. **GPU-Resident Processing**
   - All video data stays in NVMM (GPU memory) throughout the pipeline
   - No CPU RAM transfers except for metadata
   - Achieves minimal latency and maximum throughput

2. **LUT-Based Rendering**
   - Pre-computed lookup tables for all warping/projection operations
   - Cached and only recalculated when parameters change
   - Enables real-time performance on complex transformations

3. **Fixed Buffer Pools**
   - 8-10 buffers per stage with round-robin allocation
   - Prevents memory fragmentation and allocation overhead
   - Guarantees predictable memory usage

4. **Separation of Concerns**
   - Each plugin has a single, well-defined responsibility
   - Metadata flows separately from video data
   - Analysis on live feed, recording on delayed feed

---

## Component Details

### 1. my_steach (Panorama Stitching)

**Location:** `/home/user/ds_pipeline/my_steach/`
**Plugin:** `libnvdsstitch.so` (97KB)
**Language:** C++/CUDA

**Purpose:** Stitches dual 4K camera feeds into a single panoramic view using spherical projection.

**Key Files:**
- `src/cuda_stitch_kernel.cu` - CUDA kernel for warping and blending
- `nvdsstitch_config.h` - Configuration constants
- `libnvdsstitch.so` - Compiled GStreamer plugin

**Input:** 2× 3840×2160 RGBA streams (NVMM)
**Output:** 5700×1900 RGBA panorama (NVMM)

**Implementation Details:**
- Spherical projection with weight-based blending in overlap region
- Requires pre-computed warp maps in `warp_maps/*.bin`
- Uses camera calibration from `calibration/calibration_result_standard.pkl`
- 85.19° angle between cameras with ~15° overlap

**Test Scripts:**
- `panorama_cameras_realtime.py` - Real-time dual camera panorama
- `panorama_stream.py` - File-based panorama testing
- `test_fps.py` - Performance benchmarking

**Documentation Status:** ⚠️ Empty (README.md is 1 line)

---

### 2. my_virt_cam (Virtual Camera)

**Location:** `/home/user/ds_pipeline/my_virt_cam/`
**Plugin:** `src/libnvdsvirtualcam.so` (91KB)
**Language:** C++/CUDA

**Purpose:** Extracts a 1920×1080 virtual camera view from the panorama with controllable angles and auto-follow mode.

**Key Files:**
- `src/gstnvdsvirtualcam.cpp` - Main plugin implementation (88KB)
- `src/cuda_virtual_cam_kernel.cu` - CUDA projection kernel
- `src/nvdsvirtualcam_config.h` - Configuration constants
- `src/Makefile` - Build system

**Input:** 5700×1900 panorama (equirectangular projection, NVMM)
**Output:** 1920×1080 virtual camera view (NVMM)

**Controllable Parameters:**
- **Yaw:** -90° to +90° (pan left/right across field)
- **Pitch:** -27° to +27° (tilt up/down)
- **Roll:** -28° to +28° (camera tilt)
- **FOV:** 55° to 68° (zoom in/out)

**Auto-Follow Mode:**
- Receives ball coordinates via GstMeta from YOLO detection
- Calculates required camera angles to center the ball
- Applies smooth transitions to avoid jerky movements
- Falls back to player center-of-mass when ball is not detected

**Performance:**
- 45-47 FPS on full pipeline
- CUDA block size: 32×16 threads
- LUT caching enabled (only recalculates on angle change)

**Test Scripts:**
- `test_virtual_camera_sliders.py` - Interactive angle adjustment
- `test_virtual_camera_interactive.py` - Keyboard control
- `test_boundaries_full.py` - Edge case testing
- `test_full_pipeline.py` - End-to-end integration

**Documentation Status:** ✅ Excellent (30+ markdown files)

**Key Documentation:**
- `README.md` - Comprehensive overview (364 lines)
- `STABILITY_FIXES.md` - Production-ready fixes
- `FOV_TUNING_GUIDE.md` - Camera tuning instructions
- `OPTIMIZATION_RESULTS.md` - Performance analysis
- `BOUNDARY_FIX_FINAL.md` - Edge handling logic

---

### 3. my_ring_buffer (Delayed Buffer)

**Location:** `/home/user/ds_pipeline/my_ring_buffer/`
**Plugin:** `libgstnvdsringbuf.so` (29KB)
**Language:** C++

**Purpose:** Provides 3-second fixed delay for replay functionality while keeping all data GPU-resident.

**Key Files:**
- `gstnvdsringbuf.cpp` - Main plugin
- `gstnvdsringbuf_delayed.cpp` - Delayed variant
- `Makefile` - Build system

**Configuration:**
- **Delay:** 90 frames @ 30 FPS = 3 seconds
- **Memory:** ~1.38GB for panorama frames
- **Mode:** Accumulate first N frames, then output with delay

**Use Case:**
The pipeline splits after stitching:
- **Path 1 (Live):** Panorama → Tile Batcher → YOLO → Ball Tracking
- **Path 2 (Delayed):** Panorama → Ring Buffer → Virtual Camera → Recording

This allows real-time analysis while recording has a 3-second buffer for instant replay.

**Test Scripts:**
- `test_stitch_ringbuffer_full.py` - Full integration test
- `test_ringbuf_standalone.py` - Isolated testing
- `test_performance_benchmark.py` - Performance analysis

**Documentation Status:** ✅ Good

**Key Documentation:**
- `README.md` - Feature documentation
- `RING_BUFFER_FIX_SUMMARY.md` - Implementation details
- `TEST_RESULTS.md` - Validation results

---

### 4. my_tile_batcher (Tile Extraction)

**Location:** `/home/user/ds_pipeline/my_tile_batcher/`
**Plugin:** `src/libnvtilebatcher.so` (60KB)
**Language:** C++/CUDA

**Purpose:** Extracts fixed tiles from panorama for YOLO object detection.

**Key Files:**
- `src/cuda_tile_extractor.cu` - CUDA extraction kernel
- `src/Makefile` - Build system

**Configuration:**
- **Tiles:** 6× 1024×1024 pixels
- **Layout:** Horizontally spaced with 434px vertical offset
- **Output:** GPU batch for YOLO inference

**Test Scripts:**
- `test_complete_pipeline.py` - Full pipeline test
- `test_tilebatcher.py` - Plugin testing
- `test_performance.py` - Performance analysis

**Documentation Status:** ⚠️ Empty (README.md is 1 line)

---

### 5. new_week (Main Application)

**Location:** `/home/user/ds_pipeline/new_week/`
**Language:** Python 3

**Purpose:** Main application integrating all components for complete football analysis.

**Key Files:**
- `version_masr_multiclass_RINGBUF.py` (2,781 lines) - **Production version**
- `version_masr_multiclass.py` (3,344 lines) - Multi-class detection (no ring buffer)
- `version_masr.py` (2,547 lines) - Basic ball detection
- `config_infer.txt` - YOLO inference configuration
- `labels.txt` - Class labels

**Features:**
- YOLO11n multi-class detection (ball, players, staff, referees)
- Ball trajectory tracking with Kalman filtering/interpolation
- Field mask filtering (removes out-of-bounds detections)
- Supports 3 modes: `panorama`, `virtualcam`, `streaming`
- TSV export of ball events
- Hardware H.264/H.265 encoding

**Usage:**
```bash
# File-based processing
python3 version_masr_multiclass_RINGBUF.py \
    --mode virtualcam \
    --source files \
    --left left_cam.mp4 \
    --right right_cam.mp4

# Real-time from cameras
python3 version_masr_multiclass_RINGBUF.py \
    --mode virtualcam \
    --source cameras
```

**Shell Scripts:**
- `auto_restart.sh` - Auto-restart on failure
- `rest_cameras.sh` - Reset cameras
- `test_stitch_ringbuffer.sh` - Ring buffer test

**Documentation:**
- `RINGBUFFER_INTEGRATION_REPORT.md` - Integration testing

---

### 6. soft_record_video (Camera Recording)

**Location:** `/home/user/ds_pipeline/soft_record_video/`
**Language:** Python 3

**Purpose:** Synchronized dual camera recording utilities.

**Key Files:**
- `synced_dual_record_robust.py` - **Recommended** robust synced recording
- `synced_dual_record.py` - Basic hardware-synced recording
- `fixed_exposure_recorder.py` - Fixed exposure mode
- `check_frame_sync.py` - Verify frame synchronization

**Synchronization:**
- Hardware sync via V4L2 master/slave mode
- Sensor mode 0 (no HDR for IMX678)
- Shared clock for timestamp alignment
- Achieves <1 frame sync accuracy (~33ms @ 30fps)

**Usage:**
```bash
python3 synced_dual_record_robust.py \
    --master /dev/video0 \
    --slave /dev/video1 \
    --duration 60 \
    --output-dir ./recordings/
```

**Documentation Status:** ✅ Excellent

**Key Documentation:**
- `QUICK_START.md` - Getting started
- `SYNC_README.md` - Detailed synchronization guide
- `COMPARISON.md` - Sync methods comparison
- `TROUBLESHOOTING.md` - Common issues

---

### 7. calibration (Stereo Calibration)

**Location:** `/home/user/ds_pipeline/calibration/`
**Language:** Python 3

**Purpose:** Stereo camera calibration for panorama stitching.

**Key Files:**
- `stereo_calibration.py` - Main stereo calibration
- `stereo_essential_matrix.py` - Essential matrix method (for wide FOV)
- `recalibrate_cleaned.py` - Individual camera calibration
- `analyze_stereo_calibration.py` - Quality analysis
- `calibration_result_standard.pkl` - **Output file** (1.1KB)

**Calibration Details:**
- **Camera Model:** FSM:GO-IMX678C-M12-L100A-PM-A1Q1
- **Chessboard:** 8×6 internal corners, 25mm squares
- **Quality:** RMS error 0.18-0.20 pixels (excellent)
- **Camera Angle:** 85.19° between cameras
- **FOV:** 100° per camera with ~15° overlap

**Output Format (calibration_result_standard.pkl):**
```python
{
    'left_camera_matrix': np.array,     # 3×3
    'left_dist_coeffs': np.array,       # 1×5
    'right_camera_matrix': np.array,    # 3×3
    'right_dist_coeffs': np.array,      # 1×5
    'rotation_matrix': np.array,        # 3×3
    'translation_vector': np.array,     # 3×1
    'rms_error': float
}
```

**Documentation:**
- `CALIBRATION_FILES_SUMMARY.md` - Comprehensive summary
- `ANALYSIS_SUMMARY.md` - Quality analysis

---

### 8. sliser (Panorama/Tile Saver)

**Location:** `/home/user/ds_pipeline/sliser/`
**Language:** Python 3

**Purpose:** Save panoramic frames and tiles to disk for dataset creation.

**Key Files:**
- `panorama_tiles_saver.py` - Main saver script (16KB)
- `DEPENDENCIES_INSTALL.md` - **Important:** Installation guide

**Features:**
- Saves full panoramas every N frames
- Extracts and saves 6 tiles per frame
- Timestamp-based file naming

---

### 9. ds_doc (DeepStream Documentation)

**Location:** `/home/user/ds_pipeline/ds_doc/7.1/`

**Purpose:** NVIDIA DeepStream 7.1 SDK documentation (HTML).

**Contents:**
- Python API reference
- SDK API documentation
- GraphTools documentation
- GXF (Graph Execution Framework) docs

---

## Technology Stack

### Core Frameworks

| Component | Version | Purpose |
|-----------|---------|---------|
| **DeepStream** | 7.1 | NVIDIA video analytics SDK |
| **GStreamer** | 1.14+ | Media pipeline framework |
| **CUDA** | 12.6 | GPU computing |
| **TensorRT** | Latest | YOLO inference optimization |
| **OpenCV** | 4.0+ | Image processing, calibration |

### Programming Languages

- **Python 3.6+** - Main application logic, pipeline management
- **C++14** - Custom GStreamer plugins
- **CUDA** - GPU acceleration kernels
- **Bash** - Automation scripts

### Python Libraries

```python
# Core (no requirements.txt - these are expected to be installed)
gi              # PyGObject for GStreamer bindings
pyds            # DeepStream Python bindings
numpy           # Numerical operations
cv2             # OpenCV Python bindings

# Standard library
ctypes          # C library interfacing
dataclasses     # Data structures
collections     # Data structures
threading       # Concurrency
argparse        # CLI arguments
```

### System Libraries (C/C++)

```
# GStreamer
gstreamer-1.0
gstreamer-base-1.0
gstreamer-video-1.0

# DeepStream
libnvdsgst_helper
libnvdsgst_meta
libnvds_meta
libnvbufsurface
libnvbufsurftransform

# CUDA
cudart          # CUDA runtime
```

### Hardware

- **Platform:** NVIDIA Jetson AGX Orin Developer Kit
- **Compute Capability:** SM 87 (Ampere)
- **Cameras:** Dual MIPI CSI IMX678 (4K, 100° FOV)
- **Memory:** ~2-3GB GPU memory for full pipeline

---

## Development Environment

### System Requirements

```bash
# Platform
- NVIDIA Jetson AGX Orin (recommended) or compatible Jetson device
- JetPack 6.x (includes CUDA 12.6, DeepStream 7.1)

# Software
- Ubuntu 20.04/22.04 (Jetson Linux)
- GStreamer 1.14+
- DeepStream 7.1 SDK
- CUDA 12.6
- Python 3.6+
- OpenCV 4.0+
```

### Installation Guide

**Full installation guide:** `/home/user/ds_pipeline/sliser/DEPENDENCIES_INSTALL.md`

```bash
# Core packages
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3-numpy \
    python3-opencv

# DeepStream 7.1 (follow NVIDIA installation guide)
# JetPack includes DeepStream, or install separately
```

### Building Plugins

Each plugin has its own Makefile. To rebuild:

```bash
# Example: Virtual Camera plugin
cd my_virt_cam/src
make clean
make

# Example: Panorama Stitching plugin
cd my_steach
make clean
make

# Verify plugin compilation
ls -lh lib*.so
```

**Plugin locations after build:**
- `my_steach/libnvdsstitch.so`
- `my_virt_cam/src/libnvdsvirtualcam.so`
- `my_ring_buffer/libgstnvdsringbuf.so`
- `my_tile_batcher/src/libnvtilebatcher.so`

**Important:** Plugins must be in `GST_PLUGIN_PATH` before `Gst.init()`:

```python
os.environ['GST_PLUGIN_PATH'] = (
    '/home/user/ds_pipeline/my_virt_cam/src:'
    '/home/user/ds_pipeline/my_steach:'
    '/home/user/ds_pipeline/my_ring_buffer:'
    '/home/user/ds_pipeline/my_tile_batcher/src:'
    + os.environ.get('GST_PLUGIN_PATH', '')
)
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
```

---

## Key Conventions

### Code Organization Pattern

**Plugin Directory Structure:**
```
plugin_directory/
├── src/                                    # Source code (optional subdirectory)
│   ├── gst<plugin_name>.cpp               # Main plugin implementation
│   ├── gst<plugin_name>.h                 # Plugin header
│   ├── gst<plugin_name>_allocator.cpp     # Buffer allocator (optional)
│   ├── gst<plugin_name>_allocator.h
│   ├── cuda_<plugin>_kernel.cu            # CUDA kernel
│   ├── cuda_<plugin>_kernel.h
│   ├── <plugin>_config.h                  # Configuration constants
│   ├── Makefile                           # Build system
│   └── lib<plugin>.so                     # Compiled output
├── test_*.py                              # Test scripts
├── *.md                                   # Documentation
└── README.md                              # Main documentation
```

### Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| **Plugin Name** | `nvds<name>` or `nv<name>` | `nvdsstitch`, `nvdsvirtualcam` |
| **Library File** | `lib<plugin>.so` | `libnvdsstitch.so` |
| **CUDA Kernel** | `cuda_<name>_kernel.cu` | `cuda_stitch_kernel.cu` |
| **Config Header** | `nvds<name>_config.h` | `nvdsvirtualcam_config.h` |
| **Test Script** | `test_<feature>.py` | `test_full_pipeline.py` |
| **Documentation** | `UPPERCASE_WITH_UNDERSCORES.md` | `STABILITY_FIXES.md` |

### Memory Management Pattern

**CRITICAL: All video data stays in GPU memory (NVMM)**

```cpp
// CORRECT: NVMM input/output
GstCaps *caps = gst_caps_new_simple("video/x-raw",
    "format", G_TYPE_STRING, "RGBA",
    "width", G_TYPE_INT, width,
    "height", G_TYPE_INT, height,
    "framerate", GST_TYPE_FRACTION, fps, 1,
    NULL);
gst_caps_set_features(caps, 0, gst_caps_features_new("memory:NVMM", NULL));

// Buffer allocation from NVMM pool
buffer = gst_buffer_pool_acquire_buffer(pool, &buffer, NULL);

// Map NVMM buffer for GPU access
GstMapInfo map;
gst_buffer_map(buffer, &map, GST_MAP_READWRITE);
NvBufSurface *surf = (NvBufSurface *)map.data;

// WRONG: Don't create CPU buffers
// GstBuffer *cpu_buf = gst_buffer_new_allocate(NULL, size, NULL);  // ❌
```

**Buffer Pool Pattern:**
```cpp
// Fixed pool size (8-10 buffers)
#define BUFFER_POOL_SIZE 8

// Round-robin allocation
buffer_index = (buffer_index + 1) % BUFFER_POOL_SIZE;
```

**Pitch Alignment:**
```cpp
// NVMM requires 32-byte pitch alignment
int pitch = (width * 4 + 31) & ~31;  // For RGBA
```

### GStreamer Pipeline Pattern

**Standard Python pipeline setup:**

```python
import os
import gi

# 1. Set plugin paths BEFORE Gst.init()
os.environ['GST_PLUGIN_PATH'] = '/path/to/plugins:' + os.environ.get('GST_PLUGIN_PATH', '')

# 2. Initialize GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

# 3. Parse launch string
pipeline = Gst.parse_launch("""
    source ! decoder ! nvvideoconvert !
    video/x-raw(memory:NVMM),format=RGBA !
    custom_plugin !
    encoder ! sink
""")

# 4. Add probes for metadata
def probe_callback(pad, info):
    gst_buffer = info.get_buffer()
    # Process buffer/metadata
    return Gst.PadProbeType.OK

pad = element.get_static_pad("src")
pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback)

# 5. Set up bus for messages
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

# 6. Run main loop
loop = GLib.MainLoop()
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
```

### Configuration Pattern

**All numeric constants in separate `*_config.h` files:**

```cpp
// Example: nvdsvirtualcam_config.h
#ifndef NVDSVIRTUALCAM_CONFIG_H
#define NVDSVIRTUALCAM_CONFIG_H

// Input panorama dimensions
#define PANO_WIDTH 5700
#define PANO_HEIGHT 1900

// Output virtual camera dimensions
#define OUTPUT_WIDTH 1920
#define OUTPUT_HEIGHT 1080

// Camera angle limits (degrees)
#define MIN_YAW -90.0f
#define MAX_YAW 90.0f
#define MIN_PITCH -27.0f
#define MAX_PITCH 27.0f

// CUDA configuration
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

#endif
```

### Error Handling Pattern

**CUDA error checking:**
```cpp
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    GST_ERROR("CUDA malloc failed: %s", cudaGetErrorString(err));
    return FALSE;
}

// After kernel launch
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    GST_ERROR("Kernel launch failed: %s", cudaGetErrorString(err));
    return GST_FLOW_ERROR;
}
```

**GStreamer flow returns:**
```cpp
return GST_FLOW_OK;                         // Success
return GST_FLOW_ERROR;                      // Error, stop pipeline
return GST_BASE_TRANSFORM_FLOW_DROPPED;     // Drop this frame
```

**Python error handling:**
```python
try:
    pipeline.set_state(Gst.State.PLAYING)
except Exception as e:
    print(f"Failed to start pipeline: {e}")
    pipeline.set_state(Gst.State.NULL)
    sys.exit(1)
```

### Metadata Pattern (DeepStream)

**Extracting object detections:**

```python
import pyds

def probe_callback(pad, info):
    gst_buffer = info.get_buffer()

    # Get batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # Iterate frames
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Iterate objects
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Extract detection
            class_id = obj_meta.class_id
            confidence = obj_meta.confidence
            bbox = obj_meta.rect_params
            x = bbox.left
            y = bbox.top
            w = bbox.width
            h = bbox.height

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeType.OK
```

**Creating custom metadata:**

```python
# Attach custom data to buffer
user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
user_meta.user_meta_data = custom_data
user_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_USER_META
pyds.nvds_add_user_meta_to_frame(frame_meta, user_meta)
```

### Logging and Debugging

**GStreamer debug levels:**

```bash
# Enable debug logging for specific plugin
GST_DEBUG=nvdsvirtualcam:5 python3 test_script.py

# Debug all plugins
GST_DEBUG=*:3 python3 test_script.py

# Save to file
GST_DEBUG=nvdsvirtualcam:5 python3 test_script.py 2>&1 | tee debug.log
```

**Debug levels:**
- 0: None
- 1: ERROR
- 2: WARNING
- 3: FIXME
- 4: INFO
- 5: DEBUG
- 6: LOG
- 7: TRACE

**C++ logging in plugins:**
```cpp
GST_ERROR("Critical error: %s", error_msg);
GST_WARNING("Warning: %s", warning_msg);
GST_INFO("Info: %s", info_msg);
GST_DEBUG("Debug: %s", debug_msg);
```

### Language Conventions

**Note:** This repository contains mixed Russian/English documentation and comments.

- **English** - Most documentation, variable names, function names
- **Russian** - Some documentation files (e.g., `ИНСТРУКЦИЯ_ЗАПУСКА.md`), occasional code comments

**When contributing:**
- Prefer English for new documentation
- Keep existing language in files when making small edits
- Variable/function names should always be English

---

## Common Workflows

### 1. Build Workflow

**Rebuilding a plugin after modification:**

```bash
# 1. Navigate to plugin directory
cd my_virt_cam/src

# 2. Clean previous build
make clean

# 3. Rebuild
make

# 4. Verify compilation
ls -lh libnvdsvirtualcam.so

# 5. Test with verbose logging
cd ..
GST_DEBUG=nvdsvirtualcam:5 python3 test_virtual_camera_sliders.py panorama.png
```

**Common build issues:**

```bash
# Missing pkg-config paths
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig

# CUDA arch mismatch
# Edit Makefile, ensure: -gencode arch=compute_87,code=sm_87

# Missing headers
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### 2. Test Workflow

**Running tests for a component:**

```bash
# Unit test (single plugin)
cd my_virt_cam
python3 test_virtual_camera_images.py test_panorama.png

# Performance test
python3 test_virtualcam_fps.py test_panorama.png

# Boundary test (edge cases)
python3 test_boundaries_full.py test_panorama.png

# Integration test (full pipeline)
cd ../new_week
python3 version_masr_multiclass.py --mode virtualcam --source files --left left.mp4 --right right.mp4
```

**Test script naming:**
- `test_simple.py` - Basic functionality
- `test_performance.py` - Performance/FPS benchmarks
- `test_boundaries*.py` - Edge case testing
- `test_full_pipeline.py` - End-to-end integration
- `test_*_interactive.py` - Interactive/visual tests

### 3. Calibration Workflow

**Re-calibrating cameras:**

```bash
cd calibration

# 1. Capture calibration images (if needed)
python3 stereo_capture.py
# Follow prompts, capture 20-30 image pairs with chessboard

# 2. Clean bad images (manual review)
ls -lh left/ right/

# 3. Run individual camera calibration
python3 recalibrate_cleaned.py

# 4. Run stereo calibration
python3 stereo_essential_matrix.py

# 5. Create standard PKL file
python3 create_standard_calibration.py

# 6. Analyze calibration quality
python3 analyze_stereo_calibration.py

# 7. Verify output
ls -lh calibration_result_standard.pkl
# Should be ~1.1KB
```

**Expected calibration quality:**
- RMS error: < 0.25 pixels (excellent: < 0.20)
- Camera angle: ~85° (should match physical setup)
- Per-camera FOV: ~100° (for IMX678 with L100A lens)

### 4. Camera Recording Workflow

**Recording synchronized dual camera video:**

```bash
cd soft_record_video

# 1. Verify camera devices
ls -l /dev/video*
v4l2-ctl --list-devices

# 2. Test synchronization
python3 check_frame_sync.py /dev/video0 /dev/video1

# 3. Record synchronized video
python3 synced_dual_record_robust.py \
    --master /dev/video0 \
    --slave /dev/video1 \
    --duration 120 \
    --output-dir ../recordings/ \
    --exposure 10000 \
    --gain 100

# 4. Verify frame sync
python3 check_frame_sync.py ../recordings/left.mp4 ../recordings/right.mp4
```

### 5. Full Pipeline Deployment

**Running the complete analysis pipeline:**

```bash
cd new_week

# 1. Verify all plugins are built
ls -lh ../my_virt_cam/src/libnvdsvirtualcam.so
ls -lh ../my_steach/libnvdsstitch.so
ls -lh ../my_ring_buffer/libgstnvdsringbuf.so
ls -lh ../my_tile_batcher/src/libnvtilebatcher.so

# 2. Check YOLO model
ls -lh yolo11n_model.engine  # Should exist, or will be built on first run

# 3. Run with ring buffer (recommended)
python3 version_masr_multiclass_RINGBUF.py \
    --mode virtualcam \
    --source files \
    --left ../recordings/left.mp4 \
    --right ../recordings/right.mp4

# 4. Or run from live cameras
python3 version_masr_multiclass_RINGBUF.py \
    --mode virtualcam \
    --source cameras

# 5. Output will be saved as:
# - virtualcam_output.mp4 (virtual camera following ball)
# - ball_events.tsv (ball detection log)
```

**Available modes:**
- `panorama` - Outputs full panorama (5700×1900)
- `virtualcam` - Outputs virtual camera (1920×1080) with ball tracking
- `streaming` - Outputs to RTSP stream

### 6. Debugging Pipeline Issues

**Common debugging steps:**

```bash
# 1. Check plugin registration
gst-inspect-1.0 nvdsvirtualcam
gst-inspect-1.0 nvdsstitch

# 2. Run with debug logging
GST_DEBUG=3 python3 script.py 2>&1 | tee debug.log

# 3. Check GPU memory usage
tegrastats  # On Jetson

# 4. Monitor FPS
GST_DEBUG=fpsdisplaysink:5 python3 script.py

# 5. Verify NVMM buffers
GST_DEBUG=nvbufsurface:5 python3 script.py
```

**Pipeline not starting:**
- Check `GST_PLUGIN_PATH` is set before `Gst.init()`
- Verify all `.so` files exist and have correct permissions
- Check GStreamer element compatibility with `gst-inspect-1.0`

**Low FPS:**
- Monitor with `tegrastats` for GPU/CPU bottlenecks
- Check CUDA block sizes in `*_config.h`
- Verify LUT caching is enabled
- Ensure using NVENC for encoding, not software encoder

**Crashes:**
- Check buffer pool sizes (may need to increase)
- Verify NVMM buffer mapping/unmapping is balanced
- Check for CUDA memory leaks with `cuda-memcheck`

---

## Testing Guidelines

### Unit Testing

**Each plugin should have:**

1. **Basic functionality test** (`test_simple.py`)
   - Verifies plugin loads and processes a frame
   - Checks output dimensions/format

2. **Performance test** (`test_performance.py` or `test_fps.py`)
   - Measures FPS
   - Monitors GPU memory usage
   - Logs frame drop rate

3. **Boundary test** (`test_boundaries*.py`)
   - Tests edge cases (min/max values)
   - Validates range limits
   - Checks error handling

### Integration Testing

**Full pipeline tests:**

```bash
# Test panorama stitching only
cd my_steach
python3 panorama_cameras_realtime.py

# Test panorama + virtual camera
cd my_virt_cam
python3 test_full_pipeline.py

# Test complete analysis pipeline
cd new_week
python3 version_masr_multiclass.py --mode virtualcam --source files --left test_left.mp4 --right test_right.mp4
```

### Performance Benchmarks

**Expected performance metrics:**

| Component | Expected FPS | Notes |
|-----------|-------------|-------|
| Panorama Stitching | 50+ | Dual 4K → 5700×1900 |
| Virtual Camera | 50+ | 5700×1900 → 1920×1080 |
| YOLO Detection | 30+ | 6× 1024×1024 tiles |
| **Full Pipeline** | **45-47** | All components |

**Measuring performance:**

```python
import time

start = time.time()
frame_count = 0

def probe_callback(pad, info):
    global frame_count
    frame_count += 1

    if frame_count % 100 == 0:
        elapsed = time.time() - start
        fps = frame_count / elapsed
        print(f"FPS: {fps:.2f}")

    return Gst.PadProbeType.OK
```

### Test Data

**Sample test files:**

```bash
# Panorama test images
my_virt_cam/test_panorama.png               # 5700×1900 sample panorama
my_virt_cam/pano_equirect.png               # Alternative test panorama

# Camera test videos
recordings/left.mp4                         # Left camera recording
recordings/right.mp4                        # Right camera recording

# Calibration
calibration/left/*.jpg                      # Left camera chessboard images
calibration/right/*.jpg                     # Right camera chessboard images
```

**Creating test data:**

```bash
# Record short test videos
cd soft_record_video
python3 synced_dual_record_robust.py --master /dev/video0 --slave /dev/video1 --duration 10 --output-dir test_data/

# Generate test panorama
cd my_steach
python3 panorama_cameras_realtime.py --save-frame test_panorama.png
```

---

## Important Gotchas

### 1. Plugin Path Timing

**CRITICAL:** Must set `GST_PLUGIN_PATH` **before** calling `Gst.init()`:

```python
# ✅ CORRECT
import os
os.environ['GST_PLUGIN_PATH'] = '/path/to/plugins'
from gi.repository import Gst
Gst.init(None)

# ❌ WRONG - Too late!
from gi.repository import Gst
Gst.init(None)
os.environ['GST_PLUGIN_PATH'] = '/path/to/plugins'  # Won't work
```

### 2. NVMM Memory

**All buffers must use `memory:NVMM` feature:**

```python
# ✅ CORRECT
caps_str = "video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080"

# ❌ WRONG - CPU memory, kills performance
caps_str = "video/x-raw,format=RGBA,width=1920,height=1080"
```

**Mapping NVMM buffers:**

```cpp
// ✅ CORRECT
GstMapInfo map;
if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
    return GST_FLOW_ERROR;
}
NvBufSurface *surf = (NvBufSurface *)map.data;
// ... use surf ...
gst_buffer_unmap(buffer, &map);

// ❌ WRONG - Don't access map.data directly as raw pixels
// uint8_t *pixels = map.data;  // This is NOT pixel data!
```

### 3. CUDA Architecture

**Must compile for correct SM architecture:**

```makefile
# Jetson AGX Orin = SM 87
CUDA_ARCH := -gencode arch=compute_87,code=sm_87

# ❌ WRONG architecture will cause runtime errors
# CUDA_ARCH := -gencode arch=compute_72,code=sm_72  # Wrong!
```

### 4. Metadata Iteration

**DeepStream metadata lists require careful iteration:**

```python
# ✅ CORRECT - Handle StopIteration
l_obj = frame_meta.obj_meta_list
while l_obj is not None:
    try:
        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        # ... process obj_meta ...
        l_obj = l_obj.next
    except StopIteration:
        break

# ❌ WRONG - Will crash on empty list
# for obj_meta in frame_meta.obj_meta_list:  # Don't use for loop
#     ...
```

### 5. Buffer Pool Exhaustion

**Symptoms:** Pipeline stalls, frames dropped, "no buffer available" errors

**Solution:** Increase buffer pool size

```cpp
// In plugin code
#define BUFFER_POOL_SIZE 10  // Increase from 8 to 10

// Or in Python
config = gst_structure_new_empty("config")
gst_structure_set(config, "max-buffers", G_TYPE_UINT, 10, NULL)
pool.set_config(config)
```

### 6. Camera Device Indices

**Camera devices may change after reboot:**

```bash
# Always verify device indices
ls -l /dev/video*
v4l2-ctl --list-devices

# Use consistent device names
# Master camera is usually video0, slave is video1
# But check with v4l2-ctl
```

### 7. Warp Maps Location

**Panorama stitching requires pre-computed warp maps:**

```bash
# Must exist:
warp_maps/left_warp_x.bin
warp_maps/left_warp_y.bin
warp_maps/left_weights.bin
warp_maps/right_warp_x.bin
warp_maps/right_warp_y.bin
warp_maps/right_weights.bin

# If missing, generate with calibration script
# (Script not in repo - may need to create)
```

### 8. YOLO Model Engine

**TensorRT engines are GPU-specific:**

```bash
# Engine built on Jetson AGX Orin won't work on different GPU
# Always rebuild engine on target device:
rm yolo11n_model.engine
python3 version_masr_multiclass.py  # Will rebuild engine
```

### 9. Frame Synchronization

**For synchronized recording:**

- Must set master/slave mode **before** starting capture
- Both cameras must use same sensor mode (mode 0 for IMX678)
- Disable HDR (not supported in sync mode)
- Use shared system clock, not per-camera timestamps

```python
# Set V4L2 controls before creating pipeline
os.system(f"v4l2-ctl -d {master_dev} -c sensor_mode=0")
os.system(f"v4l2-ctl -d {slave_dev} -c sensor_mode=0")
```

### 10. Pitch Alignment

**NVMM buffers require 32-byte pitch alignment:**

```cpp
// ❌ WRONG
int pitch = width * 4;  // For RGBA

// ✅ CORRECT
int pitch = (width * 4 + 31) & ~31;
```

**Symptoms of incorrect pitch:**
- Corrupted/shifted video
- Crashes in CUDA kernels
- "Invalid pitch" errors in logs

---

## File Structure Reference

### Directory Tree

```
ds_pipeline/
├── .git/                           # Git repository
├── .gitignore
├── CLAUDE.md                       # This file
├── nvidia_jetson_orin_nx_16GB_super_arch.pdf  # Hardware documentation
│
├── my_steach/                      # Panorama Stitching Plugin
│   ├── src/
│   │   ├── cuda_stitch_kernel.cu
│   │   ├── cuda_stitch_kernel.h
│   │   └── nvdsstitch_config.h
│   ├── libnvdsstitch.so           # Compiled plugin
│   ├── Makefile
│   ├── panorama_cameras_realtime.py
│   ├── panorama_stream.py
│   ├── test_fps.py
│   ├── README.md                  # ⚠️ Empty
│   └── ARCHITECTURE.md            # ⚠️ Empty
│
├── my_virt_cam/                   # Virtual Camera Plugin
│   ├── src/
│   │   ├── gstnvdsvirtualcam.cpp  # Main plugin (88KB)
│   │   ├── gstnvdsvirtualcam.h
│   │   ├── gstnvdsvirtualcam_allocator.cpp
│   │   ├── gstnvdsvirtualcam_allocator.h
│   │   ├── cuda_virtual_cam_kernel.cu
│   │   ├── cuda_virtual_cam_kernel.h
│   │   ├── nvdsvirtualcam_config.h
│   │   ├── Makefile
│   │   └── libnvdsvirtualcam.so  # Compiled plugin (91KB)
│   ├── test_*.py                 # 15+ test scripts
│   ├── README.md                 # ✅ Comprehensive (364 lines)
│   ├── STABILITY_FIXES.md        # ✅ Production fixes
│   ├── FOV_TUNING_GUIDE.md
│   ├── OPTIMIZATION_RESULTS.md
│   └── 25+ other documentation files
│
├── my_ring_buffer/               # Ring Buffer Plugin
│   ├── gstnvdsringbuf.cpp
│   ├── gstnvdsringbuf_delayed.cpp
│   ├── libgstnvdsringbuf.so     # Compiled plugin (29KB)
│   ├── Makefile
│   ├── test_*.py                # 5+ test scripts
│   ├── README.md                # ✅ Good documentation
│   ├── RING_BUFFER_FIX_SUMMARY.md
│   └── ИНСТРУКЦИЯ_ЗАПУСКА.md    # Russian instructions
│
├── my_tile_batcher/             # Tile Extraction Plugin
│   ├── src/
│   │   ├── cuda_tile_extractor.cu
│   │   ├── cuda_tile_extractor.h
│   │   ├── Makefile
│   │   └── libnvtilebatcher.so # Compiled plugin (60KB)
│   ├── test_*.py               # 4+ test scripts
│   ├── README.md               # ⚠️ Empty
│   └── STABILITY_*.md          # ⚠️ Empty
│
├── new_week/                    # Main Application
│   ├── version_masr_multiclass_RINGBUF.py  # ✅ Production (2,781 lines)
│   ├── version_masr_multiclass.py          # Multi-class (3,344 lines)
│   ├── version_masr.py                     # Basic ball detection (2,547 lines)
│   ├── config_infer.txt                    # YOLO config
│   ├── labels.txt                          # Class labels
│   ├── auto_restart.sh                     # Auto-restart script
│   ├── rest_cameras.sh                     # Camera reset utility
│   ├── test_stitch_ringbuffer.sh
│   └── RINGBUFFER_INTEGRATION_REPORT.md
│
├── soft_record_video/           # Camera Recording Utilities
│   ├── synced_dual_record_robust.py  # ✅ Recommended
│   ├── synced_dual_record.py
│   ├── fixed_exposure_recorder.py
│   ├── check_frame_sync.py
│   ├── dual_record.py
│   ├── QUICK_START.md          # ✅ Getting started
│   ├── SYNC_README.md          # ✅ Detailed sync guide
│   ├── COMPARISON.md
│   └── TROUBLESHOOTING.md
│
├── calibration/                 # Stereo Calibration
│   ├── stereo_calibration.py
│   ├── stereo_essential_matrix.py
│   ├── recalibrate_cleaned.py
│   ├── analyze_stereo_calibration.py
│   ├── create_standard_calibration.py
│   ├── calibration_result_standard.pkl  # ✅ Output (1.1KB)
│   ├── CALIBRATION_FILES_SUMMARY.md  # ✅ Excellent docs
│   └── ANALYSIS_SUMMARY.md
│
├── sliser/                      # Panorama/Tile Saver
│   ├── panorama_tiles_saver.py
│   ├── test_gstreamer_import.py
│   └── DEPENDENCIES_INSTALL.md # ✅ Installation guide
│
└── ds_doc/                      # DeepStream Documentation
    └── 7.1/                     # DeepStream 7.1 HTML docs
        ├── index.html
        ├── python/
        ├── sdk/
        └── graphtools/
```

### Key Files Quick Reference

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `my_virt_cam/src/libnvdsvirtualcam.so` | Virtual camera plugin | 91KB | ✅ Production |
| `my_steach/libnvdsstitch.so` | Panorama stitching | 97KB | ✅ Production |
| `my_ring_buffer/libgstnvdsringbuf.so` | Ring buffer | 29KB | ✅ Production |
| `my_tile_batcher/src/libnvtilebatcher.so` | Tile extraction | 60KB | ✅ Production |
| `new_week/version_masr_multiclass_RINGBUF.py` | Main application | 2,781 lines | ✅ Production |
| `calibration/calibration_result_standard.pkl` | Calibration data | 1.1KB | ✅ Current |
| `sliser/DEPENDENCIES_INSTALL.md` | Installation guide | - | ✅ Important |
| `my_virt_cam/README.md` | Virtual cam docs | 364 lines | ✅ Excellent |

---

## Quick Start for AI Assistants

### When asked to modify the pipeline:

1. **Identify the component:**
   - Panorama dimensions? → `my_steach`
   - Virtual camera angles/FOV? → `my_virt_cam`
   - Delay buffer? → `my_ring_buffer`
   - Detection tiles? → `my_tile_batcher`
   - Application logic? → `new_week`

2. **Read relevant documentation:**
   - `my_virt_cam/README.md` - Best documented
   - Check `*_config.h` for current settings

3. **Make changes:**
   - Modify `*_config.h` for constants
   - Modify `.cpp`/`.cu` for logic changes
   - Rebuild with `make clean && make`

4. **Test:**
   - Use existing `test_*.py` scripts
   - Check FPS and memory with `tegrastats`

### When asked about the system:

1. **Check documentation first:**
   - `my_virt_cam/` has 30+ docs - very comprehensive
   - `calibration/CALIBRATION_FILES_SUMMARY.md` - calibration details
   - `soft_record_video/SYNC_README.md` - camera sync

2. **Common questions:**
   - "How does ball tracking work?" → `new_week/version_masr_multiclass_RINGBUF.py`
   - "Camera angle limits?" → `my_virt_cam/src/nvdsvirtualcam_config.h`
   - "Pipeline architecture?" → This file (Architecture section)

### When asked to add features:

1. **Determine location:**
   - New detection class? → `new_week/config_infer.txt` + `labels.txt`
   - New camera mode? → `my_virt_cam/src/nvdsvirtualcam_config.h`
   - New recording mode? → `soft_record_video/`

2. **Follow existing patterns:**
   - Use NVMM for all buffers
   - Add tests (`test_*.py`)
   - Update documentation

3. **Validate:**
   - Performance targets (45+ FPS)
   - GPU memory usage
   - Code quality (error handling, logging)

---

## Additional Resources

### External Documentation

- **DeepStream 7.1:** `/home/user/ds_pipeline/ds_doc/7.1/index.html`
- **GStreamer:** https://gstreamer.freedesktop.org/documentation/
- **CUDA:** https://docs.nvidia.com/cuda/
- **Jetson:** https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit

### Support and Issues

- Check existing documentation in component directories
- Review `TROUBLESHOOTING.md` in `soft_record_video/`
- Check GStreamer debug logs with `GST_DEBUG=3`
- Monitor GPU with `tegrastats`

---

## Maintenance Notes

### Regular Maintenance

- **Re-calibrate cameras:** Every 3-6 months or after camera movement
- **YOLO model:** Update when new models available
- **Dependencies:** Keep DeepStream/CUDA updated with JetPack
- **Test recordings:** Verify frame sync regularly

### Before Production Deployment

- [ ] All plugins compiled for target GPU architecture
- [ ] Calibration verified (RMS < 0.25 pixels)
- [ ] Frame sync verified (<1 frame difference)
- [ ] YOLO model engine built on target device
- [ ] Warp maps generated from current calibration
- [ ] Full pipeline tested at target FPS (45+)
- [ ] Auto-restart script configured (`auto_restart.sh`)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-16 | Initial CLAUDE.md creation |

---

**End of CLAUDE.md**
