# DeepStream Sports Analytics Pipeline - System Architecture

## Overview

Real-time AI sports analytics system running on NVIDIA Jetson Orin NX 16GB, processing dual 4K camera feeds for panoramic video generation, object detection, and intelligent virtual camera control.

**Core Technologies**: DeepStream SDK 7.1 | CUDA 12.6 | TensorRT 10.5 | GStreamer 1.0

---

## Hardware Architecture

### NVIDIA Jetson Orin NX 16GB Platform

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVIDIA Jetson Orin NX 16GB                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐           ┌──────────────────────────┐  │
│  │  CPU Complex       │           │  GPU Complex             │  │
│  │  8× Cortex-A78AE   │◄─────────►│  1024 CUDA cores         │  │
│  │  @ 2.0 GHz         │           │  32 Tensor cores         │  │
│  └────────────────────┘           │  @ 918 MHz               │  │
│           │                       └──────────────────────────┘  │
│           │                                  │                  │
│           └─────────┬────────────────────────┘                  │
│                     │                                           │
│              ┌──────▼──────────┐                                │
│              │ Unified Memory  │                                │
│              │ 16 GB LPDDR5    │                                │
│              │ @ 102 GB/s      │                                │
│              └─────────────────┘                                │
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  Video Decode    │      │  Video Encode    │                 │
│  │  2× 4K60         │      │  1× 4K60         │                 │
│  │  HEVC/H.264/AV1  │      │  HEVC/H.264      │                 │
│  └──────────────────┘      └──────────────────┘                 │
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  NVDLA Engines   │      │  ISP             │                 │
│  │  2× ~20 TOPS     │      │  MIPI CSI-2      │                 │
│  └──────────────────┘      └──────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
         ┌──────────▼─────────┐   ┌────▼──────────────┐
         │  Camera 0 (Left)   │   │ Camera 1 (Right)  │
         │  Sony IMX678       │   │ Sony IMX678       │
         │  3840×2160 @ 30fps │   │ 3840×2160 @ 30fps │
         │  100° FOV          │   │ 100° FOV          │
         │  MIPI CSI-2        │   │ MIPI CSI-2        │
         └────────────────────┘   └───────────────────┘
              85° separation, 15° overlap
```

**Key Characteristics**:
- **Unified Memory**: CPU and GPU share 16GB physical RAM (no VRAM separation)
- **Memory Bandwidth**: 102 GB/s shared between CPU/GPU (critical constraint)
- **AI Performance**: ~100 TOPS (INT8), optimized for TensorRT inference
- **I/O Coherency**: Compute capability ≥7.2, zero-copy NVMM buffers

---

## Software Architecture

### Layered System Design

```
┌───────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                              │
│                 version_masr_multiclass.py                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ PanoramaWithVirtualCamera (Orchestrator)                    │  │
│  │ • Configuration management                                  │  │
│  │ • Component composition                                     │  │
│  │ • Pipeline lifecycle control                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
┌───────────▼──────┐  ┌───────▼───────┐  ┌─────▼──────────┐
│  BUSINESS LOGIC  │  │  PROCESSING   │  │   RENDERING    │
│      LAYER       │  │     LAYER     │  │     LAYER      │
├──────────────────┤  ├───────────────┤  ├────────────────┤
│ • core/          │  │ • processing/ │  │ • rendering/   │
│   History Mgmt   │  │   YOLO        │  │   Display      │
│   Trajectories   │  │   Analysis    │  │   VirtualCam   │
│   Players        │  │   Tensor      │  │                │
│                  │  │               │  │                │
│ • utils/         │  │               │  │                │
│   Field Mask     │  │               │  │                │
│   CSV Logger     │  │               │  │                │
│   NMS            │  │               │  │                │
└──────────────────┘  └───────────────┘  └────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
┌───────────────────────────────▼───────────────────────────────────┐
│                    PIPELINE LAYER                                 │
│                     pipeline/                                     │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │ Pipeline     │  │ Playback     │  │ Buffer                 │   │
│  │ Builder      │  │ Builder      │  │ Manager                │   │
│  │ (Analysis)   │  │ (Display)    │  │ (7s sync)              │   │
│  └──────────────┘  └──────────────┘  └────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                              │
┌───────────────────────────────▼───────────────────────────────────┐
│                  GSTREAMER FRAMEWORK                              │
│                    (Media Pipeline)                               │
│                                                                   │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐  ┌──────────────┐      │
│  │nvstreammux│→│nvdsstitch  │→│nvtile    │→│nvinfer        │      │
│  │          │  │(my_steach) │  │batcher  │  │(TensorRT)    │      │
│  └──────────┘  └────────────┘  └─────────┘  └──────────────┘      │
└───────────────────────────────────────────────────────────────────┘
                              │
┌───────────────────────────────▼───────────────────────────────────┐
│                    CUDA/HARDWARE LAYER                            │
│                                                                   │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ CUDA       │  │ TensorRT    │  │ NVMM        │                 │
│  │ Kernels    │  │ Inference   │  │ Zero-copy   │                 │
│  │ (stitching,│  │ Engine      │  │ Buffers     │                 │
│  │  tiling,   │  │             │  │             │                 │
│  │  virtcam)  │  │             │  │             │                 │
│  └────────────┘  └─────────────┘  └─────────────┘                 │
└───────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Dual-Pipeline Design

```
┌────────────────────────────────────────────────────────────────────┐
│                        CAMERA INPUTS                               │
│         Camera 0 (3840×2160)          Camera 1 (3840×2160)         │
└───────────────┬──────────────────────────────┬─────────────────────┘
                │                              │
                ▼                              ▼
         [nvarguscamerasrc]            [nvarguscamerasrc]
                │                              │
                ▼                              ▼
         [nvvideoconvert]              [nvvideoconvert]
          RGBA (NVMM)                   RGBA (NVMM)
                │                              │
                └──────────┬───────────────────┘
                           ▼
                   [nvstreammux]
                   batch-size=2
                   GPU memory only
                           │
                           ▼
               ┌────────────────────────┐
               │   MY_STEACH PLUGIN     │
               │  Panorama Stitching    │
               │   • LUT-based warping  │
               │   • Color correction   │
               │   • 5700×1900 RGBA     │
               └────────────┬───────────┘
                           │
                       [queue]
                    GPU (NVMM)
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│  ANALYSIS BRANCH      │     │  DISPLAY BRANCH       │
│  (Real-time)          │     │  (7s buffered)        │
└───────────────────────┘     └───────────────────────┘
            │                             │
            ▼                             ▼
  [MY_TILE_BATCHER]                 [appsrc]
  6×1024×1024 tiles               Buffered playback
            │                             │
            ▼                             │
     [nvinfer]                            │
   YOLOv11n/s FP16                        │
   TensorRT engine                        │
            │                             │
            ▼                             │
  [Tensor Processing]                     │
  Post-NMS multiclass:                    │
  • ball (class 0)                        │
  • player (class 1)                      │
  • staff (class 2)                       │
  • side_referee (class 3)                │
  • main_referee (class 4)                │
            │                             │
            ▼                             │
  [BallDetectionHistory]                  │
  + PlayersHistory                        │
  Raw → Processed → Confirmed             │
  (10s history, interpolation)            │
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  MY_VIRT_CAM PLUGIN     │
          │  CUDA perspective       │
          │  projection             │
          │                         │
          │  Input: 5700×1900       │
          │  Output: 1920×1080      │
          │                         │
          │  Auto tracking:         │
          │  • Ball (primary)       │
          │  • Players (fallback)   │
          │  • Speed-based zoom     │
          │  • Smooth pursuit       │
          └────────────┬────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
        [nvdsosd]             [nvdsosd]
      Panorama view         Virtual camera
      (16 obj max)          + overlays
            │                     │
            ▼                     ▼
     Output options:       Output options:
     • Display             • Display
     • RTMP stream         • RTMP stream
     • Video file          • Video file
```

### Memory Flow (Zero-Copy Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVMM BUFFER FLOW                             │
│                  (GPU-resident throughout)                      │
└─────────────────────────────────────────────────────────────────┘

Camera Capture
     │
     ├─→ NVMM Buffer Pool (nvarguscamerasrc)
     │   Allocation: 2× 3840×2160×4 bytes = ~66 MB
     │
     ▼
Stitching (my_steach)
     │
     ├─→ NVMM Buffer Pool (8 buffers)
     │   Allocation: 8× 5700×1900×4 bytes = ~347 MB
     │
     ▼
Tile Batching (my_tile_batcher)
     │
     ├─→ NVMM Buffer Pool (4 buffers, 6 surfaces each)
     │   Allocation: 4× 6× 1024×1024×4 bytes = ~100 MB
     │
     ▼
Inference (nvinfer)
     │
     ├─→ TensorRT Managed Memory
     │   Workspace: ~2 GB (model + activations)
     │
     ▼
Virtual Camera (my_virt_cam)
     │
     ├─→ EGL-mapped CUDA Memory
     │   Allocation: 8× 1920×1080×4 bytes = ~64 MB
     │   + LUT Cache: ~42 MB (ray + coordinate maps)
     │
     ▼
Display (nveglglessink)
     │
     └─→ Direct NVMM Consumption (no copy)

┌─────────────────────────────────────────────────────────────────┐
│  CPU COPIES ONLY FOR:                                           │
│  • Analysis results (metadata, ~few KB)                         │
│  • RAM buffer (encoded H.264, ~1MB per frame)                   │
│  • CSV logging (text, <1 KB per detection)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Modular System Design (Post-Refactoring)

```
┌────────────────────────────────────────────────────────────────┐
│         PanoramaWithVirtualCamera (Orchestrator)               │
│                      ~400 lines                                │
└────┬───────┬────────┬─────────┬──────────┬─────────────────────┘
     │       │        │         │          │
     │creates│creates │creates  │creates   │creates
     │       │        │         │          │
     ▼       ▼        ▼         ▼          ▼
┌─────┐ ┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐
│Field│ │History │ │Players │ │Tensor│ │Buffer  │
│Mask │ │Manager │ │History │ │Proc  │ │Manager │
└──┬──┘ └───┬────┘ └───┬────┘ └──┬───┘ └───┬────┘
   │        │           │         │         │
   │ used by│used by    │used by  │used by  │manages
   ▼        ▼           ▼         ▼         ▼
┌──────────────────────────────────────────────────┐
│         Probe Handlers (created later)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Analysis  │  │VirtualCam│  │Display   │        │
│  │Probe     │  │Probe     │  │Probe     │        │
│  └──────────┘  └──────────┘  └──────────┘        │
└──────────────────────────────────────────────────┘
         │              │              │
         │ callbacks    │ callbacks    │ callbacks
         ▼              ▼              ▼
┌────────────────────────────────────────────────────┐
│            GStreamer Pipelines                     │
│  ┌─────────────┐         ┌─────────────┐           │
│  │ Analysis    │─────────│ Playback    │           │
│  │ Pipeline    │ buffers │ Pipeline    │           │
│  └─────────────┘         └─────────────┘           │
└────────────────────────────────────────────────────┘
```

### Module Responsibilities

#### **utils/** - General Utilities
```
FieldMaskBinary
├─ Load binary field mask (1900×5700 bitmap)
├─ is_inside_field(x, y) → bool
└─ Fast pixel lookup (~O(1))

CSV Logger
├─ save_detection_to_csv(timestamp, x, y, confidence)
└─ TSV format for ball events

NMS
├─ apply_nms(detections, iou_threshold)
└─ Remove overlapping bboxes
```

#### **core/** - Detection & History
```
HistoryManager (replaces BallDetectionHistory)
├─ DetectionStorage (3-tier)
│  ├─ Raw future history (incoming)
│  ├─ Processed future history (cleaned)
│  └─ Confirmed history (displayed)
├─ TrajectoryFilter
│  ├─ Outlier detection
│  └─ Blacklist management
└─ TrajectoryInterpolator
   ├─ Parabolic interpolation (flight)
   └─ Linear interpolation (ground)

PlayersHistory
├─ Store player positions per timestamp
├─ Calculate center-of-mass
└─ EMA smoothing (α = 0.18)
```

#### **processing/** - YOLO Analysis
```
TensorProcessor
├─ postprocess_yolo_output()
│  ├─ Parse tensor (21504×9)
│  ├─ Extract bbox + class scores
│  └─ Filter by confidence
└─ Coordinate transform (tile → panorama)

AnalysisProbeHandler
├─ analysis_probe(pad, info, u_data)
├─ Process 6 tiles per batch
├─ Apply field mask filter
├─ Store in history
└─ Update all_detections_history
```

#### **rendering/** - Display & Camera
```
VirtualCameraProbeHandler
├─ vcam_update_probe(pad, info, u_data)
├─ Ball tracking
│  ├─ Speed-based FOV (300-1200 px/s)
│  ├─ Smooth pursuit (factor 0.3)
│  └─ Loss recovery (expand 2°/s)
└─ Player fallback (>3s ball loss)

DisplayProbeHandler
├─ playback_draw_probe(pad, info, u_data)
├─ Multi-class bbox rendering
│  ├─ Ball: red, border=3
│  ├─ Players: green, border=2
│  └─ Staff/Refs: other colors
├─ Future trajectory (dotted line)
└─ Limit to 16 objects (Jetson constraint)
```

#### **pipeline/** - Pipeline Building
```
ConfigBuilder
└─ create_inference_config() → config_infer.txt

PipelineBuilder
├─ build() → Analysis pipeline
├─ Source config (files vs cameras)
├─ nvstreammux setup
├─ my_steach (stitching)
├─ my_tile_batcher (tiling)
├─ nvinfer (YOLO)
└─ appsink (buffering)

PlaybackPipelineBuilder
├─ build() → Playback pipeline
├─ appsrc (from buffer)
├─ my_virt_cam (perspective)
├─ nvdsosd (overlay)
└─ Output modes:
   ├─ panorama (full view)
   ├─ virtualcam (tracked)
   ├─ stream (RTMP)
   └─ record (MP4)

BufferManager
├─ Frame buffer (deque, 210 frames @ 30fps)
├─ Audio buffer (deque, synchronized)
├─ _buffer_loop() (background thread)
├─ Timestamp-based retrieval
└─ Audio/video sync
```

---

## Performance Architecture

### Latency Budget (Target: <100ms)

```
Component              Target    Measured   Headroom
───────────────────────────────────────────────────────
Camera Capture         ~33ms     ~33ms      ✓ On target
Stitching (my_steach)  ≤10ms     ~10ms      ✓ On target
Tile Batcher           ≤1ms      ~1ms       ✓ On target
Inference (YOLO)       ≤20ms     ~20ms      ✓ On target
Virtual Camera         ≤22ms     ~21ms      ✓ On target
Display                ~10ms     ~10ms      ✓ On target
───────────────────────────────────────────────────────
TOTAL END-TO-END       ≤100ms    ~95ms      ✓ 5ms margin
```

### Memory Budget (16 GB total)

```
Component                    Allocation    Notes
────────────────────────────────────────────────────────
System/OS                    ~2 GB         Ubuntu base
DeepStream SDK               ~1 GB         Framework
Camera Buffers (NVMM)        ~0.13 GB      2× 66 MB
Stitching Buffers (NVMM)     ~0.35 GB      8× 43 MB
Tile Buffers (NVMM)          ~0.10 GB      4× 25 MB
TensorRT Engine/Workspace    ~2 GB         Model + activations
Virtual Camera Buffers       ~0.06 GB      8× 8 MB
LUT Caches (steach + vcam)   ~0.08 GB      Coordinate maps
Frame Buffer (7s @ 30fps)    ~3 GB         210× 15 MB (H.264)
Audio Buffer                 ~0.05 GB      7s @ 48kHz
────────────────────────────────────────────────────────
SUBTOTAL                     ~8.8 GB
Available Headroom           ~7.2 GB       For OS/apps
────────────────────────────────────────────────────────
Peak Usage (measured)        ~10 GB        Leaves 6 GB free
```

### GPU Utilization Target

```
Component              GPU Load    Notes
──────────────────────────────────────────────────────
Stitching (CUDA)       ~15%        LUT-based, coalesced
Tile Batching (CUDA)   ~5%         Simple copy
Inference (TensorRT)   ~40%        Batch of 6 tiles
Virtual Camera (CUDA)  ~10%        Perspective projection
──────────────────────────────────────────────────────
TOTAL                  ~70%        Safe margin for 30 FPS
──────────────────────────────────────────────────────
Target: <80% for stability
Measured: ~70% average, ~85% peak
```

---

## CUDA Kernel Architecture

### my_steach: Panorama Stitching Kernel

```c
__global__ void panorama_lut_kernel(
    cudaTextureObject_t left_tex,
    cudaTextureObject_t right_tex,
    float* lut_left_x,  float* lut_left_y,
    float* lut_right_x, float* lut_right_y,
    float* weight_left, float* weight_right,
    uint8_t* output,    // RGBA output
    int out_width, int out_height
) {
    // Grid: 179×238 blocks, Block: 32×8 threads
    // Total: ~1.4M threads for 5700×1900 pixels

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    // LUT lookup (pre-computed warping)
    int idx = y * out_width + x;
    float src_x_l = lut_left_x[idx];
    float src_y_l = lut_left_y[idx];
    float src_x_r = lut_right_x[idx];
    float src_y_r = lut_right_y[idx];

    // Bilinear interpolation from textures
    uchar4 pixel_left = tex2D<uchar4>(left_tex, src_x_l, src_y_l);
    uchar4 pixel_right = tex2D<uchar4>(right_tex, src_x_r, src_y_r);

    // Weighted blending (overlap zone)
    float w_l = weight_left[idx];
    float w_r = weight_right[idx];

    output[idx*4 + 0] = w_l * pixel_left.x + w_r * pixel_right.x;  // R
    output[idx*4 + 1] = w_l * pixel_left.y + w_r * pixel_right.y;  // G
    output[idx*4 + 2] = w_l * pixel_left.z + w_r * pixel_right.z;  // B
    output[idx*4 + 3] = 255;                                       // A
}
```

**Performance**: ~10ms for 5700×1900 @ 30 FPS on Jetson Orin NX

### my_tile_batcher: Tile Extraction Kernel

```c
__global__ void extract_tiles_kernel_multi(
    uint8_t* src_panorama,  // 5700×1900 RGBA
    uint8_t* dst_tiles,     // 6×1024×1024 RGBA
    int panorama_width,
    __constant__ TilePosition positions[6]
) {
    // Grid: (32, 32, 6) blocks, Block: (32, 32, 1) threads
    // Process all 6 tiles in parallel

    int tile_id = blockIdx.z;  // 0-5
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1024 || y >= 1024) return;

    // Source coordinates
    int src_x = positions[tile_id].x + x;
    int src_y = positions[tile_id].y + y;
    int src_idx = (src_y * panorama_width + src_x) * 4;

    // Destination coordinates
    int dst_idx = (tile_id * 1024 * 1024 + y * 1024 + x) * 4;

    // Coalesced read/write (4 bytes RGBA)
    *((uint32_t*)&dst_tiles[dst_idx]) = *((uint32_t*)&src_panorama[src_idx]);
}
```

**Performance**: ~1ms for 6×1024×1024 extraction @ 30 FPS

### my_virt_cam: Virtual Camera Kernel

```c
__global__ void virtual_camera_kernel(
    cudaTextureObject_t panorama_tex,  // 5700×1900 equirectangular
    uint8_t* output,                   // 1920×1080 perspective
    float* lut_x, float* lut_y,        // Pre-computed sampling coords
    int out_width, int out_height
) {
    // Grid: 8,160 blocks, Block: 16×16 threads

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    // LUT lookup (3D rotation + spherical mapping)
    int idx = y * out_width + x;
    float src_x = lut_x[idx];
    float src_y = lut_y[idx];

    // Sample from panorama (bilinear via texture unit)
    uchar4 pixel = tex2D<uchar4>(panorama_tex, src_x, src_y);

    // Write to output
    int out_idx = idx * 4;
    output[out_idx + 0] = pixel.x;  // R
    output[out_idx + 1] = pixel.y;  // G
    output[out_idx + 2] = pixel.z;  // B
    output[out_idx + 3] = 255;      // A
}
```

**Performance**: ~21ms for 1920×1080 @ 47 FPS (can run faster than 30 FPS pipeline)

---

## DeepStream Metadata Architecture

### Metadata Hierarchy

```
NvDsBatchMeta (created by nvstreammux)
├─ base_meta
│  ├─ batch_id
│  ├─ num_frames_in_batch
│  └─ meta_mutex (for thread safety)
│
├─ frame_meta_list → NvDsFrameMeta (per frame in batch)
│  ├─ base_meta
│  │  ├─ frame_num
│  │  ├─ source_id
│  │  └─ ntp_timestamp
│  │
│  ├─ obj_meta_list → NvDsObjectMeta (per detected object)
│  │  ├─ class_id (0=ball, 1=player, etc.)
│  │  ├─ confidence
│  │  ├─ rect_params (bbox: x, y, w, h)
│  │  └─ obj_user_meta_list
│  │
│  ├─ display_meta_list → NvDsDisplayMeta (for OSD)
│  │  ├─ num_rects (max 16 on Jetson)
│  │  ├─ num_labels
│  │  ├─ num_lines
│  │  └─ rect_params[] (array of rectangles)
│  │
│  └─ frame_user_meta_list → NvDsUserMeta
│     ├─ base_meta.meta_type (NVDSINFER_TENSOR_OUTPUT_META)
│     └─ user_meta_data → NvDsInferTensorMeta
│        ├─ output_layer_info
│        ├─ num_output_layers
│        └─ out_buf_ptrs_host[] (YOLO tensors)
│
└─ batch_user_meta_list → NvDsUserMeta (batch-level custom data)
```

### Metadata Flow in Probes

```
1. analysis_probe (after nvinfer)
   ├─ Extract tensor from frame_user_meta_list
   ├─ Parse YOLO output (TensorProcessor)
   ├─ Filter by field mask (FieldMaskBinary)
   ├─ Store in history (HistoryManager, PlayersHistory)
   └─ Update all_detections_history (for display)

2. vcam_update_probe (before rendering)
   ├─ Query history for ball position
   ├─ Calculate target yaw, pitch, FOV
   ├─ Apply smooth pursuit
   ├─ Set virtual camera properties
   └─ Fallback to players if ball lost

3. playback_draw_probe (on playback pipeline)
   ├─ Acquire display_meta from pool
   ├─ Query all_detections_history
   ├─ Add rectangles (ball: red, players: green)
   ├─ Add lines (future trajectory)
   ├─ Limit to 16 objects (Jetson constraint)
   └─ Release metadata to pipeline
```

---

## Security & Constraints

### Jetson Platform Constraints

1. **Memory Bandwidth** (102 GB/s shared)
   - **Mitigation**: NVMM zero-copy, no CPU transfers
   - **Monitoring**: Track bandwidth usage in nvidia-smi

2. **GPU Power Budget** (15W TDP mode)
   - **Mitigation**: Keep GPU load <80%
   - **Monitoring**: nvidia-smi power draw

3. **Thermal Throttling** (85°C max)
   - **Mitigation**: Passive cooling, ambient <30°C
   - **Monitoring**: nvidia-smi temperature

4. **nvdsosd Limit** (16 objects max on Jetson)
   - **Mitigation**: Priority rendering (ball > players)
   - **Workaround**: Custom CUDA overlay if needed

5. **NVENC Slots** (1 hardware encoder)
   - **Mitigation**: Use software encoder for secondary streams
   - **Note**: CPU overhead for multi-stream

---

## Deployment Architecture

### Production Configuration

```
┌─────────────────────────────────────────────────────────┐
│                   Jetson Orin NX                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  DeepStream Pipeline                              │  │
│  │  • Auto-restart on failure                        │  │
│  │  • Watchdog monitoring                            │  │
│  │  • Log rotation                                   │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                      │
│                  ├─→ Local Display (HDMI)               │
│                  ├─→ RTMP Stream (network)              │
│                  └─→ Video Recording (SSD)              │
└─────────────────────────────────────────────────────────┘
                      │
                      ├─→ Remote Monitoring (SSH/VNC)
                      ├─→ Metrics Export (Prometheus)
                      └─→ Log Aggregation (rsyslog)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Target Platform**: NVIDIA Jetson Orin NX 16GB
**DeepStream Version**: 7.1
