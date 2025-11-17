# DeepStream Sports Analytics Pipeline - Architectural Decisions

This document records all significant architectural and design decisions made during the development of the DeepStream Sports Analytics Pipeline.

**Format**: Each decision includes context, alternatives considered, decision made, rationale, consequences, and status.

---

## ADR-001: NVIDIA Jetson Orin NX Platform Selection

**Date**: 2025-09-01
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need embedded platform for real-time dual 4K camera processing with AI inference.

### Alternatives Considered
1. **NVIDIA Jetson Orin NX SUPER 16GB** (selected)
2. NVIDIA Jetson AGX Orin 32GB (more powerful but larger/expensive)
3. NVIDIA Jetson Xavier NX (older generation, less AI performance)
4. x86 workstation with dGPU (not embedded, higher power)

### Decision
Use **NVIDIA Jetson Orin NX SUPER 16GB**

### Rationale
- **AI Performance**: 100 TOPS (INT8) sufficient for YOLOv11 on 6 tiles (but we are usin FP16 model engine)
- **Memory**: 16GB unified RAM adequate for pipeline + 7s buffer
- **Form Factor**: Compact embedded design
- **Power**: 40W TDP (maximum for NVIDIA Jetson Orin NX SUper 16GB)
- **Video**: 2× 4K60 decode, 1× 4K60 encode (matches dual camera requirement)
- **Cost**: Balanced price/performance (~$699)

### Consequences
**Positive**:
- Meets real-time 30 FPS target with <100ms latency
- Unified memory simplifies zero-copy architecture
- Good community support and documentation

**Negative**:
- Memory bandwidth limited to 102 GB/s (requires careful optimization)
- nvdsosd limited to 16 objects (platform constraint)
- Single NVENC limits multi-stream encoding

### Related
- ADR-003 (NVMM zero-copy architecture)
- ADR-008 (Memory budget constraints)

---

## ADR-002: Stereo Calibration Method Selection

**Date**: 2025-09-15
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need stereo calibration for dual cameras with wide-angle (85° separation) for panorama stitching.

### Alternatives Considered
1. **Essential Matrix Method** (selected)
2. Standard cv2.stereoCalibrate() (limited to narrow baselines)
3. Homography-only method (no geometric constraints)

### Decision
Use **Essential Matrix Method** with cv2.findEssentialMat()

### Rationale
- **Wide-Angle Support**: Works with 85° camera separation
- **Geometric Constraints**: Enforces epipolar geometry
- **RANSAC Robustness**: Handles outliers in feature matching
- **Measured Results**: 85.19° angle (target: 85°), 90.5% successful pairs
- **Standard Method**: Limited to <20° baselines, failed on our setup

### Consequences
**Positive**:
- Accurate calibration for wide-angle stereo
- Robust to outliers in overlap zone
- Repeatable results across calibration sessions

**Negative**:
- T vector normalized (baseline distance unknown)
- Lower inlier ratio (3.0%) than narrow-baseline stereo
- Requires good feature distribution in overlap zone

### Implementation
- File: `calibration/stereo_essential_matrix.py`
- Results: `stereo_essential_matrix_results.json`
- Validation: 42 image pairs, 38 successful (90.5%)

### Related
- ADR-009 (Panorama stitching approach)

---

## ADR-003: NVMM Zero-Copy Memory Architecture

**Date**: 2025-09-20
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Jetson Orin NX has limited 102 GB/s memory bandwidth shared between CPU and GPU. Need to minimize memory copies.

### Alternatives Considered
1. **NVMM zero-copy throughout pipeline** (selected)
2. CPU-based processing with GPU copies for inference
3. Mixed CPU/GPU with selective copies

### Decision
Use **NVMM (NVIDIA Memory Manager) buffers** throughout entire pipeline, keeping all video data GPU-resident.

### Rationale
- **Bandwidth Savings**: Eliminates CPU↔GPU copies (~110 MB/frame at 30 FPS = 3.3 GB/s)
- **Unified Memory**: Jetson architecture supports I/O coherency
- **DeepStream Native**: All DeepStream plugins support NVMM
- **Performance**: Measured 70% GPU load vs 95%+ with CPU copies

### Consequences
**Positive**:
- 30 FPS stable at 70% GPU load
- Memory bandwidth well within limits
- Simplified memory management (single allocation)
- Lower power consumption

**Negative**:
- Must use EGL/CUDA for buffer access (more complex)
- Debugging harder (can't easily inspect pixel data)
- Buffer pool sizing critical (OOM if too small)

### Implementation Details
```
Camera → NVMM → Stitching → NVMM → Tiling → NVMM →
Inference → NVMM → VirtualCam → NVMM → Display
```

CPU copies **ONLY** for:
- Metadata (few KB)
- H.264 encoded frames for buffer (~1 MB/frame)
- CSV logging (<1 KB)

### Related
- ADR-001 (Jetson platform)
- ADR-007 (Custom GStreamer plugins)

---

## ADR-004: YOLOv11 FP16 Inference Mode

**Date**: 2025-09-25
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need efficient object detection on Jetson Orin NX for 6× 1024×1024 tiles at 30 FPS.

### Alternatives Considered
1. **YOLOv11n FP16** (selected)
2. YOLOv11s FP16 (larger model, more accurate but slower)
3. YOLOv11n INT8 (quantized, faster but accuracy loss)
4. YOLOv11n FP32 (full precision, too slow)

### Decision
Use **YOLOv11n (nano) with FP16 precision**

### Rationale
- **Performance**: ~20ms inference on 6-tile batch
- **Accuracy**: >95% ball detection, >90% player detection
- **Memory**: 8.5 MB engine file, ~2 GB workspace
- **Tensor Cores**: Ampere architecture optimized for FP16
- **Latency Budget**: Fits within <100ms end-to-end target

**INT8 Rejected**: Accuracy loss on small objects (ball at distance)
**FP32 Rejected**: 40ms+ inference, breaks real-time requirement
**YOLOv11s Rejected**: 35ms inference, diminishing returns on accuracy

### Consequences
**Positive**:
- Real-time 30 FPS achieved
- Good detection quality
- Fits in latency budget

**Negative**:
- Can't use DLA accelerators (require INT8)
- Slight accuracy loss vs FP32 (acceptable tradeoff)

### Configuration
- Model: `yolo11n_mixed_finetune_v9.engine`
- Network mode: FP16 (mode=2)
- Batch size: 6 (one per tile)
- Input size: 1024×1024 per tile

### Related
- ADR-005 (Custom tile batcher)
- ADR-006 (Multi-class detection)

---

## ADR-005: Custom Tile Batcher Plugin

**Date**: 2025-10-01
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need to extract 6× 1024×1024 tiles from 5700×1900 panorama for efficient YOLO inference batching.

### Alternatives Considered
1. **Custom GStreamer plugin (my_tile_batcher)** (selected)
2. Use nvvideoconvert with crop meta (doesn't support batching)
3. CPU-based tile extraction with numpy (too slow, breaks zero-copy)
4. Python appsink/appsrc bridge (high overhead)

### Decision
Develop **custom GStreamer plugin with CUDA tile extraction kernel**

### Rationale
- **Performance**: ~1ms for 6 tiles extraction
- **Zero-Copy**: Direct NVMM buffer manipulation
- **Batch Output**: Creates NvDsBatchMeta with 6 frames
- **Metadata**: Attaches TileRegionInfo to each frame
- **Integration**: Native GStreamer plugin, no Python overhead

### Consequences
**Positive**:
- Minimal latency (<1ms)
- Efficient GPU memory access (coalesced reads/writes)
- Clean integration with nvinfer (expects batch)
- Metadata preserved for coordinate transformation

**Negative**:
- Custom C++/CUDA code to maintain
- Requires recompilation for parameter changes
- GStreamer plugin complexity

### Implementation
- Plugin: `my_tile_batcher/`
- Kernel: `cuda_tile_extractor.cu`
- Tile layout: 6 horizontal tiles, 192px left offset, Y=434
- Block config: (32, 32, 6) blocks × (32, 32, 1) threads

### Tile Positions
```
Tile 0: (192, 434, 1024, 1024)
Tile 1: (1216, 434, 1024, 1024)
Tile 2: (2240, 434, 1024, 1024)
Tile 3: (3264, 434, 1024, 1024)
Tile 4: (4288, 434, 1024, 1024)
Tile 5: (5312, 434, 1024, 1024)
```

### Related
- ADR-004 (YOLO FP16)
- ADR-007 (Custom plugins)

---

## ADR-006: Multi-Class Detection Strategy

**Date**: 2025-10-05
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need to detect multiple object types: ball, players, staff, referees.

### Alternatives Considered
1. **Single multi-class YOLO model** (selected)
2. Separate models for ball vs people (higher latency)
3. Ball-only detection + separate classifier (two-stage slower)

### Decision
Use **single YOLOv11 multi-class model** with 5 classes

### Rationale
- **Efficiency**: Single inference pass for all classes
- **Consistency**: Same latency regardless of object count
- **Simplicity**: One model to train/deploy
- **Accuracy**: Class-specific thresholds (ball: 0.25, others: 0.40)

### Class Definitions
```
Class 0: ball          (confidence threshold: 0.25)
Class 1: player        (confidence threshold: 0.40)
Class 2: staff         (confidence threshold: 0.40) - ignoring for now
Class 3: side_referee  (confidence threshold: 0.40) - ignoring for now
Class 4: main_referee  (confidence threshold: 0.40) - ignoring for now
```

### Consequences
**Positive**:
- No additional latency for multi-class
- Unified detection pipeline
- Easier model versioning

**Negative**:
- Class imbalance (few ball instances vs many players)
- Harder to tune per-class performance
- Model size slightly larger than ball-only

### Post-Processing
- NMS per class (IoU 0.45)
- Class-specific confidence thresholds
- Field mask filtering (all classes)

### Related
- ADR-004 (YOLO FP16)
- ADR-012 (Field mask filtering)

---

## ADR-007: Custom GStreamer Plugins Strategy

**Date**: 2025-10-10
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need panorama stitching, tile batching, and virtual camera functionality not available in standard GStreamer plugins.

### Alternatives Considered
1. **Custom C++/CUDA GStreamer plugins** (selected)
2. Python appsink/appsrc bridges (too slow, breaks zero-copy)
3. External processing with IPC (high latency)
4. OpenCV VideoCapture/VideoWriter (not real-time)

### Decision
Develop **three custom GStreamer plugins** with CUDA acceleration

### Rationale
- **Native Integration**: First-class GStreamer elements
- **Zero-Copy**: Direct NVMM buffer manipulation
- **Performance**: CUDA kernels optimized for Jetson
- **Reusability**: Can be used in other pipelines

### Plugins Developed

#### 1. my_steach (Panorama Stitching)
- **Input**: 2× 3840×2160 RGBA
- **Output**: 5700×1900 RGBA
- **Algorithm**: LUT-based warping + weighted blending
- **Performance**: ~10ms @ 30 FPS

#### 2. my_tile_batcher (Tile Extraction)
- **Input**: 5700×1900 RGBA
- **Output**: Batch of 6× 1024×1024 RGBA
- **Algorithm**: CUDA multi-tile extraction
- **Performance**: ~1ms @ 30 FPS

#### 3. my_virt_cam (Virtual Camera)
- **Input**: 5700×1900 RGBA equirectangular
- **Output**: 1920×1080 RGBA perspective
- **Algorithm**: 3-stage CUDA projection (ray gen → LUT → remap)
- **Performance**: ~21ms @ 47 FPS

### Consequences
**Positive**:
- Optimal performance (GPU-accelerated)
- Clean GStreamer integration
- Property-based configuration
- Caps negotiation support

**Negative**:
- Development complexity (C++/CUDA)
- Maintenance burden (GStreamer API changes)
- Build system dependencies

### Build System
- Makefile-based build
- Dependencies: GStreamer, CUDA, EGL
- Installation: `~/.local/share/gstreamer-1.0/plugins/`

### Related
- ADR-003 (NVMM zero-copy)
- ADR-009 (Panorama stitching)
- ADR-010 (Virtual camera)

---

## ADR-008: 7-Second Buffer Design

**Date**: 2025-10-15
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need delayed playback for analysis branch to catch up with real-time detection (7 seconds lag).

### Alternatives Considered
1. **RAM buffer with timestamp synchronization** (selected)
2. SSD-based circular buffer (too slow, wear issues)
3. Network streaming with delay (adds latency, unreliable)
4. No buffering, real-time only (can't show analyzed detections)

### Decision
Implement **7-second RAM buffer** with frame/audio synchronization

### Rationale
- **Memory Available**: ~3 GB for 210 frames (7s @ 30fps × ~15 MB H.264)
- **Fast Access**: RAM lookup by timestamp (<1ms)
- **Synchronization**: Paired frame/audio buffers
- **Reliability**: No disk I/O, no network dependency

### Design
```
Analysis Branch (real-time)
    │
    ├─→ Process detections
    ├─→ Store in history
    └─→ Buffer frames (appsink → deque)

Buffer Manager (7s delay)
    │
    ├─→ Frame buffer: deque of (timestamp, H.264 data)
    ├─→ Audio buffer: deque of (timestamp, PCM data)
    └─→ Background thread for playback timing

Playback Branch (7s delayed)
    │
    ├─→ Request frame for timestamp
    ├─→ Retrieve from buffer (timestamp lookup)
    ├─→ Push to appsrc
    └─→ Render with analyzed detections
```

### Consequences
**Positive**:
- Smooth playback with synchronized analysis
- No disk I/O bottleneck
- Easy timestamp-based retrieval

**Negative**:
- Uses ~3 GB RAM (acceptable on 16 GB system)
- Deep copies identified as CPU bottleneck (optimization needed)
- O(n) timestamp scans (optimization needed)

### Optimization Opportunities
- Replace deep copies with shallow copies
- Use binary search for timestamp lookup
- Implement ring buffer for O(1) access

### Related
- ADR-011 (Detection history system)
- See `docs/reports/CODEX_report.md` for performance analysis

---

## ADR-009: LUT-Based Panorama Stitching

**Date**: 2025-10-20
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need real-time panorama stitching from dual 4K cameras at 30 FPS on Jetson Orin NX.

### Alternatives Considered
1. **Pre-computed LUT-based warping** (selected)
2. Real-time homography computation (too slow)
3. OpenCV Stitcher class (not real-time, CPU-based)
4. Feature-based stitching per frame (way too slow)

### Decision
Use **pre-computed LUT (Look-Up Table) based warping** with CUDA

### Rationale
- **Performance**: ~10ms for 5700×1900 output (vs 100ms+ for feature-based)
- **Consistency**: No variation in stitch quality between frames
- **GPU Efficiency**: Coalesced memory access, texture caching
- **Quality**: Sub-pixel accuracy with bilinear interpolation

### LUT Generation (Offline)
1. Calibrate cameras (essential matrix method)
2. Compute homography from feature matches
3. Generate coordinate maps: (src_x, src_y) for each output pixel
4. Generate weight maps: (w_left, w_right) for overlap blending
5. Save as binary files (6 maps total)

### Runtime Stitching (CUDA)
```c
For each output pixel (x, y):
    src_x_left = lut_left_x[idx]
    src_y_left = lut_left_y[idx]
    src_x_right = lut_right_x[idx]
    src_y_right = lut_right_y[idx]

    pixel_left = texture_sample(left_camera, src_x_left, src_y_left)
    pixel_right = texture_sample(right_camera, src_x_right, src_y_right)

    w_left = weight_left[idx]
    w_right = weight_right[idx]

    output[idx] = w_left * pixel_left + w_right * pixel_right
```

### Color Correction
- **Asynchronous 2-phase system**: Overlap zone analysis every 30 frames
- **Smoothing factor**: 0.1 (10% update rate)
- **Coefficients**: 6 values (R/G/B gains per camera)
- **Prevents**: Visible seams in varying lighting

### Consequences
**Positive**:
- Real-time 30 FPS achieved
- Seamless blending in overlap zones
- Deterministic output quality

**Negative**:
- Requires offline LUT generation
- Fixed camera geometry (re-generate LUTs if cameras move)
- LUT storage: ~0.3 GB (acceptable)

### Files
- Plugin: `my_steach/`
- LUTs: `warp_maps/*.bin` (6 files)
- Generator: `test_stiching.py`

### Related
- ADR-002 (Stereo calibration)
- ADR-007 (Custom plugins)

---

## ADR-010: 3-Stage Virtual Camera Projection

**Date**: 2025-10-25
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need to extract perspective view from equirectangular panorama with controllable yaw, pitch, roll, FOV.

### Alternatives Considered
1. **3-stage CUDA projection with LUT caching** (selected)
2. Direct OpenCV remap() (CPU-based, too slow)
3. OpenGL texture mapping (requires EGL context, more complex)
4. On-the-fly ray tracing (too slow)

### Decision
Implement **3-stage equirectangular → perspective projection** with aggressive LUT caching

### Algorithm Stages

#### Stage 1: Ray Generation (Camera Intrinsics)
```
For each output pixel (u, v):
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy
    ray_z = 1.0
    ray = normalize(ray_x, ray_y, ray_z)
```

#### Stage 2: LUT Generation (3D Rotation + Spherical Mapping)
```
For each ray:
    rotated_ray = rotation_matrix(yaw, pitch, roll) * ray
    theta = atan2(rotated_ray.x, rotated_ray.z)  // Azimuth
    phi = asin(rotated_ray.y)                    // Elevation

    // Map to panorama coordinates
    src_x = (theta / π) * panorama_width / 2 + panorama_width / 2
    src_y = (phi / (π/2)) * panorama_height / 2 + panorama_height / 2
```

#### Stage 3: Image Remapping (Texture Sampling)
```
For each output pixel:
    src_coords = lut_lookup(yaw, pitch, roll, fov)
    output[idx] = texture_sample(panorama, src_coords.x, src_coords.y)
```

### LUT Caching Strategy
- **Ray Cache**: FOV-dependent (24.9 MB, invalidate on >0.1° FOV change)
- **LUT Cache**: Angle-dependent (16.6 MB, invalidate on >0.1° angle change)
- **EGL Mapping Cache**: 4-8 buffer entries, <1μs lookup
- **Hit Rate**: ~99% during smooth tracking

### Consequences
**Positive**:
- 47.90 FPS achieved (exceeds 30 FPS requirement)
- 20.88ms latency (fits budget)
- Smooth camera movement

**Negative**:
- LUT storage: ~42 MB total
- Cache invalidation logic complexity
- Sub-pixel jitter on rapid changes

### Parameter Ranges
- **yaw**: -90° to +90° (horizontal pan)
- **pitch**: -32° to +22° (vertical tilt)
- **roll**: -28° to +28° (rotation)
- **fov**: 55° to 68° (zoom)

### Related
- ADR-007 (Custom plugins)
- ADR-013 (Intelligent camera control)

---

## ADR-011: Three-Tier Detection History

**Date**: 2025-11-01
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need to store ball detections with different processing stages: raw incoming, cleaned/interpolated, and confirmed (displayed).

### Alternatives Considered
1. **Three-tier storage (raw → processed → confirmed)** (selected)
2. Single history with flags (harder to manage)
3. Two-tier (raw + confirmed) (missing processed state)

### Decision
Implement **DetectionStorage with three tiers**

### Tier Definitions

#### 1. Raw Future History
- **Purpose**: Incoming detections from analysis branch
- **Contents**: All detections (including outliers)
- **Lifecycle**: Added immediately, moved to processed after cleaning

#### 2. Processed Future History
- **Purpose**: Cleaned and interpolated trajectory
- **Contents**: Valid detections + interpolated points
- **Processing**: Outlier removal, gap filling, smoothing
- **Lifecycle**: Generated from raw, moved to confirmed after display

#### 3. Confirmed History
- **Purpose**: Detections already displayed
- **Contents**: Confirmed ball positions shown to user
- **Lifecycle**: Moved from processed after 7s delay (playback)

### Rationale
- **Separation of Concerns**: Each tier has specific role
- **Debugging**: Can inspect raw vs processed
- **Recovery**: Can reprocess from raw if algorithm changes
- **Memory**: Old confirmed entries automatically pruned

### Trajectory Processing
```
Raw History (t=0s to t=10s)
    │
    ├─→ Outlier Detection (TrajectoryFilter)
    ├─→ Blacklist Management
    ├─→ Interpolation (TrajectoryInterpolator)
    │   ├─ Parabolic (flight, gap >1s)
    │   └─ Linear (ground, gap ≤1s)
    │
    ▼
Processed History (t=0s to t=10s)
    │
    ├─→ 7-second delay (BufferManager)
    │
    ▼
Confirmed History (t=-7s to t=3s)
```

### Consequences
**Positive**:
- Clear state transitions
- Easy to debug trajectory issues
- Can reprocess without losing raw data

**Negative**:
- More memory (3 copies of recent data)
- Synchronization between tiers
- Complexity in timestamp management

### Related
- ADR-008 (7-second buffer)
- ADR-014 (Trajectory interpolation)

---

## ADR-012: Binary Field Mask Filtering

**Date**: 2025-11-05
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Need to filter out detections outside the playing field to reduce false positives.

### Alternatives Considered
1. **Binary field mask (PNG)** (selected)
2. Polygon-based field boundary (complex calculations)
3. Rectangle bounding box (too coarse)
4. No filtering (too many false positives)

### Decision
Use **binary field mask image (field_mask.png)** with pixel-perfect boundary

### Rationale
- **Precision**: Pixel-level accuracy
- **Flexibility**: Can handle any field shape
- **Performance**: O(1) lookup per detection
- **Easy Updates**: Edit PNG in image editor

### Implementation
```python
class FieldMaskBinary:
    def __init__(self, mask_path):
        # Load 1900×5700 binary mask (matches panorama size)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    def is_inside_field(self, x, y):
        # Check if pixel is white (field) or black (out)
        return self.mask[y, x] > 0
```

### Mask Generation
1. Capture panorama of empty field
2. Manual annotation in GIMP/Photoshop
3. White pixels = field, black pixels = out of bounds
4. Save as 1-bit PNG (small file size)

### Consequences
**Positive**:
- Eliminates detections in stands, bench area
- Pixel-perfect accuracy
- Fast lookup (~O(1))

**Negative**:
- Requires manual mask creation
- Mask must match panorama geometry
- Re-create mask if cameras move

**Optimization Opportunity**: Move to GPU (texture lookup in CUDA)

### Related
- ADR-006 (Multi-class detection)
- See `docs/reports/CODEX_report.md` for CPU bottleneck

---

## ADR-013: Intelligent Camera Control Strategy

**Date**: 2025-11-08
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Virtual camera should automatically track ball with smooth pursuit and intelligent zoom.

### Alternatives Considered
1. **Speed-based zoom + smooth pursuit + player fallback** (selected)
2. Fixed FOV with tracking (boring, misses close-up)
3. Manual control only (not automated)
4. Kalman filter prediction (overkill, adds latency)

### Decision
Implement **multi-mode intelligent camera control**

### Control Modes

#### 1. Ball Tracking (Primary)
- **Target**: Ball position from history
- **Smooth Factor**: 0.3 (30% new position per frame)
- **Zoom**: Speed-based FOV adjustment
  ```
  target_fov = BASE_FOV + (ball_radius - 20) × ZOOM_RATE
  BASE_FOV = 60°
  ZOOM_RATE = -0.15 (zoom in when closer)
  ```
- **Spherical Correction**: Account for panorama distortion
  ```
  yaw_factor = cos(normalized_yaw × π/2)
  effective_fov = target_fov × yaw_factor
  ```

#### 2. Ball Loss Recovery
- **Trigger**: No detection for 6 frames (0.2s)
- **Action**: Expand FOV at 2°/second (up to 90° max)
- **Relock**: 6-frame confirmation before tracking resumes

#### 3. Player Fallback
- **Trigger**: Ball lost for >3 seconds
- **Target**: Player center-of-mass (EMA smoothed, α=0.18)
- **FOV**: Wider angle to capture multiple players
- **Return**: Switch back to ball when detected

### Consequences
**Positive**:
- Engaging viewing experience
- Automatic zoom on action
- Never loses tracking completely

**Negative**:
- Can be disorienting if ball moves rapidly
- Smooth factor too high = laggy, too low = jittery
- Recovery mode can miss fast ball returns

### Parameters (Tunable)
```python
SMOOTH_FACTOR = 0.3          # Camera movement smoothing
BALL_LOST_THRESHOLD = 6      # Frames before recovery
FALLBACK_TIMEOUT = 3.0       # Seconds before player fallback
FOV_EXPAND_RATE = 2.0        # Degrees per second
RADIUS_SMOOTH_ALPHA = 0.3    # Zoom stability
```

### Related
- ADR-010 (Virtual camera projection)
- ADR-011 (Detection history)

---

## ADR-014: Parabolic Trajectory Interpolation

**Date**: 2025-11-10
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Ball often lost during flight (occlusion, motion blur). Need realistic interpolation.

### Alternatives Considered
1. **Parabolic interpolation (flight) + linear (ground)** (selected)
2. Linear interpolation only (unrealistic for flight)
3. Bezier curves (too flexible, not physically accurate)
4. No interpolation (gaps in trajectory)

### Decision
Use **physics-based interpolation** with mode detection

### Algorithm

#### Mode Detection
```python
if time_gap > 1.0:  # Seconds
    mode = "FLIGHT"  # Likely in air
else:
    mode = "GROUND"  # Likely rolling
```

#### Parabolic Interpolation (Flight)
```python
# Fit parabola: y = a*x^2 + b*x + c
# through (x1, y1) and (x2, y2) with gravity constraint

g = 9.8  # m/s^2 (scaled to pixel space)

# Solve for coefficients
a = -g / (2 * vx^2)
b = vy - a * 2 * x1
c = y1 - a * x1^2 - b * x1

# Generate points
for t in range(num_points):
    x = interpolate_linear(x1, x2, t)
    y = a * x^2 + b * x + c
```

#### Linear Interpolation (Ground)
```python
# Simple linear interpolation
for t in range(num_points):
    x = x1 + (x2 - x1) * t / num_points
    y = y1 + (y2 - y1) * t / num_points
```

### Consequences
**Positive**:
- Realistic ball trajectory visualization
- Smooth camera tracking during occlusion
- User sees predicted path (useful for analysis)

**Negative**:
- Parabola assumption may not fit all cases (low trajectory shots)
- Requires tuning gravity constant for pixel space
- Interpolation error accumulates for long gaps

### Max Interpolation Gap
- **Flight**: 10 seconds maximum
- **Ground**: 5 seconds maximum
- **Longer gaps**: Mark as uncertainty, don't interpolate

### Related
- ADR-011 (Detection history)
- ADR-013 (Camera control)

---

## ADR-015: Modular Refactoring Strategy

**Date**: 2025-11-16
**Status**: ✅ Accepted
**Decision Makers**: edvin3i

### Context
Original codebase grew to 3,015 lines monolithic file, hard to maintain and test.

### Alternatives Considered
1. **Modular refactoring with SOLID principles** (selected)
2. Keep monolithic (unmaintainable)
3. Split by feature (still coupled)
4. Microservices (overkill, adds latency)

### Decision
Refactor to **modular architecture with 8 focused modules**

### Architecture

```
utils/       - General utilities (field mask, CSV, NMS)
core/        - Detection & history management
processing/  - YOLO analysis & tensor processing
rendering/   - Display & virtual camera
pipeline/    - Pipeline builders & buffer manager
```

### Key Principles

#### 1. Single Responsibility
- Each module has one clear purpose
- Example: `HistoryManager` only handles ball history

#### 2. Dependency Injection
- Components receive dependencies via constructor
- Example: `AnalysisProbeHandler` gets `TensorProcessor`, `FieldMask`, etc.

#### 3. Composition Over Inheritance
- Main class composes handlers instead of implementing everything
- Example: `self.display_probe_handler = DisplayProbeHandler(...)`

#### 4. Clear Delegation
- Main class orchestrates, modules execute
- Example: `create_pipeline()` delegates to `PipelineBuilder`

### Metrics
- **Code Reduction**: 76% in main file (3,015 → 712 lines)
- **Method Reduction**: 87% in main class (45 → 6 methods)
- **Modules Created**: 8 focused modules
- **API Compatibility**: 100% (same CLI arguments)

### Consequences
**Positive**:
- Easier to understand and maintain
- Components can be tested independently
- Easy to add new features (extend modules)
- Reusable components

**Negative**:
- More files to navigate
- Learning curve for new developers
- Slight overhead from function calls (negligible)

### Migration
- **Original**: Preserved as `version_masr_multiclass_ORIGINAL_BACKUP.py`
- **Refactored**: `version_masr_multiclass_REFACTORED.py`
- **Testing**: Side-by-side validation before replacement

### Related
- See `new_week/REFACTORING_SUMMARY.md` for detailed delegation map
- See `docs/reports/DEEPSTREAM_CODE_REVIEW.md` for code quality issues

---

## Future Decisions (Pending)

### Pending: DLA Offload for Inference
**Status**: 📋 Under Consideration

**Context**: YOLO inference takes ~20ms on GPU. Could use DLA accelerators.

**Options**:
1. Keep GPU inference (current)
2. Offload to DLA (requires INT8 quantization)
3. Split workload GPU + DLA

**Blockers**: Need INT8 calibration, accuracy validation

---

### Pending: Hardware Camera Synchronization
**Status**: 📋 Under Consideration

**Context**: Currently using software sync. Hardware sync would be more precise.

**Options**:
1. Software sync (current)
2. GPIO master/slave trigger
3. External sync generator

**Blockers**: Hardware modification required, testing setup

---

### Pending: Multi-Game Support
**Status**: 📋 Future Phase 4

**Context**: Run multiple pipelines simultaneously for different games.

**Options**:
1. Single game (current)
2. Multi-process (separate Jetson per game)
3. Multi-pipeline (same Jetson, resource allocation)
4. Cloud offload (dGPU server)

**Blockers**: Resource constraints on Jetson, testing infrastructure

---

## Decision Process

### When to Create an ADR
- Significant architectural choice
- Technology selection
- Design pattern adoption
- Performance tradeoff
- API design

### ADR Template
```markdown
## ADR-XXX: Title

**Date**: YYYY-MM-DD
**Status**: 📋 Proposed | ✅ Accepted | ❌ Rejected | ⚠️ Deprecated
**Decision Makers**: Names

### Context
What is the issue/problem?

### Alternatives Considered
1. Option 1
2. Option 2
3. ...

### Decision
What was decided?

### Rationale
Why this option?

### Consequences
**Positive**:
- Benefit 1
- Benefit 2

**Negative**:
- Cost 1
- Cost 2

### Related
- Link to other ADRs
- Link to code/docs
```

---

**Document Owner**: edvin3i
**Last Updated**: 2025-11-17
**Review Frequency**: On major architectural changes
