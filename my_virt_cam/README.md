# nvdsvirtualcam Plugin - Virtual Camera with Ball Tracking

## Overview

**nvdsvirtualcam** is a custom GStreamer plugin that creates a virtual camera from an equirectangular panoramic image. It provides real-time pan, tilt, and zoom capabilities with automatic ball tracking for football/soccer video analysis.

**Version**: 1.0
**Platform**: NVIDIA Jetson Orin NX (DeepStream 7.1, JetPack 6.2)
**Performance**: 45-48 FPS on Jetson Orin with full pipeline

## Key Features

- Virtual camera with pan/tilt/zoom from panoramic image
- GPU-accelerated CUDA coordinate transformation
- Auto-follow ball tracking with smooth transitions
- Predictive edge offset for better action capture
- Speed-adaptive zoom
- LUT-based optimization for real-time performance
- Zero-copy NVMM memory operations
- Configurable output resolution (default: 1920×1080)

## Plugin Architecture

### Pipeline Position
```
nvdsstitch → [optional: nvtilebatcher → nvinfer] → nvdsvirtualcam → output
```

### Coordinate Transformation Flow
```
Output Pixel (x, y)
       ↓ [Camera Ray Generation]
3D Ray in Camera Space (rx, ry, rz)
       ↓ [Rotation: Roll → Pitch → Yaw]
3D Ray in World Space (final_x, final_y, final_z)
       ↓ [Spherical Projection]
Spherical Angles (λ, φ)
       ↓ [Equirectangular Mapping]
Panorama Pixel (u, v)
       ↓ [Nearest Neighbor Interpolation]
RGBA Color Value
```

## Technical Specifications

### Input Requirements
- **Format**: RGBA
- **Resolution**: Configurable (e.g., 5700×1900 or 6528×1800)
- **Memory**: NVMM (NVIDIA Memory Manager)
- **Projection**: Equirectangular (spherical)
- **Coverage**: Horizontal -90° to +90°, Vertical -27° to +27°

### Output Specifications
- **Format**: RGBA
- **Default Resolution**: 1920×1080 (Full HD)
- **Configurable Range**: 640×480 to 3840×2160
- **Memory**: NVMM (GPU memory)
- **Aspect Ratio**: 16:9 (fixed)
- **Frame Rate**: Same as input (typically 30 FPS)

## Configuration Properties

### Required Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `panorama-width` | uint | - | **REQUIRED** Input panorama width |
| `panorama-height` | uint | - | **REQUIRED** Input panorama height |

### Camera Control Properties

| Property | Type | Range | Default | Description |
|----------|------|-------|---------|-------------|
| `yaw` | float | -90 to +90 | 0.0 | Horizontal rotation (degrees) |
| `pitch` | float | -27 to +27 | 0.0 | Vertical tilt (degrees) |
| `roll` | float | -28 to +28 | 0.0 | Image rotation (auto-calculated) |
| `fov` | float | 40 to 68 | 68.0 | Field of view (degrees) |
| `output-width` | uint | 640-3840 | 1920 | Output video width |
| `output-height` | uint | 480-2160 | 1080 | Output video height |

### Ball Tracking Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `auto-follow` | bool | false | Enable automatic ball tracking |
| `smooth-factor` | float | 0.3 | Camera movement smoothness (0.0-1.0) |
| `ball-x` | float | - | Ball X coordinate in panorama (pixels) |
| `ball-y` | float | - | Ball Y coordinate in panorama (pixels) |
| `ball-actual-radius` | float | 20.0 | Ball radius in panorama (pixels) |
| `target-ball-size` | float | 0.055 | Desired ball size on screen (0.01-0.15) |

### Example Usage
```bash
nvdsvirtualcam \
    panorama-width=5700 \
    panorama-height=1900 \
    yaw=0.0 \
    pitch=0.0 \
    fov=68.0 \
    auto-follow=true \
    smooth-factor=0.3
```

## Virtual Camera Mechanics

### Panoramic Projection System

The plugin uses an **equirectangular (spherical) projection**:

```
Panorama Coverage:
- Horizontal: 180° (from -90° to +90°)
- Vertical: 54° (from -27° to +27°)
- Asymmetric: More coverage downward (field-focused)
```

### Camera Transformation

#### 1. Camera Ray Generation
For each output pixel, compute 3D ray based on FOV:
```cpp
float f = 0.5 * width / tan(fov_rad / 2.0);  // Focal length
float nx = (x - cx) / f;  // Normalized X
float ny = (y - cy) / f;  // Normalized Y
float ray[3] = {nx/len, ny/len, 1.0/len};  // Unit vector
```

#### 2. Rotation Application
Rotations applied in order: **ROLL → PITCH → YAW**

```cpp
// Roll (around Z-axis - image rotation)
rx_roll = rx * cos(roll) - ry * sin(roll);
ry_roll = rx * sin(roll) + ry * cos(roll);

// Pitch (around X-axis - vertical tilt)
ry_pitch = ry_roll * cos(pitch) - rz_roll * sin(pitch);
rz_pitch = ry_roll * sin(pitch) + rz_roll * cos(pitch);

// Yaw (around Y-axis - horizontal pan)
final_x = rx_pitch * cos(yaw) + rz_pitch * sin(yaw);
final_z = -rx_pitch * sin(yaw) + rz_pitch * cos(yaw);
```

#### 3. Spherical Coordinate Conversion
```cpp
lambda = atan2(final_x, final_z);  // Longitude
phi = asin(final_y);               // Latitude
```

#### 4. Panorama Pixel Mapping
```cpp
u_norm = (lambda - lon_min) / (lon_max - lon_min);
v_norm = (phi - lat_min) / (lat_max - lat_min);
u = u_norm * (pano_width - 1);
v = v_norm * (pano_height - 1);
```

### LUT Optimization

**Problem**: Spherical math is expensive (~500-1500μs per frame)

**Solution**: Pre-compute remap coordinates (u, v) for all output pixels
- Store in GPU memory as Look-Up Table (7.9 MB for 1920×1080)
- Only recalculate when angles change >0.1°
- Typical update frequency: Every 2-3 frames with smooth tracking

**Performance Impact**:
- LUT generation: 2-5ms (occasional)
- LUT lookup: ~500-1500μs per frame (always)
- Net savings: ~85-95% compared to per-pixel calculation

## Ball Tracking Implementation

### Tracking Pipeline

```
Ball Detection (external) → Set ball-x, ball-y, ball-radius
                                    ↓
                         update_camera_from_ball()
                                    ↓
                    Convert pixels → angles (yaw, pitch)
                                    ↓
                    Add edge offset (±8° yaw, ±4° pitch)
                                    ↓
                    Calculate zoom from ball size
                                    ↓
                    Apply spherical boundary limits
                                    ↓
                    smooth_camera_tracking()
                                    ↓
                    Update LUT if angles changed >0.1°
                                    ↓
                    Render frame with new view
```

### Coordinate Conversion (Pixel → Angle)

**Function**: `pano_xy_to_yaw_pitch()`

```cpp
// X coordinate → Yaw (horizontal angle)
norm_x = x / (pano_width - 1);
yaw = LON_MIN + norm_x * (LON_MAX - LON_MIN);
// Example: x=2850 (center of 5700) → yaw = 0°

// Y coordinate → Pitch (vertical angle)
norm_y = y / (pano_height - 1);
pitch = LAT_MAX - norm_y * (LAT_MAX - LAT_MIN);
// Example: y=950 (center of 1900) → pitch = 0°
```

### Predictive Edge Offset

When ball approaches frame edge, camera pre-shifts to keep action visible:

```cpp
const float EDGE_DISTANCE = 300.0px;

// Horizontal offset
if (ball_x < EDGE_DISTANCE)
    offset_yaw = +8.0°;  // Ball left → shift right
else if (ball_x > width - EDGE_DISTANCE)
    offset_yaw = -8.0°;  // Ball right → shift left

// Vertical offset
if (ball_y < EDGE_DISTANCE)
    offset_pitch = -4.0°;  // Ball top → shift down
else if (ball_y > height - EDGE_DISTANCE)
    offset_pitch = +4.0°;  // Ball bottom → shift up
```

### Auto-Zoom from Ball Size

```cpp
// Larger ball (closer) → wider FOV
// Smaller ball (farther) → narrower FOV (zoom in)
float radius = clamp(ball_radius, 5.0, 100.0);
float fov_range = FOV_MAX - FOV_MIN;  // 68 - 55 = 13°
float radius_range = 100.0 - 5.0;     // 95px
float slope = fov_range / radius_range;
target_fov = FOV_MIN + (radius - 5.0) * slope;
```

**Result**: Camera zooms in when ball is far, zooms out when ball is close

### Smooth Tracking

**Exponential smoothing** prevents jerky camera motion:

```cpp
// Dead zones prevent micro-movements
const float DEAD_ZONE = 0.1°;  // For yaw/pitch
const float FOV_DEAD_ZONE = 0.5°;

// Smooth interpolation (default: 30% per frame)
yaw_diff = target_yaw - current_yaw;
if (abs(yaw_diff) > DEAD_ZONE)
    current_yaw += yaw_diff * smooth_factor;

// At 30 FPS with smooth_factor=0.3:
// Full stabilization takes ~0.3 seconds (9 frames)
```

## GPU Performance

### CUDA Configuration

**Optimal Settings** (empirically determined):
```cpp
dim3 block(32, 16);  // 512 threads per block (optimal for this kernel)
dim3 grid(60, 68);   // For 1920×1080 output
// Total: 4,080 blocks × 512 threads = ~2M parallel threads
```

**Why 32×16 is optimal**:
- 16×16 (256 threads): **47.90 FPS** ✓
- 32×16 (512 threads): **43.88 FPS** (-8.5%)
- 32×8 (256 threads): **44.40 FPS** (-7.4%)

**Reason**: Balance between occupancy, register usage, and shared memory

### Processing Time Breakdown

| Operation | Time (μs) | Frequency |
|-----------|-----------|-----------|
| Parameter snapshot | ~1 | Every frame |
| update_camera_from_ball() | 5-10 | Every frame |
| update_lut_if_needed() | 0 or 2000-5000 | On angle change |
| Buffer mapping | 50-100 | Every frame |
| CUDA kernel | 500-1500 | Every frame |
| Stream synchronization | 100-300 | Every frame |
| **Total** | **700-2000 μs** | **Per frame** |

**Theoretical Max**: 500-1400 FPS
**Actual**: Limited by input (30 FPS) or encoding

### Memory Usage

```
CUDA Memory:
- Rays: 1920×1080×3×4 = 23.7 MB (reused for all FOV)
- LUT U: 1920×1080×4 = 7.9 MB
- LUT V: 1920×1080×4 = 7.9 MB
- Input panorama: 5700×1900×4 = 43.3 MB (NVMM, not copied)
- Output buffers: 8 × 1920×1080×4 = 63.6 MB (NVMM)
Total: ~146 MB GPU memory
```

### Optimization Techniques

1. **Fixed Buffer Pool**: 8 pre-allocated buffers (round-robin)
2. **EGL Cache**: Input buffer CUDA resources cached
3. **LUT Caching**: Only update on significant angle change (>0.1°)
4. **Asynchronous Processing**: Non-blocking CUDA stream
5. **Resource Pre-registration**: EGL images registered once

## Python Integration Example

### Basic Usage
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

pipeline = Gst.parse_launch("""
    filesrc location=panorama.mp4 ! qtdemux ! h264parse !
    nvv4l2decoder ! nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
    nvdsvirtualcam panorama-width=5700 panorama-height=1900
        yaw=0.0 pitch=0.0 fov=68.0 !
    nveglglessink
""")

pipeline.set_state(Gst.State.PLAYING)
```

### With Ball Tracking
```python
vcam = pipeline.get_by_name("virtualcam")

# Enable auto-follow
vcam.set_property("auto-follow", True)
vcam.set_property("smooth-factor", 0.3)

# Update ball position every frame
def on_ball_detected(ball_x, ball_y, ball_radius):
    vcam.set_property("ball-x", float(ball_x))
    vcam.set_property("ball-y", float(ball_y))
    vcam.set_property("ball-actual-radius", float(ball_radius))
```

### Manual Camera Control
```python
vcam = pipeline.get_by_name("virtualcam")

# Pan/tilt/zoom control
vcam.set_property("yaw", 15.0)    # Look right
vcam.set_property("pitch", -5.0)  # Look down
vcam.set_property("fov", 55.0)    # Zoom in
```

## Test Scripts

### Interactive Tests

**test_virtual_camera_keyboard.py**: W/A/S/D keys to move virtual ball
**test_virtual_camera_sliders.py**: Visual sliders showing camera parameters

### Performance Tests

**test_full_pipeline.py**: Complete pipeline test (2×4K → panorama → virtual cam)
**Result**: **47.90 FPS** average (excellent)

### Validation Tests

**test_boundaries_full.py**: Boundary testing at different FOV values
**test_spherical_boundaries.py**: Spherical geometry validation
**test_camera_range.py**: Safe camera movement range finder

### Analysis Tests

**test_ball_positions.py**: Pitch angle calculation for Y positions
**test_real_fov_coverage.py**: Angular coverage calculation
**test_zoom_sweep.py**: Automated FOV sweep test

### Simple Tests

**test_vcam_simple.py**: Static image test
**test_virtual_camera_images.py**: Multi-position test matrix

## Build Instructions

### Prerequisites
```bash
# CUDA 12.6
# GStreamer 1.0 development libraries
# DeepStream 7.1 SDK
# NVIDIA Jetson JetPack 6.2
```

### Compilation
```bash
cd my_virt_cam/src
make                # Build libnvdsvirtualcam.so
make install        # Install to system plugins directory
```

## Integration in Main Pipeline

Complete football analysis pipeline:
```python
pipeline = Gst.parse_launch("""
    # Input: Dual 4K cameras
    nvarguscamerasrc sensor-id=0 ! ... ! m.sink_0
    nvarguscamerasrc sensor-id=1 ! ... ! m.sink_1

    # Stitch to panorama
    nvstreammux name=m batch-size=2 !
    nvdsstitch panorama-width=5700 panorama-height=1900 !
    tee name=t

    # Branch 1: Analysis (tile extraction + YOLO)
    t. ! queue !
    nvtilebatcher !
    nvinfer !
    fakesink

    # Branch 2: Display (virtual camera)
    t. ! queue !
    nvdsvirtualcam panorama-width=5700 panorama-height=1900
        auto-follow=true !
    nvvideoconvert !
    nvv4l2h264enc !
    filesink location=output.mp4
""")
```

## Troubleshooting

### Black/Empty Output
- Verify panorama dimensions are set correctly
- Check yaw/pitch/roll are within valid ranges
- Ensure FOV is between 40-68°
- Verify input is RGBA NVMM format

### Camera Doesn't Move
- Check auto-follow is enabled
- Verify ball-x, ball-y properties are being updated
- Check smooth-factor is not 0.0
- Ensure angles are changing >0.1° (LUT update threshold)

### Jerky Camera Motion
- Increase smooth-factor (0.5-0.8 for smoother motion)
- Check frame rate is stable (30 FPS)
- Verify GPU is not overloaded

### Low FPS
- Reduce output resolution (e.g., 1280×720)
- Check GPU memory availability
- Verify LUT caching is working (angles not changing every frame)

## File Locations

- **Source Code**: `/home/user/ds_pipeline/my_virt_cam/src/`
- **Plugin Binary**: `/home/user/ds_pipeline/my_virt_cam/src/libnvdsvirtualcam.so`
- **Test Scripts**: `/home/user/ds_pipeline/my_virt_cam/test_*.py`
- **Configuration**: `/home/user/ds_pipeline/my_virt_cam/src/nvdsvirtualcam_config.h`

## License

Custom NVIDIA DeepStream plugin for research and development purposes.

## References

- NVIDIA DeepStream SDK: https://developer.nvidia.com/deepstream-sdk
- Equirectangular Projection: https://en.wikipedia.org/wiki/Equirectangular_projection
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- GStreamer Plugin Development: https://gstreamer.freedesktop.org/documentation/plugin-development/
