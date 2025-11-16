# nvdsstitch Plugin - 360° Panorama Stitching

## Overview

**nvdsstitch** is a custom NVIDIA DeepStream GStreamer plugin designed for real-time 360° panorama stitching from dual fisheye camera streams. It combines two 4K fisheye camera inputs into a single equirectangular panoramic output using GPU-accelerated CUDA processing.

**Version**: 1.0
**Platform**: NVIDIA Jetson Orin NX (DeepStream 7.1, JetPack 6.2)
**Performance**: 40-50 FPS on dual 4K (3840×2160) input

## Key Features

- Real-time 360° panorama creation from dual fisheye cameras
- LUT-based warping for precise spherical projection
- GPU-accelerated CUDA kernels for high-performance stitching
- Automatic color correction across overlap zones
- Zero-copy EGL interop on Jetson platforms
- Configurable output dimensions (default: 5700×1900)

## Plugin Architecture

### Pipeline Position
```
nvarguscamerasrc → nvvideoconvert → nvstreammux → nvdsstitch → [output]
nvarguscamerasrc → nvvideoconvert ↗
```

### Data Flow
```
Input (Batch) → Split to Intermediate → CUDA Warp & Blend → Output Panorama
    ↓                    ↓                     ↓                    ↓
2×4K NVMM          VIC Copy              GPU Kernel          Equirectangular
(batched)        (parallel)           (LUT + weights)         (panorama)
```

## Technical Specifications

### Input Requirements
- **Format**: RGBA (from nvstreammux batch)
- **Resolution**: 3840×2160 per camera (4K)
- **Memory**: NVMM (NVIDIA Memory Manager)
- **Source Count**: 2 (left and right cameras)
- **Batch Size**: 2

### Output Specifications
- **Format**: RGBA
- **Default Resolution**: 5700×1900 pixels
- **Maximum Tested**: 6528×1800 pixels
- **Aspect Ratio**: ~3:1 (panoramic)
- **Projection**: Equirectangular (spherical mapping)
- **Memory**: NVMM (GPU memory)
- **Field of View**: Horizontal ~180-220°, Vertical ~90-120°

## Configuration Properties

| Property | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `left-source-id` | uint | 0 | No | Source ID for left camera |
| `right-source-id` | uint | 1 | No | Source ID for right camera |
| `gpu-id` | uint | 0 | No | GPU device ID |
| `use-egl` | boolean | TRUE | No | Enable EGL interop on Jetson |
| `panorama-width` | uint | - | **YES** | Output panorama width |
| `panorama-height` | uint | - | **YES** | Output panorama height |

### Example Usage
```bash
nvdsstitch \
    left-source-id=0 \
    right-source-id=1 \
    panorama-width=5700 \
    panorama-height=1900 \
    gpu-id=0
```

## Algorithm Details

### LUT-Based Warping

The plugin uses pre-computed Look-Up Tables (LUTs) for pixel mapping:

**Required LUT Files** (in `warp_maps/` directory):
- `lut_left_x.bin` - Left camera X coordinates (5700×1900×4 bytes)
- `lut_left_y.bin` - Left camera Y coordinates
- `lut_right_x.bin` - Right camera X coordinates
- `lut_right_y.bin` - Right camera Y coordinates
- `weight_left.bin` - Left camera blend weights (0.0-1.0)
- `weight_right.bin` - Right camera blend weights (0.0-1.0)

**LUT Format**:
- Data type: 32-bit float (little-endian)
- Coordinate range: -1000 to 10000 (pixel coordinates in source image)
- Weight range: 0.0 to 1.0 (blend weights)

### CUDA Kernel

**Function**: `panorama_lut_kernel()`

**Algorithm**:
1. For each output pixel (x, y):
2. Read LUT coordinates for both cameras
3. Check validity of coordinates (valid if ≥ 0)
4. Apply bilinear interpolation from input images
5. Apply color correction gains
6. Weighted average blending based on overlap
7. Write to output with flip transform

**Bilinear Interpolation**:
```
result = (1-fx)(1-fy)·p00 + fx(1-fy)·p10 + (1-fx)fy·p01 + fx·fy·p11
```

**Weighted Blending**:
```
result = (pixel_L × gains_L × w_L + pixel_R × gains_R × w_R) / (w_L + w_R)
```

### Color Correction

- **Analysis**: `analyze_overlap_zone_kernel()` samples overlap region every 30 frames
- **Method**: Shared memory reduction for R,G,B statistics
- **Gains**: 6 floats stored in GPU constant memory (R,G,B per camera)
- **Smoothing**: 10% exponential smoothing with previous values
- **Update**: Asynchronous, non-blocking

## GPU Performance

### CUDA Configuration
```cpp
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 8
Threads per block = 256
Compute Capability = SM 8.7 (Jetson Orin)
```

### Memory Usage (5700×1900 output)
```
LUT Maps:        6 × 43.3 MB = 260 MB
Input Buffers:   2 × 32 MB = 64 MB
Output Pool:     8 × 43.3 MB = 346 MB
Total:          ~670 MB GPU memory
```

### Optimizations
- **VIC Engine Offloading**: Buffer copying uses Video Image Compositor (+18% FPS)
- **EGL Zero-Copy**: Direct CUDA access to display buffers (Jetson only)
- **Asynchronous Processing**: Color correction in background stream
- **Resource Caching**: EGL resources cached (300 frame TTL)

## Python Scripts

### panorama_cameras_realtime.py
Real-time panorama from CSI cameras on Jetson.

**Usage**:
```bash
python panorama_cameras_realtime.py [left_cam] [right_cam] [display_mode]
# Example: python panorama_cameras_realtime.py 0 1 egl
```

**Features**:
- Auto-detects camera sensors
- Multiple display modes: EGL, X11, auto-scaled
- Live source optimization
- 30 FPS batched push

### panorama_stream.py
Panorama from video files with advanced features.

**Usage**:
```bash
python panorama_stream.py left.mp4 right.mp4 [mode] [--adv] [--noloop]
# Examples:
python panorama_stream.py left.mp4 right.mp4 egl
python panorama_stream.py left.mp4 right.mp4 file  # Save to file
```

**Modes**:
- Simple: Basic file playback
- Advanced (`--adv`): Programmatic element creation
- File output: H.264 encoding to disk
- Loop: Continuous playback (default)

### test_fps.py
Performance benchmarking tool.

**Usage**:
```bash
python test_fps.py left.mp4 right.mp4 [duration_seconds]
# Example: python test_fps.py left.mp4 right.mp4 60  # 60-second test
```

**Metrics**:
- Total frames processed
- Instantaneous FPS (5-second intervals)
- Average FPS
- Average latency (ms/frame)

**Performance Thresholds**:
- Excellent: ≥45 FPS
- Good: 40-45 FPS
- Acceptable: 30-40 FPS
- Low: <30 FPS

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
cd my_steach
make                # Build libnvdsstitch.so
make install        # Install to ~/.local/share/gstreamer-1.0/plugins/
```

### Build Targets
```bash
make                # Build plugin
make install        # Install to user plugins directory
make test           # Run test pipeline
make profile        # Profile with Nsight Systems
make clean          # Remove build artifacts
```

### Output
- **Library**: `libnvdsstitch.so`
- **Size**: ~500KB (with debug symbols)
- **Install Path**: `~/.local/share/gstreamer-1.0/plugins/`

## Integration Example

Complete pipeline example:
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

pipeline = Gst.parse_launch("""
    nvarguscamerasrc sensor-id=0 !
    video/x-raw(memory:NVMM),width=3840,height=2160,format=NV12,framerate=30/1 !
    nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
    m.sink_0

    nvarguscamerasrc sensor-id=1 !
    video/x-raw(memory:NVMM),width=3840,height=2160,format=NV12,framerate=30/1 !
    nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
    m.sink_1

    nvstreammux name=m batch-size=2 width=3840 height=2160 !
    nvdsstitch panorama-width=5700 panorama-height=1900 !
    nveglglessink
""")

pipeline.set_state(Gst.State.PLAYING)
```

## Troubleshooting

### Black/Empty Output
- Verify LUT files exist in `warp_maps/` directory
- Check LUT file sizes match panorama dimensions
- Ensure source IDs are correct (0 and 1 for typical setup)

### Low FPS
- Reduce output resolution
- Check GPU memory availability
- Verify VIC engine is enabled (Jetson)
- Disable EGL on x86 platforms

### Color Mismatch
- Wait 5-10 seconds for color correction convergence
- Check camera exposure/white balance settings
- Verify LUT weight maps are properly calibrated

## File Locations

- **Source Code**: `/home/user/ds_pipeline/my_steach/src/`
- **Plugin Binary**: `/home/user/ds_pipeline/my_steach/libnvdsstitch.so`
- **Configuration**: `/home/user/ds_pipeline/my_steach/src/nvdsstitch_config.h`
- **Python Scripts**: `/home/user/ds_pipeline/my_steach/*.py`
- **Build System**: `/home/user/ds_pipeline/my_steach/Makefile`
- **LUT Maps**: `/home/user/ds_pipeline/my_steach/warp_maps/` (if exists)

## License

Custom NVIDIA DeepStream plugin for research and development purposes.

## References

- NVIDIA DeepStream SDK: https://developer.nvidia.com/deepstream-sdk
- GStreamer Plugin Development: https://gstreamer.freedesktop.org/documentation/plugin-development/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
