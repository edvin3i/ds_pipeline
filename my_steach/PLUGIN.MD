# MY_STEACH - Panorama Stitching Plugin

## Overview

`nvdsstitch` is a custom GStreamer plugin for real-time stereo panorama stitching on NVIDIA Jetson platforms. It combines two 4K camera streams into a seamless equirectangular panorama using GPU-accelerated LUT-based warping.

**Location**: `~/ds_pipeline/my_steach/`
**Plugin Name**: `nvdsstitch`
**Type**: GstBaseTransform
**Source**: 3,242 lines (C++/CUDA)

## Key Features

- **Real-time Performance**: 30 FPS on Jetson Orin NX
- **LUT-based Warping**: Pre-computed coordinate maps for efficient stitching
- **Dynamic Color Correction**: Asynchronous overlap analysis with smoothing
- **Seamless Blending**: Weight-based alpha blending in overlap zones
- **EGL Support**: Zero-copy texture mapping for Jetson
- **Configurable Output**: Any panorama size via GStreamer properties

## Input/Output Specifications

### Input
```
Format:        RGBA (NV12 auto-converted)
Resolution:    3840×2160 per stream
Streams:       2 (left and right cameras)
Memory:        NVMM (GPU-resident)
Interface:     nvstreammux batch with batch-size=2
Source IDs:    Configurable (default: 0=left, 1=right)
```

### Output
```
Format:        RGBA (32-bit)
Resolution:    Configurable (examples: 5700×1900, 4096×2048, 6528×1800)
Projection:    Equirect angular (360° panorama)
Memory:        NVMM with EGL mapping
Buffer Pool:   8 pre-allocated fixed buffers
```

## Algorithm Details

### LUT-Based Stitching

The plugin uses **6 pre-computed binary maps**:

| Map File | Size | Purpose |
|----------|------|---------|
| lut_left_x.bin | ~39 MB | Left camera X coordinates |
| lut_left_y.bin | ~39 MB | Left camera Y coordinates |
| lut_right_x.bin | ~39 MB | Right camera X coordinates |
| lut_right_y.bin | ~39 MB | Right camera Y coordinates |
| weight_left.bin | ~39 MB | Left camera blend weight |
| weight_right.bin | ~39 MB | Right camera blend weight |

**Total GPU Memory**: ~234 MB (loaded once at initialization)

### CUDA Kernel: panorama_lut_kernel

**File**: `src/cuda_stitch_kernel.cu:350-520`

**Launch Configuration**:
```cuda
dim3 block(32, 8, 1);  // 256 threads per block
dim3 grid((width + 31) / 32, (height + 7) / 8);
// Example for 5700×1900: 179×238 = 42,602 blocks
```

**Processing Flow**:
```c
For each output pixel (x, y):
1. Look up source coordinates from LUT:
   src_x_left = lut_left_x[y][x]
   src_y_left = lut_left_y[y][x]
   src_x_right = lut_right_x[y][x]
   src_y_right = lut_right_y[y][x]

2. Bilinear interpolation sampling:
   pixel_left = bilinear_sample(input_left, src_x_left, src_y_left)
   pixel_right = bilinear_sample(input_right, src_x_right, src_y_right)

3. Apply color correction:
   pixel_left *= color_gains[0..2]   // R, G, B gains
   pixel_right *= color_gains[3..5]

4. Weighted blending:
   weight_sum = weight_left + weight_right
   if weight_sum > 0:
       output = (pixel_left × weight_left + pixel_right × weight_right) / weight_sum
   else:
       output = black  // Outside camera coverage

5. Optional edge brightness boost (disabled by default)

6. Write to output with flip transformation (vertical/horizontal)
```

**Optimization Features**:
- Shared memory for temporary data
- Coalesced memory access patterns
- Texture memory support (legacy, optional)
- L2 cache optimization for LUT reads

### Color Correction System

**File**: `src/cuda_stitch_kernel.cu:104-238`

The plugin implements a **2-phase asynchronous color correction** system:

#### Phase 1: Overlap Analysis (every 30 frames)

**Kernel**: `analyze_overlap_colors_kernel`

```cuda
For each pixel in overlap zone (where both weights significant):
1. Sample pixels from both cameras
2. Compute RGB intensity differences
3. Atomic accumulate sums:
   atomic_add(&sum_left_R, pixel_left.r)
   atomic_add(&sum_right_R, pixel_right.r)
   // ... same for G, B
   atomic_add(&pixel_count, 1)

4. Shared memory reduction for block-level aggregation
5. Final atomic updates to device global counters
```

**Asynchronous Execution**:
- Runs on low-priority CUDA stream
- Does not block main stitching kernel
- Typical overhead: 2-5ms per 30-frame interval

#### Phase 2: Gain Calculation & Smoothing

```cpp
// CPU-side smoothing (every 30 frames)
float alpha = 0.1;  // 10% update rate

if (pixel_count > MIN_SAMPLES) {
    float avg_left_R = sum_left_R / pixel_count;
    float avg_right_R = sum_right_R / pixel_count;
    float ratio = avg_right_R / (avg_left_R + 1e-6);
    
    // EMA smoothing
    new_gain_R = (1 - alpha) × current_gain_R + alpha × ratio;
    current_gain_R = new_gain_R;
    
    // ... same for G, B channels
}
```

**Configuration**:
```c
COLOR_UPDATE_INTERVAL = 30 frames
COLOR_SMOOTHING_FACTOR = 0.1
MIN_OVERLAP_SAMPLES = 1000 pixels
```

## GStreamer Properties

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `left-source-id` | uint | 0 | 0-255 | Source ID for left camera stream |
| `right-source-id` | uint | 1 | 0-255 | Source ID for right camera stream |
| `gpu-id` | uint | 0 | 0-7 | CUDA device ID |
| `panorama-width` | uint | 4096 | 640-8192 | Output panorama width (pixels) |
| `panorama-height` | uint | 2048 | 480-4096 | Output panorama height (pixels) |
| `use-egl` | boolean | FALSE | - | Enable EGL texture caching (Jetson) |

## Usage Examples

### Basic Pipeline (File Sources)
```bash
gst-launch-1.0 \
    filesrc location=left.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
    nvvideoconvert ! video/x-raw\(memory:NVMM\),format=RGBA ! queue ! mux.sink_0 \
    filesrc location=right.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
    nvvideoconvert ! video/x-raw\(memory:NVMM\),format=RGBA ! queue ! mux.sink_1 \
    nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
    nvdsstitch left-source-id=0 right-source-id=1 \
               panorama-width=5700 panorama-height=1900 ! \
    nveglglessink
```

### Live Cameras (MIPI CSI)
```bash
gst-launch-1.0 \
    nvarguscamerasrc sensor-id=0 ! \
    video/x-raw\(memory:NVMM\),width=3840,height=2160,framerate=30/1,format=NV12 ! \
    nvvideoconvert ! video/x-raw\(memory:NVMM\),format=RGBA ! queue ! mux.sink_0 \
    nvarguscamerasrc sensor-id=1 ! \
    video/x-raw\(memory:NVMM\),width=3840,height=2160,framerate=30/1,format=NV12 ! \
    nvvideoconvert ! video/x-raw\(memory:NVMM\),format=RGBA ! queue ! mux.sink_1 \
    nvstreammux name=mux batch-size=2 width=3840 height=2160 ! \
    nvdsstitch panorama-width=5700 panorama-height=1900 ! \
    nveglglessink
```

### Python GStreamer Pipeline
```python
from gi.repository import Gst

pipeline_str = """
    nvstreammux name=mux batch-size=2 width=3840 height=2160 !
    nvdsstitch name=stitch 
        left-source-id=0 
        right-source-id=1 
        panorama-width=5700 
        panorama-height=1900 !
    nveglglessink
"""

pipeline = Gst.parse_launch(pipeline_str)
# ... attach sources to mux.sink_0 and mux.sink_1
```

## Performance Characteristics

### Computational Complexity

For 5700×1900 output:
```
Total output pixels: 10,830,000
Kernel blocks: 179 × 238 = 42,602 blocks
Threads per block: 32 × 8 = 256 threads
Total threads: 10,906,112 (slight overallocation)

Per-pixel operations:
- 4× LUT lookups (x/y for left/right)
- 2× bilinear interpolations (8 texture reads)
- 6× color correction multiplies
- 2× alpha blending operations
- 1× output write

Estimated: ~50-100 FLOPs per pixel
Total: ~540M-1080M FLOPs per frame
```

### Memory Bandwidth

```
Input buffers:  2 × (3840 × 2160 × 4 bytes) = 66.3 MB
Output buffer:  5700 × 1900 × 4 bytes = 43.3 MB
LUT maps:       234 MB (cached, read once)

Per-frame bandwidth: ~110 MB
At 30 FPS: 3.3 GB/s (3.2% of 102 GB/s total)
```

### Measured Performance (Jetson Orin NX)

| Metric | Value |
|--------|-------|
| **Throughput** | 30 FPS (real-time) |
| **Kernel Latency** | ~5-10 ms per frame |
| **Color Analysis** | ~2-5 ms (asynchronous, every 30 frames) |
| **Memory Usage** | ~350 MB (LUTs + buffers) |
| **GPU Load** | ~15-20% @ 30 FPS |
| **Power Draw** | Included in system 40W budget |

## Build Instructions

### Requirements

```bash
# System packages
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# NVIDIA packages (from JetPack 6.2)
# - CUDA 12.6
# - DeepStream 7.1
# - libnvbufsurface
# - libnvbufsurftransform
```

### Compilation

```bash
cd ~/ds_pipeline/my_steach

# Build
make clean
make

# Install (copies to ~/.local/share/gstreamer-1.0/plugins/)
make install

# Verify
gst-inspect-1.0 nvdsstitch
```

### Makefile Targets

| Target | Purpose |
|--------|---------|
| `all` | Build libnvdsstitch.so |
| `clean` | Remove build artifacts |
| `install` | Install to local GStreamer plugin path |
| `uninstall` | Remove from plugin path |
| `test` | Run test pipeline with pattern generators |
| `profile` | Run with Nsight Systems profiling |
| `help` | Show available targets |

### Build Flags

```makefile
CXXFLAGS = -fPIC -Wall -Wextra -O3 -std=c++14
NVCCFLAGS = -m64 -Xcompiler -fPIC -O3 -use_fast_math
            -gencode arch=compute_87,code=sm_87
            --expt-relaxed-constexpr
```

## Troubleshooting

### Common Issues

**1. Plugin not found**
```bash
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/home/user/ds_pipeline/my_steach
gst-inspect-1.0 nvdsstitch
```

**2. LUT maps not loading**
```
ERROR: Could not find warp_maps/lut_left_x.bin

Solution: Ensure warp_maps/ directory exists in plugin directory
Check file permissions: chmod 644 warp_maps/*.bin
```

**3. Color mismatch between cameras**
```
Increase COLOR_UPDATE_INTERVAL to allow more samples
Adjust COLOR_SMOOTHING_FACTOR (lower = smoother, higher = faster adaptation)
Check camera calibration (white balance, exposure)
```

**4. Performance issues**
```
Monitor GPU usage: tegrastats
Check for memory bandwidth saturation
Reduce panorama size or disable color correction
```

### Debug Logging

Enable verbose logging:
```bash
export GST_DEBUG=nvdsstitch:5
gst-launch-1.0 ... 2>&1 | tee stitch.log
```

## Technical Notes

### Memory Management

- **Fixed Buffer Pool**: 8 pre-allocated output buffers (no dynamic allocation)
- **EGL Mapping**: Input buffers mapped via EGL for CUDA access (Jetson only)
- **Reference Counting**: Proper GstBuffer refcounting prevents leaks
- **Cleanup**: Automatic resource cleanup on pipeline destruction

### Thread Safety

- CUDA operations synchronized via cudaStreamSynchronize()
- Color correction uses atomic operations for thread-safe accumulation
- No explicit mutex locks required (single-threaded transform)

### Limitations

1. **Fixed Camera Count**: 2 cameras only (hardcoded)
2. **LUT Resolution**: Must match panorama size (pre-generated)
3. **Calibration Required**: Pre-calibrated cameras for LUT generation
4. **Memory Overhead**: 234 MB for LUT maps (GPU resident)

## Source Code Structure

```
my_steach/
├── src/
│   ├── gstnvdsstitch.cpp (1,427 lines)
│   │   └── GStreamer plugin interface, property management
│   ├── gstnvdsstitch.h (122 lines)
│   │   └── Plugin data structures
│   ├── cuda_stitch_kernel.cu (765 lines)
│   │   ├── panorama_lut_kernel (stitching)
│   │   └── analyze_overlap_colors_kernel (color correction)
│   ├── cuda_stitch_kernel.h (107 lines)
│   ├── gstnvdsstitch_allocator.cpp (595 lines)
│   │   └── Custom GPU memory allocator with EGL support
│   ├── gstnvdsstitch_allocator.h (68 lines)
│   ├── nvdsstitch_config.h (92 lines)
│   │   └── Compile-time configuration constants
│   └── gstnvdsbufferpool.h (66 lines)
│       └── Buffer pool utilities
├── warp_maps/ (not checked in)
│   ├── lut_left_x.bin
│   ├── lut_left_y.bin
│   ├── lut_right_x.bin
│   ├── lut_right_y.bin
│   ├── weight_left.bin
│   └── weight_right.bin
├── Makefile (172 lines)
├── panorama_stream.py (160 lines)
│   └── Example: File-based stitching
├── panorama_cameras_realtime.py (100 lines)
│   └── Example: Live camera stitching
└── test_fps.py (100 lines)
    └── FPS measurement utility
```

## Future Enhancements

- [ ] Dynamic LUT generation (eliminate pre-computed maps)
- [ ] Support for >2 cameras (N-way stitching)
- [ ] GPU-based LUT interpolation for runtime panorama resizing
- [ ] HDR tone mapping in overlap zones
- [ ] Vignetting correction per camera
- [ ] Real-time geometry calibration refinement

## References

- **DeepStream Custom Plugin Guide**: https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_sample_custom_gstream.html
- **GStreamer Plugin Writing Guide**: https://gstreamer.freedesktop.org/documentation/plugin-development/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**Maintainer**: See main repository
**Last Updated**: 2025-11-16
