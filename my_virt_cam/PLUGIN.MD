# MY_VIRT_CAM - Virtual Camera Plugin

**See main documentation**: `/home/user/ds_pipeline/CLAUDE.MD` Section 2

## Quick Reference

**Plugin**: `nvvirtualcam`
**Type**: GstBaseTransform  
**Purpose**: GPU-accelerated panorama-to-perspective projection with intelligent tracking
**Source**: 2,100+ lines C++/CUDA
**Location**: `/home/user/ds_pipeline/my_virt_cam/src/`

## Key Specifications

- **Input**: 6528×1632 RGBA equirectangular panorama
- **Output**: 1920×1080 RGBA perspective view
- **Performance**: 47.90 FPS, 20.88ms latency
- **GPU Memory**: ~50 MB (LUT cache + buffers)
- **Optimization**: 80-90% GPU occupancy

## Control Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| yaw | -90° to +90° | 0° | Horizontal pan |
| pitch | -32° to +22° | 0° | Vertical tilt |
| roll | -28° to +28° | 0° | Image rotation |
| fov | 55° to 68° | 60° | Zoom level |
| smooth-factor | 0.0 to 1.0 | 0.3 | Movement interpolation |

## CUDA Kernel Details

**File**: `virtual_camera_kernel.cu`
**Block size**: 16×16 threads (256/block)
**Grid**: 8,160 blocks for 1920×1080
**Algorithm**: 3-stage pipeline (ray generation → LUT → remapping)

## Usage Example

```bash
gst-launch-1.0 videotestsrc ! nvvirtualcam yaw=30 pitch=10 fov=65 ! nveglglessink
```

For complete documentation see: `CLAUDE.MD`
