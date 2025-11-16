# nvtilebatcher Plugin - Panorama Tile Extraction for Inference

## Overview

**nvtilebatcher** is a custom GStreamer plugin for NVIDIA DeepStream that converts panoramic video frames into batched tiles for object detection. It extracts 6 non-overlapping 1024×1024 tiles from a panoramic frame for parallel inference processing.

**Version**: 1.0
**Platform**: NVIDIA Jetson Orin NX (DeepStream 7.1, JetPack 6.2)
**Use Case**: Football field monitoring with multi-region object detection

## Key Features

- Extracts 6 tiles (1024×1024) from panorama in parallel
- GPU-accelerated CUDA tile extraction
- Zero-copy NVMM memory operations
- DeepStream metadata integration
- Fixed buffer pool for stable memory usage
- EGL resource caching for performance

## Plugin Architecture

### Pipeline Position
```
nvdsstitch → nvtilebatcher → nvinfer (YOLO11)
```

### Data Flow
```
Panorama Frame (5700×1900 NVMM)
           ↓
    [Calculate Positions]
           ↓
    [CUDA Tile Extraction]  → 6 tiles in parallel
           ↓
Batch Buffer (6×1024×1024 NVMM)
           ↓
[DeepStream Batch Metadata] → 6 NvDsFrameMeta
           ↓
    → nvinfer (batch=6)
```

## Technical Specifications

### Input Requirements
- **Format**: RGBA
- **Resolution**: Configurable (e.g., 5700×1900 or 6528×1632)
- **Memory**: NVMM (NVIDIA Memory Manager)
- **Source**: Single panoramic frame

### Output Specifications
- **Format**: RGBA
- **Tile Resolution**: 1024×1024 pixels (fixed)
- **Tile Count**: 6 (fixed)
- **Batch Size**: 6
- **Memory**: NVMM (GPU memory)
- **Memory Type**: `NVBUF_MEM_SURFACE_ARRAY` (Jetson-specific)

## Configuration Properties

| Property | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `gpu-id` | uint | 0 | No | GPU device ID (0-7) |
| `silent` | boolean | FALSE | No | Disable info logging |
| `panorama-width` | uint | - | **YES** | Input panorama width (e.g., 5700) |
| `panorama-height` | uint | - | **YES** | Input panorama height (e.g., 1900) |
| `tile-offset-y` | uint | 434 | No | Vertical offset for tile extraction |

### Example Usage
```bash
nvtilebatcher \
    panorama-width=5700 \
    panorama-height=1900 \
    tile-offset-y=434 \
    gpu-id=0
```

**IMPORTANT**: `panorama-width` and `panorama-height` are **mandatory**. Plugin will error if not set.

## Tile Extraction Logic

### Tile Position Calculation

**X Positions** (horizontal layout):
```
Tile 0: x=192   (192px left margin)
Tile 1: x=1216  (192 + 1024)
Tile 2: x=2240  (192 + 2×1024)
Tile 3: x=3264  (192 + 3×1024)
Tile 4: x=4288  (192 + 4×1024)
Tile 5: x=5312  (192 + 5×1024)
```

**Y Position**: `tile_offset_y` (default: 434px from top)

**Result**: 6 tiles arranged horizontally across the panorama
```
Tile 0: (192, 434) → (1216, 1458)
Tile 1: (1216, 434) → (2240, 1458)
...
Tile 5: (5312, 434) → (6336, 1458)
```

### CUDA Kernel

**Function**: `extract_tiles_kernel_multi()`

**Grid Configuration**:
```cpp
dim3 block(32, 32, 1);  // 1024 threads per block
dim3 grid(
    32,  // X blocks (1024/32)
    32,  // Y blocks (1024/32)
    6    // Z blocks (one per tile)
);

Total: 32 × 32 × 6 = 6,144 blocks
Total threads: 6,291,456 threads in parallel
```

**Algorithm**:
```cpp
// Each thread processes one pixel
tile_id = blockIdx.z;  // Which tile (0-5)
tile_x = blockIdx.x * blockDim.x + threadIdx.x;
tile_y = blockIdx.y * blockDim.y + threadIdx.y;

src_x = d_tile_positions[tile_id].x + tile_x;
src_y = d_tile_positions[tile_id].y + tile_y;

// Copy pixel (or black if out of bounds)
if (src_x < pano_width && src_y < pano_height)
    dst_pixel = src_pixel;
else
    dst_pixel = 0xFF000000;  // Black
```

**Optimizations**:
- `__restrict__` pointers for aliasing optimization
- Coalesced 4-byte RGBA reads
- Constant memory for tile positions
- Out-of-bounds handling (black fill)

## Processing Pipeline

### Phase 1: Initialization
1. Create CUDA stream (non-blocking)
2. Create CUDA event for synchronization
3. Calculate tile positions based on panorama dimensions
4. Copy positions to GPU constant memory
5. Initialize EGL cache (hash table)

### Phase 2: Buffer Pool Setup
1. Create GStreamer buffer pool with custom allocator
2. Allocate 4 output buffers (each containing 6 tiles)
3. Map EGL images for all tiles in all buffers
4. Register all EGL images with CUDA (24 registrations)
5. Store CUDA resource handles

### Phase 3: Per-Frame Processing
1. Validate panorama size matches properties
2. Map input panorama buffer
3. Check EGL cache for CUDA resource (register if new)
4. Get output buffer from fixed pool (round-robin with mutex)
5. Set tile output pointers in CUDA constant memory
6. Launch CUDA extraction kernel
7. Wait for CUDA event completion
8. Create DeepStream batch metadata (6 frame metas)
9. Push buffer downstream to nvinfer

## DeepStream Metadata

For each tile, the plugin creates:
```cpp
NvDsFrameMeta {
    source_id: from input
    batch_id: 0-5 (tile index)
    surface_index: 0-5
    source_frame_width: 1024
    source_frame_height: 1024
    buf_pts: copied from input
    ntp_timestamp: copied from input

    // User metadata: TileRegionInfo
    user_meta {
        tile_id: 0-5
        panorama_x: tile X position in panorama
        panorama_y: tile Y position in panorama
        tile_width: 1024
        tile_height: 1024
    }
}
```

This metadata allows downstream elements (e.g., detection post-processing) to map detections back to panorama coordinates:
```python
global_x = tile_region.panorama_x + detection_x
global_y = tile_region.panorama_y + detection_y
```

## Memory Architecture

### NvBufSurface Structure
```cpp
NvBufSurface {
    gpuId: 0
    batchSize: 6
    numFilled: 6
    memType: NVBUF_MEM_SURFACE_ARRAY

    surfaceList[6]: [
        {
            width: 1024
            height: 1024
            colorFormat: NVBUF_COLOR_FORMAT_RGBA
            layout: NVBUF_LAYOUT_PITCH
            planeParams.pitch[0]: 4096 (256-byte aligned)
            dataSize: 4,194,304 bytes (1024×1024×4)
            mappedAddr.eglImage: <EGL handle>
        },
        ... (5 more identical structures)
    ]
}
```

### Fixed Buffer Pool
```cpp
#define FIXED_OUTPUT_POOL_SIZE 4

// Pre-allocated buffers (round-robin reuse)
GstBuffer* buffers[4];
NvBufSurface* surfaces[4];
bool registered[4];
```

**Benefits**:
- Zero allocation during runtime
- Stable memory usage
- No fragmentation
- Pre-registered CUDA resources

### EGL Resource Caching
```cpp
std::unordered_map<void*, EGLCacheEntry> g_egl_cache;

// First access: Register
cuGraphicsEGLRegisterImage(&cuda_resource, egl_image, ...);

// Subsequent access: Cache hit (instant)
return cached_cuda_ptr;
```

**Performance**: Saves ~1-2ms per frame after warmup

## Performance

### GPU Memory Usage
```
Tile position LUTs: ~100 bytes (constant memory)
Input panorama ref: 0 bytes (EGL mapped, not copied)
Output buffers: 4 × 6 × 4MB = 96 MB
EGL cache: ~1 KB
Total: ~96 MB
```

### Processing Time
```
Parameter validation: ~1 μs
Buffer mapping: 50-100 μs (EGL, cached)
CUDA kernel: 200-500 μs (parallel extraction)
Synchronization: 100-200 μs
Metadata creation: 50-100 μs
Total: ~500-1000 μs per frame

Theoretical: 1000-2000 FPS
Actual: Limited by upstream/downstream (30-60 FPS)
```

### Optimization Techniques
- **Constant Memory**: Tile positions cached in GPU
- **Fixed Pool**: No allocation overhead
- **EGL Caching**: No re-registration overhead
- **Streams**: Non-blocking CUDA execution
- **Coalesced Access**: Memory bandwidth efficiency

## Test Scripts

### test_complete_pipeline.py
Full end-to-end integration test with YOLO inference.

**Pipeline**:
```
filesrc → jpegdec → nvvideoconvert → nvstreammux →
nvdsstitch → nvtilebatcher → nvinfer → fakesink
```

**Tests**:
- Batch structure validation (6 tiles)
- Metadata validation
- Tile dimensions
- YOLO inference on tiles
- Detection counting

### test_performance.py
FPS benchmarking tool.

**Usage**:
```bash
python test_performance.py
```

**Metrics**:
- Total buffers processed
- Processing time
- Average FPS
- Performance rating

### test_simple.py
Basic functionality test.

**Pipeline**:
```
filesrc → jpegdec → nvvideoconvert → nvtilebatcher → fakesink
```

**Checks**:
- Plugin loads correctly
- Buffers flow through
- No errors
- Buffer validation

### test_tilebatcher.py
Comprehensive unit testing.

**Tests**:
1. Batch creation (6 tiles)
2. Batch content validation
3. Direct NvBufSurface access
4. Integration with nvinfer

## Integration Example

Complete pipeline:
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

pipeline = Gst.parse_launch("""
    filesrc location=panorama.jpg ! jpegdec !
    nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA !
    nvtilebatcher panorama-width=5700 panorama-height=1900 tile-offset-y=434 !
    nvinfer config-file-path=config_infer.txt batch-size=6 !
    nvdsosd ! nveglglessink
""")

pipeline.set_state(Gst.State.PLAYING)
```

### Python Property Updates
```python
tilebatcher = pipeline.get_by_name("tilebatcher")
tilebatcher.set_property("panorama-width", 5700)
tilebatcher.set_property("panorama-height", 1900)
tilebatcher.set_property("tile-offset-y", 434)
tilebatcher.set_property("gpu-id", 0)
```

## Troubleshooting

### Plugin Fails to Start
- **Error**: "panorama-width and panorama-height must be set"
- **Solution**: Set both properties via command line or code

### Black Tiles
- Check panorama dimensions are correct
- Verify tile_offset_y is within panorama height
- Ensure input is RGBA NVMM format

### Low FPS
- Check GPU memory availability
- Verify EGL caching is working (check logs)
- Reduce panorama resolution if possible

### Metadata Not Propagating
- Ensure DeepStream 7.1+ (metadata API changed)
- Check nvinfer batch-size matches tile count (6)
- Verify metadata copy in downstream probes

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
cd my_tile_batcher/src
make                # Build libnvtilebatcher.so
make install        # Install to system plugins directory
```

### Makefile Targets
```bash
make                # Build plugin
make install        # Install plugin
make clean          # Clean build artifacts
```

## File Locations

- **Source Code**: `/home/user/ds_pipeline/my_tile_batcher/src/`
- **Plugin Binary**: `/home/user/ds_pipeline/my_tile_batcher/src/libnvtilebatcher.so`
- **Test Scripts**: `/home/user/ds_pipeline/my_tile_batcher/test_*.py`
- **Header Files**: `/home/user/ds_pipeline/my_tile_batcher/src/*.h`
- **CUDA Kernel**: `/home/user/ds_pipeline/my_tile_batcher/src/cuda_tile_extractor.cu`

## License

Custom NVIDIA DeepStream plugin for research and development purposes.

## References

- NVIDIA DeepStream SDK: https://developer.nvidia.com/deepstream-sdk
- GStreamer Custom Allocator: https://gstreamer.freedesktop.org/documentation/gstreamer/gstallocator.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- DeepStream Metadata: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html
