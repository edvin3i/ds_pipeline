# MY_TILE_BATCHER - Tile Extraction Plugin

**See main documentation**: `/home/user/ds_pipeline/CLAUDE.MD` Section 3

## Quick Reference

**Plugin**: `nvtilebatcher`
**Type**: GstBaseTransform
**Purpose**: Extract 6×1024×1024 tiles from panorama for batch inference
**Source**: 1,733 lines C++/CUDA
**Location**: `/home/user/ds_pipeline/my_tile_batcher/src/`

## Key Specifications

- **Input**: 6528×1632 RGBA panorama (or configurable)
- **Output**: NvDsBatchMeta with 6 tile surfaces (1024×1024 each)
- **Performance**: ≥30 FPS on Jetson Orin NX
- **GPU Memory**: ~25 MB per batch

## Tile Layout

```
Tile 0: X=192,   Y=434, Size=1024×1024
Tile 1: X=1216,  Y=434, Size=1024×1024
Tile 2: X=2240,  Y=434, Size=1024×1024
Tile 3: X=3264,  Y=434, Size=1024×1024
Tile 4: X=4288,  Y=434, Size=1024×1024
Tile 5: X=5312,  Y=434, Size=1024×1024

Vertical offset Y=434 calculated from field_mask.png
Horizontal offset X=192 for centering
```

## Properties

```bash
nvtilebatcher \
    gpu-id=0 \
    panorama-width=6528 \
    panorama-height=1632 \
    tile-offset-y=434 \
    silent=false
```

## CUDA Kernel

**File**: `cuda_tile_extractor.cu:extract_tiles_kernel_multi`
**Launch**: (32, 32, 6) blocks × (32, 32, 1) threads
**Processing**: 6,291,456 pixels per frame
**Memory**: Coalesced 4-byte RGBA access

## TileRegionInfo Metadata

Each tile carries metadata for panorama coordinate conversion:

```c
struct TileRegionInfo {
    uint tile_id;        // 0-5
    uint panorama_x;     // Source X in panorama
    uint panorama_y;     // Source Y in panorama  
    uint tile_width;     // 1024
    uint tile_height;    // 1024
};
```

For complete documentation see: `CLAUDE.MD`
