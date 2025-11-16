# MASR Multi-Class Inference Pipeline

**See main documentation**: `/home/user/ds_pipeline/CLAUDE.MD` Section 5

## Quick Reference

**Main Script**: `version_masr_multiclass.py` (2,500+ lines Python)
**Model**: YOLOv11n/s INT8 quantized (TensorRT)
**Classes**: 5 (ball, player, staff, side_referee, main_referee)
**Performance**: 30 FPS with 6-tile batch

## Detection Classes

| Class ID | Name | Threshold | Priority | Color (Panorama) |
|----------|------|-----------|----------|------------------|
| 0 | ball | 0.25 | Highest | Red (border=3) |
| 1 | player | 0.40 | High | Green (border=2) |
| 2 | staff | 0.40 | Low | Disabled (save slots) |
| 3 | side_referee | 0.40 | Low | Disabled |
| 4 | main_referee | 0.40 | Low | Disabled |

## Model Configuration

**File**: `config_infer.txt`
```ini
[property]
model-engine-file=yolo11n_mixed_finetune_v9.engine
batch-size=6
network-mode=2  # INT8
num-detected-classes=1  # Legacy, actual: 5 classes
output-tensor-meta=1
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.25
topk=100
nms-iou-threshold=0.45
```

## BallDetectionHistory System

**3-tier temporal buffer** (10 seconds):

1. **Raw Future History**: Incoming detections from analysis
2. **Processed Future History**: Cleaned + interpolated
3. **Confirmed History**: Already displayed (7s delayed)

**Key Features**:
- Duplicate filtering (≤2px threshold)
- Parabolic interpolation for flight (gap >1s)
- Outlier removal with permanent blacklist
- Backward interpolation for smooth recovery

## Virtual Camera Control

### Ball Tracking
```python
# Speed-based auto-zoom
speed_px_per_sec = calculate_speed(current_pos, last_pos, dt)
if 300 < speed < 1200:
    zoom_factor = map_range(speed, 300, 1200, 1.0, 3.0)
    target_fov = base_fov / zoom_factor

# Smooth movement
smooth_factor = 0.3  # 30% new position per frame
vcam.set_property("yaw", smooth_yaw)
vcam.set_property("fov", smooth_fov)
```

### Ball Loss Recovery
```python
if no_detection_for_frames > 6:  # 0.2 seconds
    ball_lost = True
    fov_expand_rate = 2.0  # degrees/second
    max_search_fov = 90.0
    
    # Fallback to players center-of-mass
    players_center = calculate_center_of_mass(players_history)
    target_position = ema_smooth(players_center, alpha=0.18)
```

## Pipeline Modes

### 1. Panorama Mode
```bash
python3 version_masr_multiclass.py --display-mode panorama
```
- Full 5700×1900 view
- Max 16 objects rendered (nvdsosd limit)

### 2. Virtual Camera Mode
```bash
python3 version_masr_multiclass.py --display-mode virtualcam --auto-zoom
```
- Intelligent ball tracking
- 1920×1080 output

### 3. Streaming Mode
```bash
python3 version_masr_multiclass.py \
    --display-mode stream \
    --stream-url rtmp://live.twitch.tv/app \
    --stream-key YOUR_KEY
```

### 4. Recording Mode
```bash
python3 version_masr_multiclass.py \
    --display-mode record \
    --output-file game.mp4 \
    --buffer-duration 7.0
```

## Memory Architecture

**7-Second Buffer** @ 30 FPS:
- Frame count: 210 frames
- Per-frame size: ~15 MB (H.264 encoded)
- Total RAM: ~3 GB
- Audio buffer: ~700 chunks (if available)

**Synchronization**:
- Analysis branch: Real-time (timestamp T)
- Display branch: 7s delayed (timestamp T-7)
- Metadata synced via `update_display_timestamp()`

## Performance Tuning

### Confidence Threshold
```python
confidence_threshold = 0.35  # Lower = more detections, higher recall
```

### Analysis Skip Interval
```python
analysis_skip_interval = 5  # Process every 5th frame (saves GPU)
```

### Field Mask Filtering
```python
field_mask = FieldMaskBinary('field_mask.png')
if not field_mask.is_inside_field(x, y):
    reject_detection()  # O(1) lookup
```

## Command-Line Arguments

```bash
python3 version_masr_multiclass.py \
    --source-type cameras \          # or "files"
    --video1 0 \                      # Left camera ID or file path
    --video2 1 \                      # Right camera ID or file path
    --display-mode virtualcam \       # panorama|virtualcam|stream|record
    --buffer-duration 7.0 \           # Seconds
    --enable-analysis \               # Enable inference
    --analysis-skip-interval 5 \      # Process every Nth frame
    --confidence-threshold 0.35 \     # Detection threshold
    --auto-zoom \                     # Enable speed-based zoom
    --output-file game.mp4 \          # For record mode
    --stream-url rtmp://... \         # For stream mode
    --stream-key YOUR_KEY             # RTMP stream key
```

For complete documentation see: `CLAUDE.MD`
