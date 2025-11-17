# GPU/CPU Optimization Plan - DeepStream Pipeline
**Date**: 2025-11-17
**Platform**: NVIDIA Jetson Orin NX 16GB
**Current Status**: GPU 94-99%, CPU 45-60%, RAM 13.2G/15.3G, Micro-freezes in playback

---

## Executive Summary

The pipeline is experiencing severe performance bottlenecks due to:
1. **NVMM→CPU memory path break** causing unnecessary GPU-CPU copies
2. **Deep buffer copies** consuming ~43MB per frame
3. **Heavy Python processing** in GStreamer probe callbacks
4. **O(n) searches** through 210-frame buffer on every frame

**Critical Finding**: GPU is saturated (94-99%) but micro-freezes are caused by **CPU bottlenecks competing for unified memory bandwidth** on Jetson's shared 102 GB/s LPDDR5.

---

## Root Cause Analysis

### 1. NVMM Pipeline Break (CRITICAL - Highest Impact)

**Location**: `new_week/pipeline/pipeline_builder.py:228-230`

```python
nvvideoconvert name=display_convert compute-hw=1 !
capsfilter caps="video/x-raw,format=RGB" !
appsink name=display_sink
```

**Problem**:
- Converts NVMM (GPU memory) → RGB (CPU memory)
- Every frame (5700×1900 RGBA = ~43MB) copied from GPU to CPU
- Violates DeepStream zero-copy architecture
- At 30 FPS: **1.29 GB/s bandwidth consumed** (1.26% of 102 GB/s total)

**Evidence from Documentation**:
> "DeepStream adds zero-memory copy between plugins, enabling state-of-the-art performance"
> "Using memory:NVMM ensures video frames allocated in NVIDIA Memory, avoiding unnecessary data copying"

**Impact**:
- Competes with inference for memory bandwidth
- Forces subsequent deep copies in BufferManager
- Prevents true zero-copy architecture

---

### 2. BufferManager Deep Copies (CRITICAL)

**Location**: `new_week/pipeline/buffer_manager.py:147`

```python
buffer_copy = buffer.copy_deep() if hasattr(buffer, 'copy_deep') else buffer.copy()
```

**Problem**:
- Deep copies every frame received from appsink
- 210 frames × 43MB = **9.03 GB total buffer memory**
- Each copy touches all pixels, monopolizing CPU cores
- Copies already-CPU memory (due to NVMM break above)

**From CODEX Report**:
> "Deep-copying every frame inside the appsink callback contradicts guidance. Forces Cortex-A78AE cluster to touch every pixel of 5700×1900 panorama, burning ~43 MB bandwidth per frame"

**Impact**:
- Excessive RAM usage (13.2G/15.3G)
- CPU saturation (45-60% on all 8 cores)
- Memory bandwidth contention with GPU

---

### 3. O(n) Buffer Searches (HIGH Impact)

**Location**: `new_week/pipeline/buffer_manager.py:231-234`

```python
for frame in self.frame_buffer:  # O(n) search
    if frame['timestamp'] >= self.current_playback_time:
        frame_to_send = frame
        break
```

**Problem**:
- Linear search through 210 frames @ 30 FPS = **6,300 comparisons/sec**
- Runs in GStreamer streaming thread (critical path)
- Audio sync does same thing (lines 289-292)

**Impact**:
- Adds latency jitter (causes micro-freezes)
- CPU cycles wasted on searches instead of inference post-processing

---

### 4. Heavy Python Processing in Probes (HIGH Impact)

**Locations**:
- `new_week/processing/analysis_probe.py:150-537`
- `new_week/rendering/display_probe.py:75-398`
- `new_week/rendering/virtual_camera_probe.py:83-238`

**Problems**:
- Multiclass detection filtering (5 classes)
- Multiple list comprehensions per frame
- NMS in pure Python (`utils/nms.py:24-86`)
- Tensor copy to NumPy (`tensor_processor.py:16-88`)
- History sorting and interpolation
- All executed inside GStreamer probe callbacks

**From DeepStream Documentation**:
> "Avoid time-consuming work inside probes - they sit on the critical path"

**Impact**:
- Violates DeepStream best practices
- CPU spikes during crowded frames (many detections)
- Competes with GPU for unified memory bandwidth

---

### 5. Missing CUDA Memory Type Specification

**Location**: `new_week/pipeline/pipeline_builder.py` (entire pipeline)

**Problem**:
- No `nvbuf-mem-type` specification for elements
- Default behavior may not use CUDA unified memory
- Inconsistent memory types between plugins

**From DeepStream Best Practices**:
> "Use nvbuf-mem-cuda-unified and ensure all elements use same CUDA memory type. Give special attention when using nvtiler, nvstreammux, nvvideoconvert"

---

## DLA Analysis (User Question)

**Question**: Can DLA offload YOLOv11 inference to reduce GPU load?

**Answer**: ❌ **NO - DLA not useful for this use case**

**Evidence from Research**:
1. **DLA optimized for INT8**: 15x faster than FP16 on DLA
2. **FP16 required for ball precision**: User cannot use INT8
3. **YOLOv11 on DLA**: Experimental, complex deployment (GitHub issue #18012)
4. **Recommendation**: GPU TensorRT FP16 is optimal for YOLOv11 on Orin NX

**Conclusion**: User's intuition was correct - stick with GPU inference in FP16 mode.

---

## Optimization Recommendations (Prioritized by Impact)

### P0 - Critical (Highest Impact, Must Fix)

#### 1. Eliminate NVMM→CPU Conversion (Est. -30% CPU, +15% GPU headroom)

**Current**:
```python
nvvideoconvert name=display_convert compute-hw=1 !
capsfilter caps="video/x-raw,format=RGB" !
appsink
```

**Proposed**:
```python
capsfilter caps="video/x-raw(memory:NVMM),format=RGBA" !
appsink name=display_sink
```

**Changes Required**:
- Remove `nvvideoconvert` before appsink
- Keep buffers in NVMM format
- Access via `NvBufSurface` API instead of CPU pointer
- Update BufferManager to handle NVMM buffers

**Benefits**:
- Eliminates 1.29 GB/s CPU→GPU bandwidth
- Enables true zero-copy architecture
- Reduces CPU load significantly

**Risks**:
- Requires rewriting BufferManager buffer handling
- PlaybackPipelineBuilder must accept NVMM input
- Testing required for all display modes

---

#### 2. Replace Deep Copies with NVMM Reference Counting (Est. -40% RAM, -20% CPU)

**Current**:
```python
buffer_copy = buffer.copy_deep()  # Full 43MB copy
self.frame_buffer.append({'timestamp': ts, 'buffer': buffer_copy, ...})
```

**Proposed**:
```python
# Keep reference to original NVMM buffer, increment ref count
gst_buffer_ref(buffer)  # GStreamer built-in ref counting
self.frame_buffer.append({'timestamp': ts, 'buffer': buffer, ...})
```

**Changes Required**:
- Use GStreamer ref counting instead of deep copy
- Ensure proper `gst_buffer_unref()` when removing old frames
- Lock NVMM surfaces during access

**Benefits**:
- Reduces RAM from 13.2G to ~4-5G
- Eliminates CPU copy overhead
- Maintains buffer duration without memory explosion

**Risks**:
- Must manage reference counts carefully (potential memory leak)
- Need to ensure upstream doesn't reuse buffer while in queue

---

#### 3. Use Ring Buffer with Binary Search (Est. -5% CPU, eliminates jitter)

**Current**:
```python
for frame in self.frame_buffer:  # O(n)
    if frame['timestamp'] >= self.current_playback_time:
        return frame
```

**Proposed**:
```python
import bisect

# Maintain sorted timestamp index
idx = bisect.bisect_left(self.timestamp_index, self.current_playback_time)
if idx < len(self.frame_buffer):
    return self.frame_buffer[idx]
```

**Changes Required**:
- Maintain sorted timestamp index alongside buffer
- Use `bisect` module for O(log n) search
- Update index when adding/removing frames

**Benefits**:
- O(log n) vs O(n): 210 comparisons → 8 comparisons
- Eliminates search-induced latency jitter
- Minimal code change

**Risks**:
- Low risk - standard Python library usage

---

### P1 - High Priority (Significant Impact)

#### 4. Move Heavy Processing Out of Probes

**Strategy**: Offload to separate thread with queue

**Current**:
```python
def analysis_probe(pad, info, user_data):
    # Heavy processing directly in probe
    detections = process_all_tensors()  # CPU-heavy
    apply_filters()  # CPU-heavy
    nms()  # CPU-heavy
    return Gst.PadProbeReturn.OK
```

**Proposed**:
```python
def analysis_probe(pad, info, user_data):
    # Lightweight: just extract metadata and queue
    meta = extract_metadata(info)
    processing_queue.put(meta)
    return Gst.PadProbeReturn.OK

def processing_thread():
    while running:
        meta = processing_queue.get()
        # Heavy processing here (off critical path)
        detections = process_all_tensors()
        apply_filters()
        nms()
```

**Benefits**:
- Probe returns immediately (no pipeline blocking)
- Heavy work on separate CPU cores
- Better CPU utilization

**Risks**:
- Adds threading complexity
- Need queue size management
- Potential delay in detection availability

---

#### 5. Use DeepStream Native NMS

**Current**: Pure Python NMS (`utils/nms.py`)

**Proposed**: Use `nvdsinfer` built-in clustering

**Config Change** (`config_infer.txt`):
```ini
# Enable native NMS clustering
cluster-mode=2  # DBSCAN clustering
# Or cluster-mode=4 for hybrid clustering
```

**Benefits**:
- GPU-accelerated or optimized C++ implementation
- Removes Python NMS overhead
- Follows DeepStream best practices

**Risks**:
- May need to tune clustering parameters
- Different output format (need adapter)

---

#### 6. Batch Metadata Processing

**Current**: Per-frame processing in probes

**Proposed**: Collect batch metadata, process together

```python
# Accumulate metadata from multiple frames
metadata_batch = []

def analysis_probe(pad, info, user_data):
    metadata_batch.append(extract_meta(info))

    # Process in batches of 5 frames
    if len(metadata_batch) >= 5:
        process_batch(metadata_batch)
        metadata_batch.clear()

    return Gst.PadProbeReturn.OK
```

**Benefits**:
- Better CPU cache utilization
- Vectorizable operations (NumPy)
- Reduced overhead per frame

**Risks**:
- Increases latency by batch size
- More complex state management

---

### P2 - Medium Priority (Optimization)

#### 7. Specify CUDA Unified Memory Type

**Add to all nvvideoconvert elements**:
```python
nvvideoconvert nvbuf-mem-type=3  # 3 = CUDA unified memory
```

**Add to nvstreammux**:
```python
nvstreammux nvbuf-memory-type=3
```

**Benefits**:
- Consistent memory allocation
- Better unified memory performance
- Follows DeepStream best practices

---

#### 8. Optimize History Manager

**Current**: Sorts entire history on every lookup

**Proposed**:
- Maintain pre-sorted structure (SortedDict)
- Lazy interpolation (compute only when needed)
- LRU cache for repeated timestamp queries

---

#### 9. Use Numba JIT for Hot Paths

**Apply to**:
- NMS calculations
- Distance filters
- Trajectory interpolation

```python
from numba import jit

@jit(nopython=True)
def calculate_iou_fast(box1, box2):
    # Compiled to machine code
    ...
```

---

## Implementation Plan

### Phase 1 (Week 1-2): Critical Path - NVMM Zero-Copy

**Goal**: Eliminate GPU↔CPU copies

1. **Day 1-3**: Refactor BufferManager for NVMM
   - Create NVMM buffer wrapper class
   - Implement NvBufSurface API access
   - Add reference counting

2. **Day 4-5**: Update PipelineBuilder
   - Remove nvvideoconvert before appsink
   - Add NVMM caps filter
   - Specify nvbuf-mem-type=3

3. **Day 6-7**: Update PlaybackPipelineBuilder
   - Accept NVMM input from appsrc
   - Ensure NVMM flow to display/encode

4. **Testing**: Verify functionality across all modes
   - File sources
   - Camera sources
   - Panorama mode
   - Virtual camera mode
   - Stream mode
   - Recording mode

**Expected Result**: -30% CPU, -40% RAM, +15% GPU headroom

---

### Phase 2 (Week 3): Performance Tuning

1. **Binary Search**: Replace O(n) with O(log n) searches
2. **Native NMS**: Enable DeepStream clustering
3. **Memory Type**: Add nvbuf-mem-type specifications

**Expected Result**: -10% CPU, smoother playback

---

### Phase 3 (Week 4): Advanced Optimizations

1. **Threading**: Move heavy processing out of probes
2. **Batching**: Process metadata in batches
3. **Numba**: JIT compile hot paths

**Expected Result**: -15% CPU, higher throughput potential

---

## Risk Assessment

### High Risk

**P0 Optimizations**: NVMM zero-copy refactor
- **Risk**: Breaking pipeline functionality
- **Mitigation**:
  - Create feature branch
  - Extensive testing before merge
  - Keep backup of working version
  - Test on file sources first, then cameras

### Medium Risk

**Threading**: Moving processing off probes
- **Risk**: Race conditions, delayed detections
- **Mitigation**:
  - Use thread-safe queues
  - Add timeout mechanisms
  - Monitor queue depth

### Low Risk

**Binary Search, Native NMS, Numba**
- Standard optimizations
- Easy to revert if issues

---

## Expected Performance Impact

### Current State
- GPU: 94-99%
- CPU: 45-60% (all cores)
- RAM: 13.2G / 15.3G (86%)
- Micro-freezes: Present

### After Phase 1 (NVMM Zero-Copy)
- GPU: 85-90% (inference still dominant)
- CPU: 25-35% (eliminated copies)
- RAM: 4-5G / 15.3G (30%)
- Micro-freezes: Eliminated

### After Phase 2+3 (Full Optimization)
- GPU: 85-90%
- CPU: 15-25%
- RAM: 4-5G / 15.3G
- Headroom: 10-15GB RAM, 10-15% GPU, 75% CPU available

---

## Monitoring & Validation

### Metrics to Track

1. **Memory Bandwidth Usage**
   ```bash
   sudo tegrastats
   ```
   - Monitor EMC (memory controller) utilization

2. **Pipeline Latency**
   - Add timestamps at key points
   - Measure end-to-end delay

3. **Frame Drops**
   - Monitor appsink drops
   - Check inference queue depth

4. **CPU Per-Core Usage**
   ```bash
   htop
   ```

5. **GPU Utilization**
   ```bash
   sudo jetson_clocks --show
   ```

### Success Criteria

- ✅ No micro-freezes in playback
- ✅ CPU usage < 30% average
- ✅ RAM usage < 6GB
- ✅ GPU usage 80-90% (inference-limited)
- ✅ Stable 30 FPS in all modes
- ✅ Latency < 150ms end-to-end

---

## References

1. **DeepStream Best Practices**: Zero-copy architecture via memory:NVMM
2. **CODEX Report**: `docs/reports/CODEX_report.md`
3. **DeepStream 7.1 Docs**: `/ds_doc/7.1/`
4. **Jetson Orin NX Architecture**: `nvidia_jetson_orin_nx_16GB_super_arch.pdf`
5. **DLA Research**: DLA optimized for INT8, not FP16 - not useful for YOLOv11

---

## Next Steps

1. ✅ Review this plan with team
2. Create feature branch: `optimize/nvmm-zero-copy`
3. Start Phase 1 implementation
4. Daily performance monitoring
5. Update TODO.md with implementation tasks

---

**Author**: Claude Code Analysis
**Status**: Ready for Implementation
**Priority**: P0 - Critical Performance Issues
