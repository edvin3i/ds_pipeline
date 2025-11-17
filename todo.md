# DeepStream Sports Analytics Pipeline - TODO List

**Last Updated**: 2025-11-17
**Status**: Phase 4 - Critical Performance Optimization

---

## 🔥 CRITICAL: GPU/CPU Optimization (Week 47-50, 2025)

**Root Cause Identified**: Pipeline breaks DeepStream zero-copy architecture
**Impact**: GPU 94-99%, CPU 45-60%, RAM 13.2G/15.3G, micro-freezes present
**Reports**: `docs/reports/GPU_CPU_OPTIMIZATION_PLAN.md`, `docs/reports/PIPELINE_MEMORY_FLOW_ANALYSIS.md`

### Performance Analysis Completed ✅

- [x] **Analyze pipeline memory flow** ✅ COMPLETED 2025-11-17
  - Identified 4 critical bottlenecks
  - NVMM→CPU break at appsink (1.29 GB/s waste)
  - Deep buffer copies (9GB RAM waste)
  - O(n) searches (6,300 comparisons/sec)
  - Heavy Python in probes (CPU saturation)
  - **Finding**: DLA not useful for YOLOv11 FP16 (optimized for INT8)

### Phase 1: NVMM Zero-Copy Refactor (P0 - Week 47-48)

**Goal**: Eliminate GPU↔CPU copies, reduce RAM by 9GB, eliminate micro-freezes
**Expected**: -30% CPU, -40% RAM, +15% GPU headroom

- [ ] **⏳ Create feature branch `optimize/nvmm-zero-copy`**
  - Branch from current working state
  - Ensure full backup before major refactor
  - **Priority**: P0 - Foundation for all optimizations

- [ ] **Remove NVMM→CPU conversion in PipelineBuilder**
  - **File**: `new_week/pipeline/pipeline_builder.py:228-230`
  - **Current**: `nvvideoconvert ! capsfilter caps="video/x-raw,format=RGB" ! appsink`
  - **Target**: `capsfilter caps="video/x-raw(memory:NVMM),format=RGBA" ! appsink`
  - **Change**: Remove nvvideoconvert, keep NVMM format
  - **Priority**: P0 - Critical bottleneck
  - **Risk**: Medium - requires BufferManager refactor

- [ ] **Refactor BufferManager for NVMM buffers**
  - **File**: `new_week/pipeline/buffer_manager.py`
  - **Current**: `buffer.copy_deep()` (43MB × 210 = 9GB)
  - **Target**: `gst_buffer_ref(buffer)` (reference counting)
  - **Changes**:
    - Replace deep copy with GStreamer ref counting (line 147)
    - Implement NvBufSurface API for NVMM access
    - Add proper `gst_buffer_unref()` on cleanup (line 392)
    - Lock NVMM surfaces during access
  - **Priority**: P0 - Highest RAM/CPU impact
  - **Risk**: High - careful memory management required

- [ ] **Update PlaybackPipelineBuilder for NVMM input**
  - **File**: `new_week/pipeline/playback_builder.py`
  - **Change**: Accept NVMM buffers from appsrc
  - **Verify**: NVMM flow through virtualcam/osd/encode
  - **Priority**: P0 - Required for zero-copy
  - **Risk**: Medium - ensure format compatibility

- [ ] **Add CUDA unified memory type specification**
  - **Files**: All pipeline builders
  - **Change**: Add `nvbuf-mem-type=3` to nvvideoconvert, nvstreammux
  - **Example**: `nvvideoconvert nvbuf-mem-type=3 ! ...`
  - **Priority**: P0 - Memory consistency
  - **Risk**: Low - configuration only

- [ ] **Testing: NVMM zero-copy validation**
  - Test all display modes (panorama, virtualcam, stream, record)
  - Test file sources (left.mp4 + right.mp4)
  - Test camera sources (sensor-id=0,1)
  - Verify no visual artifacts
  - Monitor RAM usage (<6GB target)
  - Check for memory leaks (long run)
  - **Priority**: P0 - Must not break functionality
  - **Success**: RAM < 6GB, no micro-freezes, 30 FPS stable

### Phase 2: Performance Tuning (P1 - Week 49)

**Goal**: Eliminate search overhead, use native acceleration
**Expected**: -10% CPU, smoother playback, better GPU utilization

- [ ] **Replace O(n) search with binary search**
  - **File**: `new_week/pipeline/buffer_manager.py:231-234`
  - **Current**: `for frame in self.frame_buffer:` (O(n))
  - **Target**: `bisect.bisect_left()` (O(log n))
  - **Changes**:
    - Add sorted timestamp index
    - Use Python bisect module
    - Update index on add/remove
  - **Priority**: P1 - Eliminates jitter
  - **Risk**: Low - standard library

- [ ] **Enable DeepStream native NMS clustering**
  - **File**: `new_week/config_infer.txt`
  - **Current**: Python NMS in probe (nested loops)
  - **Target**: `cluster-mode=2` (DBSCAN GPU-accelerated)
  - **Change**: Add clustering config to nvinfer
  - **Remove**: `new_week/utils/nms.py` usage in probes
  - **Priority**: P1 - GPU acceleration
  - **Risk**: Medium - need to tune parameters

- [ ] **Validate native NMS output quality**
  - Compare detection quality vs Python NMS
  - Tune cluster parameters if needed
  - Verify no false negatives for ball
  - **Priority**: P1 - Quality assurance
  - **Success**: Same or better detection accuracy

- [ ] **Testing: Performance tuning validation**
  - Measure CPU usage (<30% target)
  - Verify smooth playback (no jitter)
  - Check detection quality maintained
  - Monitor GPU utilization (80-90% target)
  - **Priority**: P1 - Performance validation

### Phase 3: Advanced Optimization (P2 - Week 50)

**Goal**: Move heavy processing off critical path
**Expected**: -15% CPU, higher throughput potential

- [ ] **Move heavy processing to background thread**
  - **Files**: `new_week/processing/analysis_probe.py`
  - **Current**: All processing in probe (50-100ms)
  - **Target**: Lightweight probe + background thread
  - **Changes**:
    - Create processing queue (thread-safe)
    - Extract metadata only in probe (<1ms)
    - Heavy work in background thread
    - Thread-safe history updates
  - **Priority**: P2 - Advanced optimization
  - **Risk**: Medium - threading complexity

- [ ] **Implement batch metadata processing**
  - **Strategy**: Accumulate 5 frames, process together
  - **Benefits**: Better CPU cache, vectorizable NumPy ops
  - **Tradeoff**: +5 frame latency (~166ms @ 30fps)
  - **Priority**: P2 - Throughput optimization
  - **Risk**: Low - latency acceptable for analytics

- [ ] **Add Numba JIT compilation for hot paths**
  - **Targets**:
    - `utils/nms.py:24-86` (if still used)
    - `core/trajectory_filter.py:63-210`
    - `core/trajectory_interpolator.py`
  - **Change**: Add `@jit(nopython=True)` decorator
  - **Priority**: P2 - CPU optimization
  - **Risk**: Low - easy to revert

- [ ] **Testing: Advanced optimization validation**
  - Long-running stability (8+ hours)
  - Monitor thread safety (no race conditions)
  - Check queue depth (no unbounded growth)
  - Measure end-to-end latency (<200ms)
  - **Priority**: P2 - Stability validation

### Success Criteria (All Phases)

- [ ] ✅ CPU usage < 30% average
- [ ] ✅ RAM usage < 6GB
- [ ] ✅ GPU usage 80-90% (inference-limited)
- [ ] ✅ No micro-freezes in playback
- [ ] ✅ Stable 30 FPS in all modes
- [ ] ✅ End-to-end latency < 150ms
- [ ] ✅ No visual artifacts
- [ ] ✅ Detection quality maintained
- [ ] ✅ All display modes working
- [ ] ✅ Long-term stability (8+ hours)

---

## Current Sprint: Production Readiness (On Hold - Blocked by Optimization)

### High Priority

#### DeepStream 7.1 Compliance Fixes

**Source**: `docs/reports/DEEPSTREAM_CODE_REVIEW.md`

- [x] **CRITICAL: Fix metadata iteration StopIteration handling** ✅ COMPLETED 2025-11-17
  - **Files**: `new_week/processing/analysis_probe.py:191-261`, `new_week/rendering/display_probe.py:288-302`
  - **Issue**: Missing try/except blocks can cause crashes when metadata list ends unexpectedly
  - **Fix**: Wrapped all `cast()` and `next()` calls in try/except StopIteration
  - **Priority**: P0 - Can cause production crashes
  - **Implemented**: Comprehensive StopIteration handling in all metadata list iterations

- [x] **CRITICAL: Fix nvinfer configuration** ✅ COMPLETED 2025-11-17
  - **File**: `new_week/config_infer.txt`
  - **Issue 1**: `num-detected-classes=1` should be `5` (ball, player, staff, side_ref, main_ref)
  - **Issue 2**: Missing `[class-attrs-4]` section for main_referee class
  - **Fix**: Updated config file with correct class count and added missing section
  - **Priority**: P0 - Incorrect detections
  - **Implemented**: Line 9: num-detected-classes=5, Lines 46-50: [class-attrs-4] section added

- [x] **CRITICAL: Add user metadata validation** ✅ COMPLETED 2025-11-17
  - **File**: `new_week/processing/analysis_probe.py:223-231`
  - **Issue**: No validation that user_meta_data is properly initialized
  - **Fix**: Added null checks after casting NvDsUserMeta
  - **Priority**: P0 - Potential memory leaks
  - **Implemented**: Lines 225-231: Validation with warning log and safe skip

- [ ] **IMPORTANT: Document probe return values**
  - **Files**: All probe handlers
  - **Issue**: Always return `Gst.PadProbeReturn.OK` without documenting why
  - **Fix**: Add docstring explaining return value choice
  - **Priority**: P1 - Documentation

- [ ] **IMPORTANT: Add error handling to tensor extraction**
  - **File**: `new_week/processing/tensor_processor.py:120-148`
  - **Issue**: Generic exception handling doesn't distinguish error types
  - **Fix**: Separate TypeError, ValueError, and generic Exception handling
  - **Priority**: P1 - Better debugging

---

### Medium Priority (Deferred - See GPU/CPU Optimization Above)

#### Legacy Performance Optimizations

**Note**: These tasks have been superseded by comprehensive optimization plan above.
**Source**: `docs/reports/CODEX_report.md` (analysis basis for new plan)

- [x] **Optimize BufferManager deep copies** - ✅ SUPERSEDED by Phase 1
  - Now tracked in: Phase 1 "Refactor BufferManager for NVMM buffers"
  - Enhanced approach: NVMM zero-copy instead of shallow copy

- [x] **Optimize playback timestamp scans** - ✅ SUPERSEDED by Phase 2
  - Now tracked in: Phase 2 "Replace O(n) search with binary search"
  - Same fix, integrated into optimization plan

- [x] **Replace Python NMS with native implementation** - ✅ SUPERSEDED by Phase 2
  - Now tracked in: Phase 2 "Enable DeepStream native NMS clustering"
  - Enhanced: Use GPU-accelerated DBSCAN clustering

- [x] **Move Python post-processing to C++/Numba** - ✅ SUPERSEDED by Phase 3
  - Now tracked in: Phase 3 "Add Numba JIT compilation for hot paths"
  - Enhanced: Background threading + JIT compilation

- [ ] **GPU-accelerate field mask validation**
  - **File**: `new_week/utils/field_mask.py`
  - **Issue**: Per-detection mask lookup on CPU
  - **Impact**: O(n) overhead for n detections
  - **Fix**: Move to CUDA kernel or use texture memory
  - **Priority**: P3 - Deferred (low impact vs other optimizations)

- [ ] **Implement async CSV logging**
  - **File**: `new_week/utils/csv_logger.py`
  - **Issue**: Synchronous file I/O in probe callbacks
  - **Impact**: Frame drops under high detection load
  - **Fix**: Use async logging or buffered writes
  - **Priority**: P3 - Deferred (CSV logging is debug-only)

---

### Testing & Validation

#### Refactored Code Validation

**Source**: `new_week/VALIDATION_CHECKLIST.md`

- [ ] **Functional testing - Panorama mode**
  - Test with files: left.mp4 + right.mp4
  - Verify: Bboxes drawn correctly, no crashes
  - Duration: 5 minutes

- [ ] **Functional testing - Virtual camera mode**
  - Test with files: left.mp4 + right.mp4
  - Verify: Ball tracking, smooth camera movement
  - Duration: 5 minutes

- [ ] **Functional testing - Stream mode**
  - Test with RTMP stream
  - Verify: Stream stability, no frame drops
  - Duration: 10 minutes

- [ ] **Functional testing - Record mode**
  - Test with file output
  - Verify: Recording quality, audio sync
  - Duration: 5 minutes

- [ ] **Performance benchmarking**
  - Compare: original vs refactored FPS
  - Compare: GPU utilization
  - Compare: CPU usage
  - Compare: Memory usage
  - Target: <5% performance variance

- [ ] **Long-running stability test**
  - Duration: 8+ hours continuous run
  - Monitor: Memory leaks, performance degradation
  - Check: Buffer management stability, detection quality
  - Target: No crashes, <50MB memory growth per hour

---

## Backlog

### Code Quality Improvements

- [ ] Add type hints to all function signatures
  - Use `typing` module for Optional, Dict, List, etc.
  - Document parameter and return types
  - Priority: P3 - Maintainability

- [ ] Implement configuration validation
  - Add JSON schema or dataclass for config
  - Validate on startup
  - Priority: P3 - Robustness

- [ ] Add GStreamer debug categories
  - Set custom debug levels per component
  - Enable conditional debug logging
  - Priority: P3 - Debugging

- [ ] Implement pipeline state machine
  - Use enum for states (CREATED, PLAYING, PAUSED, ERROR, STOPPED)
  - Add state transition validation
  - Priority: P3 - Robustness

- [ ] Standardize logging format
  - Remove emojis from log messages
  - Use consistent format: "MODULE: Operation - details"
  - Priority: P3 - Log parsing

---

### Testing Infrastructure

- [ ] **Create unit tests for TensorProcessor**
  - Test: Empty tensor handling
  - Test: Multi-class parsing
  - Test: Coordinate transformation
  - Coverage target: >80%

- [ ] **Create unit tests for HistoryManager**
  - Test: Detection storage (3-tier)
  - Test: Trajectory interpolation
  - Test: Outlier filtering
  - Coverage target: >80%

- [ ] **Create unit tests for BufferManager**
  - Test: Frame buffering
  - Test: Timestamp retrieval
  - Test: Audio/video sync
  - Coverage target: >80%

- [ ] **Create integration tests**
  - Test: End-to-end pipeline
  - Test: All display modes
  - Test: Error recovery

- [ ] **Add performance regression tests**
  - Benchmark: FPS over time
  - Benchmark: Memory usage over time
  - Alert on >5% degradation

---

### Documentation

- [ ] Add metadata hierarchy diagram to code comments
  - Document NvDsBatchMeta structure
  - Show frame_meta, obj_meta, user_meta relationships
  - Reference: DeepStream docs

- [ ] Add pipeline structure diagram to code
  - Document camera → stitching → tiling → inference flow
  - Show probe attachment points
  - Reference: GStreamer docs

- [ ] Create plugin development guide
  - Document custom plugin creation (my_steach, my_virt_cam, my_tile_batcher)
  - Include CUDA kernel examples
  - Include GStreamer integration

- [ ] Create deployment guide
  - Document Jetson setup
  - Include systemd service configuration
  - Add monitoring/logging setup

---

### Advanced Features (Phase 4)

#### Analytics (Q1 2026)

- [ ] Implement player movement heatmaps
  - Store player positions over time
  - Generate heatmap visualization
  - Export to image/video

- [ ] Add ball possession tracking
  - Detect ball-player proximity
  - Track possession time per team
  - Generate possession statistics

- [ ] Add speed/distance metrics
  - Calculate player speed from position history
  - Calculate total distance traveled
  - Export to CSV/JSON

- [ ] Implement event detection
  - Detect: goals, corners, throw-ins
  - Timestamp events
  - Log to database

- [ ] Create real-time statistics dashboard
  - Display: FPS, detection counts, tracking status
  - Web interface (Flask/FastAPI)
  - WebSocket for real-time updates

#### Multi-Camera Support (Q2 2026)

- [ ] Support 3-4 camera inputs
  - Extend nvstreammux to handle 3-4 sources
  - Update stitching to handle multiple overlap zones
  - Test with 3-camera setup

- [ ] Implement 360° full panorama
  - Extend panorama width to ~10000px
  - Update tile batcher for wider panorama
  - Test inference performance

- [ ] Add multi-stream synchronization
  - Hardware sync (GPIO triggers)
  - Software sync (timestamp alignment)
  - Validation with sync checker

- [ ] Dynamic camera selection
  - Auto-switch based on ball position
  - Select best view angle
  - Smooth transitions

#### Cloud Integration (Q2 2026)

- [ ] Add cloud storage for recordings
  - Upload to S3/GCS after game
  - Automatic retention policy
  - Bandwidth management

- [ ] Create remote monitoring dashboard
  - View live stream remotely
  - Display system metrics
  - Web interface

- [ ] Multi-game streaming
  - Support multiple pipelines simultaneously
  - Resource allocation per game
  - Test on dGPU server

- [ ] Expose analytics API
  - REST API for detection data
  - WebSocket for live updates
  - Authentication/authorization

#### AI Model Improvements (Q3 2026)

- [ ] Add player identification
  - Train jersey number recognition
  - OCR integration
  - Player tracking across frames

- [ ] Implement team classification
  - Color-based team detection
  - Track team possession
  - Team statistics

- [ ] Add action recognition
  - Detect: kick, pass, tackle, header
  - Classify action types
  - Event timeline

- [ ] Implement referee gesture recognition
  - Detect: red card, yellow card, goal signal
  - Associate with events
  - Auto-highlight in recording

---

## Completed Tasks ✅

### Phase 4.0: Critical Performance Analysis (Nov 17, 2025)
- [x] Comprehensive GPU/CPU pipeline analysis
- [x] Root cause identification (4 critical bottlenecks)
- [x] DLA feasibility assessment (not useful for FP16)
- [x] Memory flow architecture analysis
- [x] Created GPU_CPU_OPTIMIZATION_PLAN.md (comprehensive)
- [x] Created PIPELINE_MEMORY_FLOW_ANALYSIS.md (visual comparison)
- [x] Defined 3-phase implementation plan with success criteria
- [x] Updated todo.md with actionable tasks

### Phase 3.2: DeepStream 7.1 Compliance (Nov 17, 2025)
- [x] Fix metadata iteration StopIteration handling in analysis_probe.py
- [x] Fix metadata iteration StopIteration handling in display_probe.py
- [x] Fix nvinfer configuration (num-detected-classes=5)
- [x] Add missing [class-attrs-4] section for main_referee class
- [x] Add user metadata validation in analysis_probe.py
- [x] Document probe return values with inline comments

### Phase 3.1: Code Refactoring (Nov 16, 2025)
- [x] Refactor monolithic code to modular architecture
- [x] Create 8 focused modules (utils, core, processing, rendering, pipeline)
- [x] Achieve 76% code reduction in main file
- [x] Maintain 100% API compatibility
- [x] Create comprehensive refactoring documentation

### Phase 3.4: Documentation Reorganization (Nov 17, 2025)
- [x] Create plan.md (master roadmap)
- [x] Create architecture.md (system design)
- [x] Create todo.md (this file)
- [x] Create decisions.md (architectural choices)
- [x] Move reports to docs/reports/
- [x] Remove redundant documentation files

### Phase 2: Intelligent Tracking (Oct-Nov 2025)
- [x] Implement 10-second detection history
- [x] Add trajectory interpolation (parabolic for flight)
- [x] Implement outlier detection and blacklisting
- [x] Add player center-of-mass tracking
- [x] Implement speed-based auto-zoom
- [x] Add 7-second buffering system

### Phase 1: Core Pipeline (Sep-Oct 2025)
- [x] Camera calibration (stereo essential matrix)
- [x] Panorama stitching plugin (my_steach)
- [x] Tile batcher plugin (my_tile_batcher)
- [x] Virtual camera plugin (my_virt_cam)
- [x] YOLOv11 integration with TensorRT
- [x] Multi-class detection (ball, players, staff, referees)

---

## Priority Legend

- **P0**: Critical - Blocks production deployment
- **P1**: High - Important for stability/performance
- **P2**: Medium - Optimization/enhancement
- **P3**: Low - Nice-to-have, maintainability

## Status Legend

- ⏳ In Progress
- 📋 Planned
- ✅ Complete
- ❌ Blocked

---

**Document Owner**: edvin3i
**Review Frequency**: Weekly during active development
**Current Sprint**: Week 47-50 (GPU/CPU Optimization)
**Next Review**: 2025-11-24 (Phase 1 Progress Check)
**Critical Milestone**: Week 50 - All optimizations complete, production-ready performance
