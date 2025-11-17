# DeepStream Sports Analytics Pipeline - TODO List

**Last Updated**: 2025-11-17
**Status**: Phase 3 - Production Optimization

---

## Current Sprint: Production Readiness

### High Priority (Week 2: Nov 25 - Dec 1)

#### DeepStream 7.1 Compliance Fixes

**Source**: `docs/reports/DEEPSTREAM_CODE_REVIEW.md`

- [ ] **CRITICAL: Fix metadata iteration StopIteration handling**
  - **Files**: `new_week/processing/analysis_probe.py:174-214`, `new_week/rendering/display_probe.py:285-288`
  - **Issue**: Missing try/except blocks can cause crashes when metadata list ends unexpectedly
  - **Fix**: Wrap all `cast()` and `next()` calls in try/except StopIteration
  - **Priority**: P0 - Can cause production crashes

- [ ] **CRITICAL: Fix nvinfer configuration**
  - **File**: `new_week/config_infer.txt`
  - **Issue 1**: `num-detected-classes=1` should be `2` (ball, player), ignore for now: staff, side_ref, main_ref
  - **Issue 2**: Missing `[class-attrs-4]` section for main_referee class
  - **Fix**: Update config file with correct class count and add missing section
  - **Priority**: P0 - Incorrect detections

- [ ] **CRITICAL: Add user metadata validation**
  - **File**: `new_week/processing/analysis_probe.py:192-211`
  - **Issue**: No validation that user_meta_data is properly initialized
  - **Fix**: Add null checks after casting NvDsUserMeta
  - **Priority**: P0 - Potential memory leaks

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

### Medium Priority (Week 3: Dec 2-8)

#### Performance Optimizations

**Source**: `docs/reports/CODEX_report.md`

- [ ] **Optimize BufferManager deep copies**
  - **File**: `new_week/pipeline/buffer_manager.py:116`
  - **Issue**: Deep copying ~15MB frames on every buffer add
  - **Impact**: Significant CPU overhead in frame buffering
  - **Fix**: Use shallow copies or zero-copy techniques
  - **Priority**: P1 - CPU bottleneck

- [ ] **Optimize playback timestamp scans**
  - **File**: `new_week/pipeline/buffer_manager.py:156`
  - **Issue**: Linear O(n) search through buffer for timestamps
  - **Impact**: Performance degrades with buffer size
  - **Fix**: Implement binary search or ring buffer with indexing
  - **Priority**: P1 - CPU bottleneck

- [ ] **Move Python post-processing to C++/Numba**
  - **Files**: `new_week/processing/analysis_probe.py`, `new_week/rendering/display_probe.py`
  - **Issue**: Heavy Python processing in critical path
  - **Impact**: CPU bottleneck in detection pipeline
  - **Fix**: JIT compile with Numba or rewrite in C++
  - **Priority**: P2 - Performance optimization

- [ ] **GPU-accelerate field mask validation**
  - **File**: `new_week/utils/field_mask.py`
  - **Issue**: Per-detection mask lookup on CPU
  - **Impact**: O(n) overhead for n detections
  - **Fix**: Move to CUDA kernel or use texture memory
  - **Priority**: P2 - Performance optimization

- [ ] **Replace Python NMS with native implementation**
  - **File**: `new_week/utils/nms.py`
  - **Issue**: Pure Python with nested loops
  - **Impact**: O(n²) complexity for overlapping detections
  - **Fix**: Use DeepStream native NMS or GPU implementation
  - **Priority**: P2 - Performance optimization

- [ ] **Implement async CSV logging**
  - **File**: `new_week/utils/csv_logger.py`
  - **Issue**: Synchronous file I/O in probe callbacks
  - **Impact**: Frame drops under high detection load
  - **Fix**: Use async logging or buffered writes
  - **Priority**: P2 - Reliability

---

### Testing & Validation (Week 4: Dec 9-15)

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
**Next Review**: 2025-11-25 (Sprint Week 2)
