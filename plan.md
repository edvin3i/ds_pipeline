# DeepStream Sports Analytics Pipeline - Master Plan

## Project Vision

Build a production-grade real-time AI sports analytics system on NVIDIA Jetson Orin NX that processes dual 4K camera feeds to create 360° panoramic video with intelligent ball/player tracking and automated virtual camera control.

**Target Platform**: NVIDIA Jetson Orin NX 16GB
**Performance Target**: 30 FPS real-time processing with <100ms latency
**Status**: Phase 3 (Production Optimization) - 85% Complete

---

## Development Phases

### Phase 1: Core Pipeline Development ✅ COMPLETE

**Goal**: Establish basic dual-camera stitching and detection pipeline

#### 1.1 Camera Calibration ✅
- [x] Individual camera calibration (IMX678 sensors)
- [x] Stereo calibration with essential matrix method
- [x] Wide-angle calibration (85° camera separation)
- [x] Lens distortion modeling (F-Tan-Theta)
- [x] Validation with chessboard pattern

**Deliverables**: `calibration/` module, calibration_result_standard.pkl

#### 1.2 Panorama Stitching ✅
- [x] LUT-based CUDA stitching kernel
- [x] Custom GStreamer plugin (my_steach)
- [x] Weighted blending in overlap zones
- [x] Color correction system (asynchronous)
- [x] 5700×1900 panorama output @ 30 FPS

**Deliverables**: `my_steach/` plugin, PLUGIN.md

#### 1.3 Object Detection ✅
- [x] YOLOv11 model integration
- [x] TensorRT FP16 optimization
- [x] Custom tile batcher plugin (my_tile_batcher)
- [x] 6×1024×1024 tile extraction
- [x] Multi-class detection (ball, players, referees, staff)

**Deliverables**: `my_tile_batcher/` plugin, `models/yolo11n_mixed_finetune_v9.engine`

#### 1.4 Virtual Camera ✅
- [x] Equirectangular → perspective projection
- [x] Custom GStreamer plugin (my_virt_cam)
- [x] 3-stage CUDA transformation kernel
- [x] LUT caching system
- [x] 1920×1080 output @ 47 FPS

**Deliverables**: `my_virt_cam/` plugin, PLUGIN.md

---

### Phase 2: Intelligent Tracking & Analysis ✅ COMPLETE

**Goal**: Implement ball tracking, history management, and intelligent camera control

#### 2.1 Detection History System ✅
- [x] 10-second temporal buffer
- [x] Three-tier storage (raw → processed → confirmed)
- [x] Trajectory interpolation (parabolic for flight)
- [x] Outlier detection and blacklisting
- [x] Field mask filtering

**Deliverables**: `new_week/core/` module (history_manager, detection_storage, trajectory_filter, trajectory_interpolator)

#### 2.2 Player Tracking ✅
- [x] Center-of-mass calculation
- [x] EMA smoothing (α = 0.18)
- [x] Fallback target when ball lost
- [x] Multi-class support (players, staff, referees)

**Deliverables**: `new_week/core/players_history.py`

#### 2.3 Intelligent Camera Control ✅
- [x] Speed-based auto-zoom (300-1200 px/s)
- [x] Smooth pursuit (smooth factor 0.3)
- [x] Ball loss recovery (FOV expansion 2°/s)
- [x] Backward interpolation for gap filling
- [x] Player fallback mode (>3s ball loss)

**Deliverables**: `new_week/rendering/virtual_camera_probe.py`

#### 2.4 Buffering & Playback ✅
- [x] 7-second RAM buffer system
- [x] Frame/audio synchronization
- [x] Timestamp-based retrieval
- [x] Background playback thread
- [x] Multiple output modes (panorama, virtualcam, stream, record)

**Deliverables**: `new_week/pipeline/buffer_manager.py`

---

### Phase 3: Production Optimization 🔄 IN PROGRESS (90%)

**Goal**: Optimize performance, improve code quality, and production-readiness

#### 3.1 Code Refactoring ✅
- [x] Monolithic → modular architecture
- [x] 76% code reduction in main file
- [x] SOLID principles application
- [x] 8 focused modules with clear responsibilities
- [x] 100% API compatibility maintained

**Deliverables**: `version_masr_multiclass_REFACTORED.py`, refactoring documentation

**Status**: Code complete, testing pending

#### 3.2 Code Quality Improvements ✅ COMPLETE
- [x] Fix DeepStream 7.1 compliance issues (3 critical P0 issues resolved)
- [x] Add StopIteration handling in metadata iteration
- [x] Fix nvinfer config (num-detected-classes=5, add class-attrs-4)
- [x] Add user metadata validation
- [x] Document probe return values with inline comments
- [ ] Improve error handling (tensor processor - P1, moved to 3.3)

**Reference**: `docs/reports/DEEPSTREAM_CODE_REVIEW.md`

**Status**: Critical P0 issues COMPLETE (Nov 17, 2025)

#### 3.3 Performance Optimization ⏳ PENDING
- [ ] Optimize BufferManager deep copies (CPU bottleneck)
- [ ] Replace O(n) timestamp scans with binary search
- [ ] Move heavy Python processing to C++/Numba
- [ ] GPU-accelerate field mask validation
- [ ] Implement native NMS (replace Python loops)
- [ ] Async CSV logging (avoid frame drops)

**Reference**: `docs/reports/CODEX_report.md`

**Status**: Bottlenecks identified, optimization planned

#### 3.4 Documentation Reorganization ✅
- [x] Create master plan (plan.md)
- [x] Create architecture doc (architecture.md)
- [x] Create TODO list (todo.md)
- [x] Create decisions log (decisions.md)
- [x] Organize reports in docs/reports/
- [x] Remove redundant documentation

**Status**: COMPLETE

#### 3.5 Testing & Validation ⏳ PENDING
- [ ] Unit tests for core components
- [ ] Integration tests for pipeline
- [ ] Performance benchmarking (refactored vs original)
- [ ] Long-running stability tests (1+ hours)
- [ ] Memory leak detection
- [ ] Regression testing

**Target**: 100% feature parity, <5% performance variance

**Status**: Test plan created, execution pending

---

### Phase 4: Advanced Features 📋 PLANNED

**Goal**: Enhanced analytics and production features

#### 4.1 Advanced Analytics (Q1 2026)
- [ ] Player movement heatmaps
- [ ] Ball possession tracking
- [ ] Speed/distance metrics
- [ ] Event detection (goals, fouls, corners)
- [ ] Real-time statistics dashboard

#### 4.2 Multi-Camera Support (Q2 2026)
- [ ] Support for 3-4 camera inputs
- [ ] 360° full panorama coverage
- [ ] Multi-stream synchronization
- [ ] Dynamic camera selection

#### 4.3 Cloud Integration (Q2 2026)
- [ ] Cloud storage for recordings
- [ ] Remote monitoring dashboard
- [ ] Multi-game simultaneous streaming
- [ ] Analytics API

#### 4.4 AI Model Improvements (Q3 2026)
- [ ] Player identification (jersey numbers)
- [ ] Team classification
- [ ] Action recognition (kick, pass, tackle)
- [ ] Referee gesture recognition

---

## Current Sprint (Nov 2025)

### Sprint Goal
Complete Phase 3.2-3.5: Code quality, performance optimization, and validation

### Sprint Tasks
1. **Week 1 (Nov 18-24)**: ✅ Documentation reorganization + DeepStream P0 fixes
   - [x] Create plan.md, architecture.md, todo.md, decisions.md
   - [x] Move reports to docs/reports/
   - [x] Clean up redundant files
   - [x] Fix metadata iteration (StopIteration handling) - COMPLETED Nov 17
   - [x] Fix nvinfer config - COMPLETED Nov 17
   - [x] Add metadata validation - COMPLETED Nov 17

2. **Week 2 (Nov 25-Dec 1)**: Performance optimization
   - [ ] Optimize BufferManager deep copies
   - [ ] Optimize playback timestamp scans
   - [ ] Test and benchmark improvements

3. **Week 3 (Dec 2-8)**: Performance optimization
   - [ ] Optimize BufferManager
   - [ ] Optimize field mask validation
   - [ ] Optimize NMS implementation
   - [ ] Benchmark improvements

4. **Week 4 (Dec 9-15)**: Testing & validation
   - [ ] Unit tests for core modules
   - [ ] Integration tests
   - [ ] Performance benchmarks
   - [ ] Stability tests

---

## Success Metrics

### Technical Metrics
- [x] **Pipeline FPS**: 30 FPS stable @ 70% GPU load
- [x] **End-to-end latency**: <100ms (camera to display)
- [x] **Detection accuracy**: >95% ball detection, >90% player detection
- [x] **Virtual camera smoothness**: <0.5° jitter
- [x] **Memory usage**: <13 GB / 16 GB available
- [ ] **Code quality**: All critical DeepStream issues fixed
- [ ] **Test coverage**: >80% for core modules

### Operational Metrics
- [x] **Stability**: No crashes in 1-hour continuous run
- [ ] **Stability**: No crashes in 8-hour continuous run (pending)
- [ ] **Memory leaks**: <50 MB growth per hour (pending validation)
- [x] **Recovery**: Auto-restart on camera failure

---

## Risk Register

### High Priority Risks
1. **Memory bandwidth saturation** (Jetson limitation: 102 GB/s)
   - **Mitigation**: NVMM zero-copy, no CPU transfers
   - **Status**: Mitigated ✅

2. **Inference latency on 6-tile batch** (~20ms)
   - **Mitigation**: DLA offload investigation
   - **Status**: Acceptable, optimization optional

3. **CPU bottlenecks in Python processing**
   - **Mitigation**: Move to C++/Numba
   - **Status**: Identified, fix planned

### Medium Priority Risks
4. **Buffer encoding overhead** (H.264 CPU encoder)
   - **Mitigation**: Use nvenc hardware encoder
   - **Status**: Workaround available

5. **Detection false positives** (staff/referees classified as players)
   - **Mitigation**: Post-processing filters, model retraining
   - **Status**: Acceptable for Phase 3

---

## Dependencies

### Hardware
- [x] NVIDIA Jetson Orin NX 16GB
- [x] 2× Sony IMX678 cameras with MIPI CSI-2
- [x] L100A lenses (100° FOV)

### Software
- [x] JetPack 6.2+
- [x] DeepStream SDK 7.1
- [x] CUDA 12.6
- [x] TensorRT 10.5+
- [x] GStreamer 1.0
- [x] Python 3.8+
- [x] OpenCV 4.5+

### External
- [x] YOLOv11 pretrained model
- [x] Calibration pattern (8×6 chessboard)
- [x] Field mask image (field_mask.png)

---

## Deployment Checklist

### Pre-Production ⏳
- [x] All Phase 1-2 features complete
- [x] Code refactored to modular architecture
- [ ] All critical bugs fixed (DeepStream compliance)
- [ ] Performance optimization complete
- [ ] Test suite passing
- [ ] Documentation complete

### Production Ready 📋
- [ ] Stability validated (8+ hours continuous)
- [ ] Recovery mechanisms tested
- [ ] Monitoring/logging in place
- [ ] Deployment scripts ready
- [ ] Rollback plan documented

### Post-Deployment 📋
- [ ] Production monitoring active
- [ ] Performance metrics tracked
- [ ] Issue tracker configured
- [ ] Maintenance schedule defined

---

## Team & Roles

**Current**: Solo development (Claude assisting)

**Future Roles** (when scaling):
- ML Engineer: Model training and optimization
- Systems Engineer: Jetson deployment and maintenance
- DevOps: Cloud integration and monitoring
- QA Engineer: Testing and validation

---

## Next Milestones

### Immediate (Dec 2025)
- [ ] Complete Phase 3 (Production Optimization)
- [ ] Deploy to production environment
- [ ] Collect 1 month of operational data

### Short-term (Q1 2026)
- [ ] Phase 4.1: Advanced analytics features
- [ ] Model improvements (player ID, action recognition)
- [ ] Multi-game support

### Long-term (Q2-Q3 2026)
- [ ] Multi-camera support (3-4 cameras)
- [ ] Cloud integration
- [ ] Commercial deployment

---

## Change Log

**2025-11-17**: Phase 3.2 complete - All critical P0 DeepStream compliance fixes
**2025-11-17**: Documentation reorganization complete
**2025-11-16**: Code refactoring complete (76% reduction)
**2025-11-15**: Buffer manager extraction complete
**2025-11**: Phase 2 (Intelligent Tracking) complete
**2025-10**: Phase 1 (Core Pipeline) complete
**2025-09**: Project initiated

---

**Document Owner**: edvin3i
**Last Updated**: 2025-11-17
**Status**: Phase 3 - 90% Complete (Phase 3.2 COMPLETE)
