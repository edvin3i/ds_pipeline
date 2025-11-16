# Refactoring Architecture Diagram

## Before: Monolithic Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│           version_masr_multiclass.py (3,015 lines)                  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                                 │  │
│  │    PanoramaWithVirtualCamera Class (2,000+ lines)             │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Pipeline Building (500 lines)                           │  │  │
│  │  │ - create_pipeline()                                     │  │  │
│  │  │ - create_playback_pipeline()                            │  │  │
│  │  │ - _create_analysis_tiles()                              │  │  │
│  │  │ - create_inference_config()                             │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Buffer Management (400 lines)                           │  │  │
│  │  │ - on_new_sample()                                       │  │  │
│  │  │ - on_new_audio_sample()                                 │  │  │
│  │  │ - _buffer_loop()                                        │  │  │
│  │  │ - _on_appsrc_need_data()                                │  │  │
│  │  │ - _on_audio_appsrc_need_data()                          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Probe Handlers (600 lines)                              │  │  │
│  │  │ - analysis_probe()                                      │  │  │
│  │  │ - vcam_update_probe()                                   │  │  │
│  │  │ - playback_draw_probe()                                 │  │  │
│  │  │ - frame_skip_probe()                                    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Utilities (200 lines)                                   │  │  │
│  │  │ - find_usb_audio_device()                               │  │  │
│  │  │ - _apply_adaptive_distance_filter()                     │  │  │
│  │  │ - _emergency_shutdown()                                 │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Orchestration (300 lines)                               │  │  │
│  │  │ - __init__()                                            │  │  │
│  │  │ - run()                                                 │  │  │
│  │  │ - stop()                                                │  │  │
│  │  │ - _on_bus_message()                                     │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Supporting Classes (800 lines)                               │  │
│  │ - BallDetectionHistory                                       │  │
│  │ - PlayersHistory                                             │  │
│  │ - TensorProcessor                                            │  │
│  │ - FieldMaskBinary                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Utility Functions (200 lines)                                │  │
│  │ - save_detection_to_csv()                                    │  │
│  │ - apply_nms()                                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

Problems:
❌ Everything in one file (3,015 lines)
❌ Tight coupling between components
❌ Hard to test individual pieces
❌ Difficult to understand flow
❌ Changes affect entire file
```

## After: Modular Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Main Orchestrator (712 lines)                          │
│             version_masr_multiclass_REFACTORED.py                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PanoramaWithVirtualCamera (400 lines)                         │    │
│  │                                                                  │    │
│  │  ┌────────────────────┐  ┌────────────────────┐               │    │
│  │  │ __init__()         │  │ Composition:       │               │    │
│  │  │ - Config storage   │  │ ✓ ConfigBuilder    │               │    │
│  │  │ - Initialize       │  │ ✓ HistoryManager   │               │    │
│  │  │   components       │  │ ✓ BufferManager    │               │    │
│  │  │ - Setup delegation │  │ ✓ ProbeHandlers    │               │    │
│  │  └────────────────────┘  └────────────────────┘               │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │ Orchestration Methods                                    │  │    │
│  │  │ • create_pipeline()    → delegates to PipelineBuilder    │  │    │
│  │  │ • create_playback()    → delegates to PlaybackBuilder    │  │    │
│  │  │ • run()                → lifecycle management            │  │    │
│  │  │ • stop()               → cleanup coordination            │  │    │
│  │  │ • _on_bus_message()    → error handling                  │  │    │
│  │  │ • frame_skip_probe()   → simple frame counting           │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  main() Function (300 lines)                                   │    │
│  │  - Argument parsing                                            │    │
│  │  - Validation                                                  │    │
│  │  - App instantiation                                           │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Uses ↓
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
┌───────────────────────────┐              ┌───────────────────────────┐
│      utils/ (150 lines)   │              │    core/ (800 lines)      │
├───────────────────────────┤              ├───────────────────────────┤
│ FieldMaskBinary           │              │ HistoryManager            │
│ - is_inside_field()       │              │ - add_detection()         │
│                           │              │ - get_detection()         │
│ CSV Logger                │              │ - get_future_trajectory() │
│ - save_detection_to_csv() │              │                           │
│                           │              │ PlayersHistory            │
│ NMS                       │              │ - add_players()           │
│ - apply_nms()             │              │ - get_players()           │
│                           │              │ - calculate_center()      │
│ Components:               │              │                           │
│ • field_mask.py           │              │ Sub-components:           │
│ • csv_logger.py           │              │ • detection_storage.py    │
│ • nms.py                  │              │ • trajectory_filter.py    │
└───────────────────────────┘              │ • trajectory_interpolator │
                                           └───────────────────────────┘
                │                                       │
                │                                       │
                ▼                                       ▼
┌───────────────────────────┐              ┌───────────────────────────┐
│ processing/ (600 lines)   │              │  rendering/ (700 lines)   │
├───────────────────────────┤              ├───────────────────────────┤
│ TensorProcessor           │              │ VirtualCameraProbeHandler │
│ - postprocess_yolo()      │              │ - vcam_update_probe()     │
│ - Parse YOLO output       │              │ - Ball tracking           │
│ - Multi-class detection   │              │ - Speed-based zoom        │
│                           │              │ - Player fallback         │
│ AnalysisProbeHandler      │              │                           │
│ - analysis_probe()        │              │ DisplayProbeHandler       │
│ - Process tiles           │              │ - playback_draw_probe()   │
│ - Apply field mask        │              │ - Draw bboxes (multi)     │
│ - Store detections        │              │ - Future trajectory       │
│                           │              │ - OSD management          │
│ Components:               │              │                           │
│ • tensor_processor.py     │              │ Components:               │
│ • analysis_probe.py       │              │ • virtual_camera_probe.py │
└───────────────────────────┘              │ • display_probe.py        │
                                           └───────────────────────────┘
                │
                │
                ▼
┌───────────────────────────────────────────┐
│       pipeline/ (1,200 lines)             │
├───────────────────────────────────────────┤
│ ConfigBuilder                             │
│ - create_inference_config()               │
│                                           │
│ PipelineBuilder                           │
│ - build() → analysis pipeline             │
│ - Source config (files/cameras)           │
│ - nvstreammux, tilebatcher, nvinfer       │
│                                           │
│ PlaybackPipelineBuilder                   │
│ - build() → playback pipeline             │
│ - Display modes (panorama/vcam/stream)    │
│ - Encoding and output                     │
│                                           │
│ BufferManager                             │
│ - on_new_sample()                         │
│ - _buffer_loop()                          │
│ - appsrc callbacks                        │
│ - Audio/video sync                        │
│                                           │
│ Components:                               │
│ • config_builder.py                       │
│ • pipeline_builder.py                     │
│ • playback_builder.py                     │
│ • buffer_manager.py                       │
└───────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  Video Files │
│  or Cameras  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS PIPELINE                            │
│                (built by PipelineBuilder)                       │
│                                                                 │
│  nvstreammux → nvtilebatcher → nvinfer → tee                   │
│                                           ├─→ analysis_probe    │
│                                           └─→ appsink           │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │ Frames captured
                                 ▼
                    ┌────────────────────────┐
                    │    BufferManager       │
                    │  • Frame buffer        │
                    │  • Audio buffer        │
                    │  • Timestamp sync      │
                    └────────┬───────────────┘
                             │
                             │ Delayed playback
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PLAYBACK PIPELINE                             │
│              (built by PlaybackPipelineBuilder)                 │
│                                                                 │
│  appsrc → nvvideoconvert → [vcam_probe] → nvdsosd → display    │
│                             │               │                   │
│                             │               └─→ draw_probe      │
│                             └─→ Virtual camera control          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Display:     │
                    │ • Panorama     │
                    │ • Virtual Cam  │
                    │ • Stream       │
                    │ • Record       │
                    └────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION FLOW                               │
│                                                                 │
│  YOLO Tensor → TensorProcessor → AnalysisProbeHandler          │
│      │             │                      │                     │
│      │             ▼                      ▼                     │
│      │      Parse multi-class       Field mask filter          │
│      │      detections              (is_inside_field)          │
│      │                                   │                     │
│      │                                   ▼                     │
│      │                          Store in histories:           │
│      │                          • HistoryManager (ball)        │
│      │                          • PlayersHistory (players)     │
│      │                          • all_detections_history       │
│      │                                                         │
│      └──────────────────────────────────────────────────────────┘
                                           │
                                           │ Used by
                                           ▼
                    ┌──────────────────────────────────────┐
                    │       Rendering Handlers             │
                    │                                      │
                    │  VirtualCameraProbeHandler           │
                    │  • Track ball position               │
                    │  • Calculate FOV/zoom                │
                    │  • Fallback to players               │
                    │                                      │
                    │  DisplayProbeHandler                 │
                    │  • Draw bboxes (ball: red)           │
                    │  • Draw bboxes (players: green)      │
                    │  • Show future trajectory            │
                    └──────────────────────────────────────┘
```

## Component Interaction Map

```
┌────────────────────────────────────────────────────────────────┐
│            PanoramaWithVirtualCamera (Orchestrator)            │
└───┬───────────┬────────────┬─────────────┬──────────┬─────────┘
    │           │            │             │          │
    │ creates   │ creates    │ creates     │ creates  │ creates
    │           │            │             │          │
    ▼           ▼            ▼             ▼          ▼
┌──────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│Field │  │History  │  │Players   │  │Tensor   │  │Buffer    │
│Mask  │  │Manager  │  │History   │  │Processor│  │Manager   │
└──┬───┘  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘
   │           │            │             │            │
   │ used by   │ used by    │ used by     │ used by    │ manages
   ▼           ▼            ▼             ▼            ▼
┌──────────────────────────────────────────────────────────┐
│              Probe Handlers (created later)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Analysis     │  │ VirtualCam   │  │ Display      │  │
│  │ ProbeHandler │  │ ProbeHandler │  │ ProbeHandler │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
         │                   │                   │
         │ callbacks         │ callbacks         │ callbacks
         ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────┐
│                    GStreamer Pipelines                     │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │ Analysis         │         │ Playback         │        │
│  │ Pipeline         │────────▶│ Pipeline         │        │
│  │ (PipelineBuilder)│ buffers │ (PlaybackBuilder)│        │
│  └──────────────────┘         └──────────────────┘        │
└────────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
version_masr_multiclass_REFACTORED.py
 ├─ utils/
 │   ├─ field_mask
 │   ├─ csv_logger
 │   └─ nms
 │
 ├─ core/
 │   ├─ history_manager
 │   │   ├─ detection_storage
 │   │   ├─ trajectory_filter
 │   │   └─ trajectory_interpolator
 │   └─ players_history
 │
 ├─ processing/
 │   ├─ tensor_processor
 │   └─ analysis_probe
 │       ├─ uses: tensor_processor
 │       ├─ uses: field_mask (utils)
 │       ├─ uses: history_manager (core)
 │       └─ uses: players_history (core)
 │
 ├─ rendering/
 │   ├─ virtual_camera_probe
 │   │   ├─ uses: history_manager (core)
 │   │   └─ uses: players_history (core)
 │   └─ display_probe
 │       ├─ uses: history_manager (core)
 │       └─ uses: players_history (core)
 │
 └─ pipeline/
     ├─ config_builder
     ├─ pipeline_builder
     ├─ playback_builder
     └─ buffer_manager
         └─ uses: history_manager (core)
```

## Benefits Visualization

```
                    BEFORE                              AFTER
                 ═══════════                         ═══════════

Complexity:      ████████████████ (High)              ███ (Low)
                 All in one file                      Separated concerns

Coupling:        ████████████████ (Tight)             ██ (Loose)
                 Everything coupled                   Clear interfaces

Testability:     ██ (Hard)                            ████████████ (Easy)
                 Test entire system                   Test components

Readability:     ███ (Hard)                           ████████████ (Easy)
                 3000+ lines to scan                  ~400 lines orchestration

Maintainability: ███ (Hard)                           ████████████ (Easy)
                 Changes affect all                   Isolated changes

Reusability:     █ (Minimal)                          ████████████ (High)
                 Can't extract parts                  Reuse components
```

## Summary

The refactoring transforms a monolithic 3,000-line file into a clean, modular architecture where:

- **Each module has one clear responsibility**
- **Dependencies flow in one direction** (main → modules, no circular deps)
- **Components are loosely coupled** via interfaces
- **Main class orchestrates** rather than implements
- **Testing is straightforward** (mock dependencies, test in isolation)
- **Extensions are easy** (add new modules or extend existing ones)

This is a **textbook example** of applying **SOLID principles** and **clean architecture** to a real-world computer vision pipeline.
