# DeepStream 7.1 Bottleneck Review and Optimization Plan

This document expands the DeepStream 7.1 bottleneck review with a concrete optimization plan that preserves current functionality while preparing the pipeline for throughput and latency gains.

## `new_week/version_masr_multiclass.py`
- The display branch converts NVMM frames to CPU memory (`video/x-raw,format=RGB`) before handing them to `appsink`, forcing a GPU→CPU copy and extra conversion. This violates the NVMM end-to-end guidance and keeps rendering enabled even when analyzing throughput.【F:new_week/version_masr_multiclass.py†L1565-L1576】  
- Action plan (non-breaking):
  - Introduce a "performance" flag that routes the display branch to NVMM-only sinks (`fakesink` or `appsink` with `memory:NVMM`) and disables `nvvideoconvert`/RGB conversion when visualization is not required. Default keeps the existing path to avoid regressions.
  - Keep `tee` + `queue` structure but cap `max-size-buffers/time` for the display queue based on measured latency to avoid frame buildup on Orin DRAM per DeepStream guidance to place queues before CPU-bound elements.【F:TARGET_DEV.MD†L11-L23】
  - Expose decoder `num-extra-surfaces` and queue limits in the CLI/config to tune buffering for 5700x1900 panorama without code changes during field runs.【F:TARGET_DEV.MD†L11-L18】

## `my_tile_batcher/src/gstnvtilebatcher.cpp`
- Sink pad currently advertises `GST_STATIC_CAPS_ANY`, allowing system-memory buffers and triggering implicit copies into NVMM before CUDA access.【F:my_tile_batcher/src/gstnvtilebatcher.cpp†L34-L52】  
- Action plan (non-breaking):
  - Tighten sink caps to `video/x-raw(memory:NVMM),format=RGBA,width=1024,height=1024` to enforce zero-copy. Gate behind a build-time option (`NVMM_ONLY`) so existing graphs still load if upstream elements misconfigure caps.
  - Add a runtime warning when non-NVMM memory is detected to highlight misconfigured sources without failing playback.

## `my_steach/src/gstnvdsstitch.cpp`
- The stitcher maps every buffer and runs CUDA work without a fast path to bypass visualization when profiling throughput; EGL cache is refreshed periodically but work still executes every frame.【F:my_steach/src/gstnvdsstitch.cpp†L659-L722】
- Action plan (non-breaking):
  - Add a property such as `perf-mode` that skips color-correction refresh and kernel launch, forwarding NVMM frames unchanged (or using VIC blit only). Default remains current behavior.
  - Insert optional `queue` depth tuning via properties to match DeepStream guidance for decoupling stages while keeping buffers in NVMM.【F:TARGET_DEV.MD†L11-L23】

## `my_virt_cam/src/gstnvdsvirtualcam.cpp`
- Each buffer is mapped and processed through CUDA/EGL without a bypass for throughput-only runs; memory mapping happens even when no rendering is needed.【F:my_virt_cam/src/gstnvdsvirtualcam.cpp†L688-L760】
- Action plan (non-breaking):
  - Add a `bypass`/`perf-mode` property to reuse the stitched frame directly (or perform a lightweight crop) when visualization is disabled, keeping NVMM surfaces intact.
  - Emit debug stats (latency per frame, cache hits) under a `perf-stats` flag to support profiling without changing functionality.

## Cross-cutting DeepStream 7.1 practices
- For performance measurements, disable OSD, tiler, and render sinks, routing to `fakesink` to free compute budget.【F:ds_doc/7.1/text/DS_Performance.html†L1197-L1204】
- Use NVMM buffers end-to-end and branch with `tee` + NVMM-capable `queue` elements so multiple consumers do not duplicate frames.【F:TARGET_DEV.MD†L11-L23】

## Near-term roadmap (safe, incremental)
1. **Config-level toggles**: add CLI/config flags to enable performance mode across the application (display bypass, perf-mode for plugins). Default stays off to preserve current UX.
2. **Caps tightening with fallbacks**: adjust `nvtilebatcher` sink caps to NVMM with warnings instead of hard failures to maintain compatibility while steering pipelines to zero-copy.
3. **Buffering and pool tuning**: surface decoder/queue/pool sizes as properties for 5700x1900 panorama to prevent DRAM contention on Orin NX without code changes.
4. **Profiling hooks**: add optional latency counters and cache hit metrics in custom plugins to validate gains without altering frame content.

These steps align the pipeline with DeepStream 7.1 guidance while avoiding functional regressions and enabling measured, reversible performance improvements.
