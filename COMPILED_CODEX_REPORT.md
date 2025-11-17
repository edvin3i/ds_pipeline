# COMPILED CODEX REPORT

This document cross-checks the independent CODEX CPU assessment with the existing `Performance_report.MD`. Items are grouped by whether both reports agree, or whether a finding appears in only one report. Citations point to the exact source for each statement.

## 1. Areas of Agreement
1. **BufferManager deep copies saturate CPU/DRAM.** Both reports flag `on_new_sample()` because every incoming panorama frame is duplicated with `copy_deep()`, consuming tens of milliseconds and ~1.3 GB/s of shared LPDDR5 bandwidth on Jetson Orin NX.【F:CODEX_report.MD†L5-L11】【F:Performance_report.MD†L650-L699】
2. **Linear scans inside `_on_appsrc_need_data()` waste CPU.** Each send operation linearly walks the 7‑second deque and performs timestamp bookkeeping under a lock, which both reports describe as O(n) work on the streaming thread.【F:CODEX_report.MD†L9-L12】【F:Performance_report.MD†L700-L741】
3. **Probe callbacks (analysis + display) are overloaded with Python/Numpy logic.** The CODEX report highlights the heavy YOLO post-processing, NMS, and drawing paths; the existing performance report quantifies the same pad probe overheads (metadata iteration, history lookups, rectangle formatting) for both the analysis and display/virtual-camera probes.【F:CODEX_report.MD†L13-L24】【F:Performance_report.MD†L180-L398】
4. **History management and trajectory filtering repeatedly sort dictionaries.** Both documents call out `HistoryManager.get_detection_for_timestamp()` and `_process_future_history()` plus the nested loops in `TrajectoryFilter` as CPU-intensive due to repeated merges, sorts, and gap interpolation on every detection update.【F:CODEX_report.MD†L17-L20】【F:Performance_report.MD†L400-L563】
5. **Center-of-mass smoothing in `display_probe.py` is expensive.** Each frame recomputes medians and weighted averages over ~7 s of history; both reports describe this as a 5‑10 ms hot path that should be cached or moved out of the sink pad probe.【F:CODEX_report.MD†L21-L24】【F:Performance_report.MD†L568-L642】

## 2. Additional Findings Documented in `Performance_report.MD`
1. **Tensor extraction and YOLO postprocessing dominate analysis CPU time.** The performance report quantifies 60‑75 ms per analyzed frame from `get_tensor_as_numpy()` and `postprocess_yolo_output()`, noting the multiple full-buffer copies, NumPy transposes, and mask allocations that were not itemized in the CODEX draft.【F:Performance_report.MD†L63-L178】
2. **NMS complexity and metadata iteration specifics.** Claude’s report details the O(n²) IoU loops in `apply_nms()` plus the exact metadata traversal pattern that compounds Python↔C++ calls every frame, extending the probe-overhead narrative with concrete complexity data.【F:Performance_report.MD†L180-L398】
3. **Lock contention & GIL impact.** The prior report measures how `history_lock` and `buffer_lock` stay held for hundreds of milliseconds per second, and how the Python GIL magnifies the contention—topics only implicitly hinted at in CODEX.【F:Performance_report.MD†L777-L879】
4. **Interpreter/ logging overhead and duplicate COM code.** It calls out modulo-based logging, redundant center-of-mass implementations, and Python dictionary churn as separate optimization targets with estimated CPU savings.【F:Performance_report.MD†L640-L900】

## 3. Additional Findings Unique to the CODEX Report
1. **Panorama tile saver synchronous JPEG pipeline.** CODEX notes that `sliser/panorama_tiles_saver.py` pulls frames into NumPy and writes panoramas plus six 1024×1024 tiles synchronously under GLib, which can starve the pipeline; this path was outside the scope of the earlier performance report.【F:CODEX_report.MD†L25-L28】

## 4. Consolidated Priorities
1. **Stop copying large NVMM buffers and tensors.** Combining both reports, the highest-impact change remains eliminating deep copies in `BufferManager` and `get_tensor_as_numpy()`—together they can reclaim several hundred milliseconds per second of CPU time.【F:CODEX_report.MD†L5-L11】【F:Performance_report.MD†L63-L178】【F:Performance_report.MD†L650-L699】
2. **Thin out probe callbacks and cache history products.** The shared findings on probes, history sorting, and center-of-mass smoothing suggest moving heavy math into batched workers or cached utilities so pad probes only perform metadata routing.【F:CODEX_report.MD†L13-L24】【F:Performance_report.MD†L180-L642】
3. **Address locking/logging overhead next.** Claude’s deeper look at lock hold times, duplicate COM code, and eager logging adds actionable follow-ups once the structural issues above are resolved.【F:Performance_report.MD†L640-L900】
