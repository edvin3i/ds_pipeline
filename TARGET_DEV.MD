# Target Device Notes

This document summarizes the key hardware and DeepStream SDK 7.1 behaviors needed for development on the NVIDIA Jetson Orin NX 16GB target. It also records project-specific context, including the panoramic output dimensions (5700x1900).

## Jetson Orin NX 16GB hardware facts
- Ampere GPU with 1024 CUDA cores and 32 Tensor Cores, delivering ~100 TOPS (INT8, sparse) at up to ~918 MHz clock speeds.
- 8-core ARM Cortex-A78AE CPU cluster running up to ~2 GHz with 4 MB shared L3 cache.
- 16 GB LPDDR5 on a 128-bit interface (~102 GB/s peak), exposed as unified DRAM shared by CPU and GPU.
- GPU-visible memory equals available system DRAM; plan allocations to avoid CPU-side usage starving GPU buffers.

## Memory model and implications
- Unified memory provides hardware cache coherency between CPU and GPU, letting GPU observe CPU updates (and vice versa) without explicit cache flushes. Avoid unnecessary CUDA Unified Memory allocations in favor of direct NVMM/pinned buffers to reduce implicit migrations and synchronization overhead.
- Use NVIDIA NVMM buffers end-to-end to keep frames on the GPU. Standard GStreamer elements that drop back to system memory introduce copies; prefer NVMM-aware plugins (NVDEC, `nvvideoconvert`, `nvstreammux`, `nvdsosd`, etc.).

## Pipeline structure guidance
- Branch streams with `tee` so multiple consumers share the same NVMM frame without duplication; `tee` increases the reference count instead of copying the buffer. Place a `queue` immediately after each `tee` branch to decouple execution while preserving NVMM buffers.
- Keep `queue` elements before any conversion that would force CPU memory; when the upstream buffers are NVMM, the queue stores NVMM-backed frames, enabling GPU-resident buffering (useful for multi-second delays).
- Configure decoders and downstream queues with sufficient surfaces for buffering (e.g., adjust `num-extra-surfaces` on decoders and queue limits) when implementing long delays or panorama fan-out.

## DeepStream SDK 7.1 reminders
- DeepStream is a GStreamer-based toolkit optimized for zero-copy graphs using GPU/DLA accelerators; it ingests camera/RTSP/file inputs, decodes with NVDEC, optionally preprocesses, batches via `nvstreammux`, and runs inference/OSD/sinks through hardware-accelerated plugins.
- For performance measurements, disable OSD, tiler, and render sinks when visualization is unnecessary: set `[osd] enable=0`, `[tiled-display] enable=0`, and route sinks to `fakesink` in config files to free compute budget.

## Project-specific
- Panorama output must target 5700x1900; ensure compositors and sinks are sized for this resolution and that batching/memory allocations account for the larger frame.
