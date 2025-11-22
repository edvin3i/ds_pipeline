/**
 * @file gstnvdsstitch.h
 * @brief Main header for nvdsstitch GStreamer plugin (panorama stitching)
 *
 * Defines the plugin structure, properties, and state for the nvdsstitch
 * GStreamer element. This plugin performs real-time 360° panorama stitching
 * from dual fisheye camera inputs using GPU-accelerated LUT-based warping.
 *
 * Plugin class: Transform (1-to-1 element)
 * Input: Dual camera frames (nvstreammux batch=2, RGBA 3840×2160)
 * Output: Stitched panorama (RGBA, configurable dimensions, e.g., 5700×1900)
 *
 * Key features:
 * - Zero-copy NVMM processing (GPU-only, no CPU copies)
 * - 2-phase async color correction with hardware frame sync
 * - 3-strike error handling for graceful degradation
 * - Fixed buffer pool (8 buffers) to prevent memory fragmentation
 * - EGL interop for Jetson platforms
 *
 * Performance (Jetson Orin NX 16GB):
 * - Stitching throughput: 51-55 FPS
 * - Pipeline latency: ~90ms (well within 100ms budget)
 * - GPU load: ~70% (healthy margin)
 *
 * @author Polycube Development Team
 * @date 2025-11-21
 *
 * @see cuda_stitch_kernel.h for CUDA operations
 * @see nvdsstitch_config.h for configuration constants
 * @see gstnvdsstitch_allocator.h for memory management
 */

#ifndef __GST_NVDS_STITCH_H__
#define __GST_NVDS_STITCH_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <cuda_runtime_api.h>
#include <nvbufsurface.h>

#ifdef __aarch64__
#include <cudaEGL.h>
#endif

#include "cuda_stitch_kernel.h"
#include "nvdsstitch_config.h"

G_BEGIN_DECLS

#define GST_TYPE_NVDS_STITCH (gst_nvds_stitch_get_type())
G_DECLARE_FINAL_TYPE(GstNvdsStitch, gst_nvds_stitch, GST, NVDS_STITCH, GstBaseTransform)

/**
 * @brief EGL resource cache entry for surface access optimization
 *
 * Caches EGL images and CUDA graphics resources to avoid repeated
 * registration overhead. Each entry corresponds to one NVMM surface.
 *
 * @note Only used on aarch64 (Jetson) platforms with EGL support
 */
typedef struct {
    gpointer egl_image;                 /**< EGL image handle (opaque pointer) */
    CUgraphicsResource cuda_resource;   /**< CUDA graphics resource for surface access */
    CUeglFrame egl_frame;               /**< EGL frame descriptor */
    gboolean registered;                /**< TRUE if CUDA resource is registered */
    guint64 last_access_frame;          /**< Frame number of last access (for LRU eviction) */
} EGLResourceCacheEntry;

/**
 * @brief Frame indices for left and right cameras
 *
 * Tracks which frames in the batch correspond to each camera.
 * Cached to avoid repeated detection.
 */
typedef struct {
    gint left_index;   /**< Batch index for left camera (typically 0) */
    gint right_index;  /**< Batch index for right camera (typically 1) */
} FrameIndices;

/**
 * @brief Fixed output buffer pool size
 *
 * Number of pre-allocated output buffers (8 buffers).
 * Prevents memory fragmentation and ensures stable performance.
 */
#define FIXED_OUTPUT_POOL_SIZE 8

/**
 * @brief Main plugin structure for nvdsstitch
 *
 * Contains all state, configuration, GPU resources, and buffers for the
 * panorama stitching plugin. Organized into logical sections:
 * - Intermediate buffers (left/right camera frames)
 * - Output dimensions and GPU settings
 * - Color correction (async 2-phase)
 * - Error handling & recovery
 * - Buffer management
 * - LUT maps (GPU memory)
 * - CUDA resources
 * - EGL management (Jetson only)
 * - Pipeline state
 * - Fixed output buffer pool
 */
struct _GstNvdsStitch {
    GstBaseTransform element;  /**< Base GstBaseTransform instance */

    /**
     * @name Intermediate Buffers
     * @brief Temporary storage for left/right camera frames
     * @{
     */
    GstBufferPool *intermediate_pool;       /**< Buffer pool for intermediate frames */
    GstBuffer *intermediate_left;           /**< Left camera frame buffer */
    GstBuffer *intermediate_right;          /**< Right camera frame buffer */
    NvBufSurface *intermediate_left_surf;   /**< Left camera NVMM surface pointer */
    NvBufSurface *intermediate_right_surf;  /**< Right camera NVMM surface pointer */
    /** @} */

    /**
     * @name Source Identification
     * @brief Camera source IDs from nvstreammux
     * @{
     */
    guint left_source_id;   /**< Source ID for left camera (property, default: 0) */
    guint right_source_id;  /**< Source ID for right camera (property, default: 1) */
    /** @} */

    /**
     * @name Output Dimensions
     * @brief Panorama output size (set dynamically via properties)
     * @{
     */
    guint output_width;   /**< Output panorama width in pixels (property, e.g., 5700) */
    guint output_height;  /**< Output panorama height in pixels (property, e.g., 1900) */
    /** @} */

    /**
     * @name Output Format
     * @brief Color format for panorama output
     * @{
     */
    GstVideoFormat output_format;  /**< Output color format (RGBA or NV12, property) */
    /** @} */

    guint gpu_id;  /**< CUDA device ID (property, default: 0) */

    /**
     * @name Legacy Parameters
     * @brief Reserved for backward compatibility (not used in panorama mode)
     * @{
     */
    guint overlap;       /**< Reserved */
    guint crop_top;      /**< Reserved */
    guint crop_bottom;   /**< Reserved */
    guint crop_sides;    /**< Reserved */
    /** @} */

    /**
     * @name Color Correction (Async 2-Phase)
     * @brief Hardware-sync-aware color correction for overlap region
     *
     * 2-phase async color correction system:
     * - Phase 1: Launch analysis kernel on low-priority stream (async)
     * - Phase 2: When analysis completes, apply factors to stitching (sync)
     *
     * Prevents pipeline stalls while adapting to lighting changes.
     * @{
     */
    cudaStream_t color_analysis_stream;         /**< Low-priority CUDA stream for async analysis */
    cudaEvent_t color_analysis_event;           /**< Event for analysis completion detection */
    float* d_color_analysis_buffer;             /**< Device buffer for reduction results (9 floats) */
    ColorCorrectionFactors current_factors;     /**< Currently applied factors (active in kernels) */
    ColorCorrectionFactors pending_factors;     /**< Factors from latest analysis (not yet applied) */
    guint frame_count;                          /**< Total frames processed since start */
    guint last_color_frame;                     /**< Frame number when last analysis was triggered */
    gboolean color_analysis_pending;            /**< TRUE if analysis kernel is in flight */

    /** @name Color Correction Properties (Configurable)
     * @{
     */
    gboolean enable_color_correction;  /**< Master enable for color correction (property) */
    float overlap_size;                /**< Overlap region size in degrees (property, 5-15 range) */
    guint color_update_interval;       /**< Analysis interval in frames (property, 0-120 range, 0=disable) */
    float color_smoothing_factor;      /**< Temporal smoothing α (property, 0.0-1.0, default: 0.15) */
    float spatial_falloff;             /**< Vignetting compensation exponent (property, default: 2.0) */
    gboolean enable_gamma;             /**< Enable gamma correction (property, default: TRUE) */
    /** @} */
    /** @} */

    /**
     * @name Error Handling & Recovery (3-Strike System)
     * @brief Graceful degradation for color correction failures
     *
     * Tracks consecutive failures to prevent pipeline crashes.
     * After 3 failures, color correction is permanently disabled.
     * @{
     */
    guint color_correction_consecutive_failures;     /**< Count of consecutive failures (0-3+) */
    gboolean color_correction_permanently_disabled;  /**< TRUE if disabled after 3+ failures */
    GstClockTime last_color_failure_time;            /**< Timestamp of last failure (for recovery logic) */
    /** @} */

    /**
     * @name Buffer Management
     * @brief GStreamer buffer pools and configuration
     * @{
     */
    GstBuffer *current_input;      /**< Current input buffer being processed */
    GstBufferPool *output_pool;    /**< Output buffer pool (legacy, replaced by fixed pool) */
    gboolean pool_configured;      /**< TRUE if buffer pool is configured and active */
    /** @} */

    /**
     * @name LUT Maps (GPU Memory)
     * @brief Precomputed Look-Up Tables for panorama warping
     *
     * 6 LUT arrays loaded from binary files:
     * - 2 coordinate maps per camera (X, Y pixel coordinates)
     * - 1 weight map per camera (blending weights, sum=1.0)
     *
     * Total size: ~24 MB for 5700×1900 panorama
     * @{
     */
    float *warp_left_x_gpu;    /**< Left camera X coordinate LUT (device memory) */
    float *warp_left_y_gpu;    /**< Left camera Y coordinate LUT (device memory) */
    float *warp_right_x_gpu;   /**< Right camera X coordinate LUT (device memory) */
    float *warp_right_y_gpu;   /**< Right camera Y coordinate LUT (device memory) */
    float *weight_left_gpu;    /**< Left camera blending weights (device memory) */
    float *weight_right_gpu;   /**< Right camera blending weights (device memory) */

    gboolean warp_maps_loaded;  /**< TRUE if LUT maps successfully loaded from disk */
    /** @} */

    /**
     * @name CUDA Resources
     * @brief CUDA streams, events, and kernel configuration
     * @{
     */
    cudaStream_t cuda_stream;          /**< Main CUDA stream for stitching kernels */
    cudaEvent_t frame_complete_event;  /**< Event for frame synchronization (prevents leaks) */
    StitchKernelConfig kernel_config;  /**< Kernel launch configuration (dimensions, pitch) */
    /** @} */

    /**
     * @name EGL Resource Management (Jetson Only)
     * @brief EGL image cache for optimized surface access
     *
     * On Jetson platforms (aarch64), EGL images provide efficient
     * GPU surface access. Caching avoids repeated registration overhead.
     * @{
     */
    gboolean use_egl;                  /**< TRUE if EGL support is enabled (aarch64 only) */
    GHashTable *egl_resource_cache;    /**< Hash table: surface pointer → EGLResourceCacheEntry */
    GMutex egl_lock;                   /**< Mutex for thread-safe cache access */
    guint egl_map_count;               /**< Total EGL mappings performed (statistics) */
    guint egl_register_count;          /**< Total CUDA registrations performed (statistics) */
    guint egl_cache_hits;              /**< Cache hits (avoided re-registration) */
    /** @} */

    /**
     * @name Pipeline State
     * @brief Runtime state and frame tracking
     * @{
     */
    GstFlowReturn last_flow_ret;       /**< Last GstFlowReturn from transform */
    guint64 current_frame_number;      /**< Current frame number (for debugging/logging) */
    FrameIndices cached_indices;       /**< Cached left/right frame indices in batch */
    /** @} */

    /**
     * @name Fixed Output Buffer Pool
     * @brief Pre-allocated output buffers (prevents memory fragmentation)
     *
     * Fixed pool of 8 NVMM buffers, rotated round-robin.
     * Eliminates allocation overhead and ensures stable memory usage.
     * @{
     */
    struct {
        GstBuffer* buffers[FIXED_OUTPUT_POOL_SIZE];     /**< Pre-allocated output buffers */
        NvBufSurface* surfaces[FIXED_OUTPUT_POOL_SIZE]; /**< NVMM surface pointers */
        gboolean registered[FIXED_OUTPUT_POOL_SIZE];    /**< TRUE if buffer's surfaces are CUDA-registered */
        gint current_index;                             /**< Current buffer index (round-robin) */
        GMutex mutex;                                   /**< Mutex for thread-safe buffer access */
    } output_pool_fixed;  /**< Fixed output buffer pool */
    /** @} */
};

G_END_DECLS

#endif /* __GST_NVDS_STITCH_H__ */
