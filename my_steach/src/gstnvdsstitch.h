// gstnvdsstitch.h - Header file for panorama mode stitching plugin
#ifndef __GST_NVDS_STITCH_H__
#define __GST_NVDS_STITCH_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
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

typedef struct {
    gpointer egl_image;
    CUgraphicsResource cuda_resource;
    CUeglFrame egl_frame;
    gboolean registered;
    guint64 last_access_frame;
} EGLResourceCacheEntry;

typedef struct {
    gint left_index;
    gint right_index;
} FrameIndices;

#define FIXED_OUTPUT_POOL_SIZE 8

struct _GstNvdsStitch {
    GstBaseTransform element;

    // Intermediate buffers for left/right camera frames
    GstBufferPool *intermediate_pool;
    GstBuffer *intermediate_left;
    GstBuffer *intermediate_right;
    NvBufSurface *intermediate_left_surf;
    NvBufSurface *intermediate_right_surf;
    
    // Source IDs
    guint left_source_id;
    guint right_source_id;

    // Output dimensions (set dynamically via properties, e.g. 5700x1900)
    guint output_width;
    guint output_height;

    // GPU
    guint gpu_id;

    // Not used in panorama mode, kept for backward compatibility
    guint overlap;
    guint crop_top;
    guint crop_bottom;
    guint crop_sides;


    // ========== COLOR CORRECTION (ASYNC) ==========
    // Hardware-sync-aware color correction for overlap region
    cudaStream_t color_analysis_stream;        // Low-priority stream for async analysis
    cudaEvent_t color_analysis_event;          // Event for analysis completion
    float* d_color_analysis_buffer;            // Device buffer for reduction results (9 floats)
    ColorCorrectionFactors current_factors;    // Currently applied factors
    ColorCorrectionFactors pending_factors;    // Factors from latest analysis (not yet applied)
    guint frame_count;                         // Total frames processed
    guint last_color_frame;                    // Frame number when last analysis was triggered
    gboolean color_analysis_pending;           // TRUE if analysis in flight

    // Color correction properties (configurable)
    gboolean enable_color_correction;          // Master enable (property)
    float overlap_size;                        // Overlap region size in degrees (property)
    guint color_update_interval;               // Analysis interval in frames (property)
    float color_smoothing_factor;              // Temporal smoothing α (property)
    float spatial_falloff;                     // Vignetting compensation exponent (property)
    gboolean enable_gamma;                     // Enable gamma correction (property)

    // ========== ERROR HANDLING & RECOVERY ==========
    // Failure tracking for graceful degradation and automatic recovery
    guint color_correction_consecutive_failures;   // Count of consecutive failures (0-3+)
    gboolean color_correction_permanently_disabled; // TRUE if disabled after 3+ failures
    GstClockTime last_color_failure_time;          // Timestamp of last failure (for recovery logic)


    // Buffer management
    GstBuffer *current_input;
    GstBufferPool *output_pool;
    gboolean pool_configured;

    // LUT maps and blending weights in GPU memory
    float *warp_left_x_gpu;   // Using legacy names for compatibility
    float *warp_left_y_gpu;
    float *warp_right_x_gpu;
    float *warp_right_y_gpu;
    float *weight_left_gpu;
    float *weight_right_gpu;
    
    gboolean warp_maps_loaded;
    
    // CUDA stream
    cudaStream_t cuda_stream;

    // CUDA event for frame synchronization (non-static to avoid leak)
    cudaEvent_t frame_complete_event;

    // Kernel configuration
    StitchKernelConfig kernel_config;

    // EGL resource management
    gboolean use_egl;
    GHashTable *egl_resource_cache;
    GMutex egl_lock;
    guint egl_map_count;
    guint egl_register_count;
    guint egl_cache_hits;

    // Pipeline state
    GstFlowReturn last_flow_ret;
    guint64 current_frame_number;
    FrameIndices cached_indices;

    // Fixed output buffer pool
    struct {
        GstBuffer* buffers[FIXED_OUTPUT_POOL_SIZE];
        NvBufSurface* surfaces[FIXED_OUTPUT_POOL_SIZE];
        gboolean registered[FIXED_OUTPUT_POOL_SIZE];
        gint current_index;
        GMutex mutex;
    } output_pool_fixed;
};

G_END_DECLS

#endif /* __GST_NVDS_STITCH_H__ */
