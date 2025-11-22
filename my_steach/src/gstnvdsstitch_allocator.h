/**
 * @file gstnvdsstitch_allocator.h
 * @brief Custom GStreamer allocator for NVMM buffers with EGL/CUDA interop
 *
 * Provides memory allocation and management for the nvdsstitch plugin with
 * full support for:
 * - NVMM (NVIDIA Multimedia Memory) zero-copy buffers
 * - EGL image mapping for GPU surface access
 * - CUDA graphics resource registration for kernel processing
 * - Thread-safe reference counting
 *
 * The allocator manages the complete lifecycle of stitched panorama output
 * buffers, enabling efficient GPU-only processing without CPU copies.
 *
 * @author Polycube Development Team
 * @date 2025-11-21
 *
 * @see gstnvdsstitch.h for plugin integration
 * @see cuda_stitch_kernel.h for CUDA operations
 */

#ifndef __GST_NVDSSTITCH_ALLOCATOR_H__
#define __GST_NVDSSTITCH_ALLOCATOR_H__

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <vector>
#include <memory>

#ifdef __aarch64__
#include <cudaEGL.h>
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVDSSTITCH_ALLOCATOR (gst_nvdsstitch_allocator_get_type())
G_DECLARE_FINAL_TYPE(GstNvdsStitchAllocator, gst_nvdsstitch_allocator,
                     GST, NVDSSTITCH_ALLOCATOR, GstAllocator)

/**
 * @brief Memory structure for stitch plugin with full EGL support
 *
 * Extends GstMemory to include NVMM surface pointer, EGL state tracking,
 * CUDA graphics resources, and thread-safe reference counting.
 *
 * Lifecycle:
 * 1. Allocated by gst_nvdsstitch_allocator_new()
 * 2. EGL mapping via gst_nvdsstitch_memory_map_egl() (if needed)
 * 3. CUDA registration via gst_nvdsstitch_memory_register_cuda() (if needed)
 * 4. Used by CUDA kernels for stitching
 * 5. Unregistered/unmapped on buffer release
 * 6. Freed when reference count reaches zero
 */
typedef struct {
    NvBufSurface *surf;  /**< NVMM buffer surface pointer */

    /**
     * @name EGL State Flags
     * @brief Track EGL and CUDA registration status
     * @{
     */
    gboolean egl_mapped;      /**< TRUE if EGL images created */
    gboolean cuda_registered; /**< TRUE if CUDA resources registered */
    /** @} */

#ifdef __aarch64__
    /**
     * @name CUDA/EGL Interop Resources (Jetson only)
     * @brief Graphics resources for EGL surface access from CUDA
     * @{
     */
    std::vector<CUgraphicsResource> cuda_resources;  /**< CUDA graphics resources (one per surface) */
    std::vector<CUeglFrame> egl_frames;              /**< EGL frame handles (one per surface) */
    /** @} */
#endif

    std::vector<void *> frame_memory_ptrs;  /**< Frame memory pointers (dataPtr for each surface) */

    /**
     * @name Thread-Safe Reference Counting
     * @brief Prevent race conditions during buffer sharing
     * @{
     */
    gint ref_count;  /**< Reference count (atomic operations) */
    GMutex lock;     /**< Mutex for thread-safe state changes */
    /** @} */
} GstNvdsStitchMemory;

/**
 * @brief Custom allocator for nvdsstitch plugin
 *
 * GStreamer allocator subclass that creates NVMM buffers with optional
 * EGL mapping for GPU access. Maintains statistics for debugging.
 */
struct _GstNvdsStitchAllocator {
    GstAllocator parent;  /**< Base GstAllocator */

    guint width;   /**< Buffer width in pixels */
    guint height;  /**< Buffer height in pixels */
    guint gpu_id;  /**< CUDA device ID */

    GstVideoFormat output_format;  /**< Output color format (RGBA or NV12) */

    gboolean use_egl;  /**< Enable EGL mapping (TRUE for Jetson aarch64) */

    /**
     * @name Statistics for Debugging
     * @brief Track allocations and deallocations
     * @{
     */
    guint total_allocated;  /**< Total buffers allocated since creation */
    guint total_freed;      /**< Total buffers freed since creation */
    /** @} */
};

/* ========== API FUNCTIONS ========== */

/**
 * @brief Create new nvdsstitch allocator
 *
 * Allocates and initializes a custom GStreamer allocator for NVMM buffers
 * with specified dimensions, GPU device, and color format.
 *
 * @param[in] width Output panorama width in pixels (e.g., 5700)
 * @param[in] height Output panorama height in pixels (e.g., 1900)
 * @param[in] gpu_id CUDA device ID (typically 0)
 * @param[in] output_format Color format (GST_VIDEO_FORMAT_RGBA or GST_VIDEO_FORMAT_NV12)
 *
 * @return GstAllocator* Allocator instance (caller must unref when done)
 * @retval NULL Allocation failed
 *
 * @note Caller must call gst_object_unref() to destroy allocator
 * @note Allocator automatically detects aarch64 and enables EGL support
 * @note Format affects buffer size: RGBA=4 bytes/pixel, NV12=1.5 bytes/pixel
 *
 * Example usage:
 * @code
 * GstAllocator *allocator = gst_nvdsstitch_allocator_new(5700, 1900, 0, GST_VIDEO_FORMAT_RGBA);
 * // ... use allocator ...
 * gst_object_unref(allocator);
 * @endcode
 */
GstAllocator *gst_nvdsstitch_allocator_new(guint width, guint height, guint gpu_id, GstVideoFormat output_format);

/**
 * @brief Get GstNvdsStitchMemory from GstBuffer
 *
 * Extracts custom memory structure from GStreamer buffer.
 * Validates that buffer was allocated by nvdsstitch allocator.
 *
 * @param[in] buffer GStreamer buffer (must be from nvdsstitch allocator)
 *
 * @return GstNvdsStitchMemory* Memory structure pointer
 * @retval NULL Buffer is NULL or not from nvdsstitch allocator
 *
 * @note Does not increase reference count
 * @warning Returned pointer is only valid while buffer is alive
 */
GstNvdsStitchMemory *gst_nvdsstitch_buffer_get_memory(GstBuffer *buffer);

/* ========== EGL MANAGEMENT ========== */

/**
 * @brief Map NVMM surface to EGL images
 *
 * Creates EGL images from NVMM surface for GPU access.
 * Required before CUDA registration on Jetson platforms.
 *
 * @param[in,out] mem Memory structure to map
 *
 * @return gboolean Success status
 * @retval TRUE EGL mapping succeeded (or already mapped)
 * @retval FALSE Mapping failed (EGL errors logged)
 *
 * @note Idempotent - safe to call multiple times
 * @note Only functional on aarch64 (Jetson) platforms
 * @note Must call before gst_nvdsstitch_memory_register_cuda()
 *
 * @see gst_nvdsstitch_memory_unmap_egl
 * @see gst_nvdsstitch_memory_register_cuda
 */
gboolean gst_nvdsstitch_memory_map_egl(GstNvdsStitchMemory *mem);

/**
 * @brief Unmap EGL images from NVMM surface
 *
 * Destroys EGL images and releases EGL resources.
 * Call before buffer destruction or when GPU access no longer needed.
 *
 * @param[in,out] mem Memory structure to unmap
 *
 * @note Idempotent - safe to call even if not mapped
 * @note Automatically unregisters CUDA resources if registered
 *
 * @see gst_nvdsstitch_memory_map_egl
 */
void gst_nvdsstitch_memory_unmap_egl(GstNvdsStitchMemory *mem);

/**
 * @brief Register CUDA graphics resources for surface access
 *
 * Registers EGL frames as CUDA graphics resources, enabling CUDA kernels
 * to directly access surface memory.
 *
 * @param[in,out] mem Memory structure to register
 *
 * @return gboolean Success status
 * @retval TRUE CUDA registration succeeded (or already registered)
 * @retval FALSE Registration failed (CUDA errors logged)
 *
 * @note Idempotent - safe to call multiple times
 * @note Requires prior EGL mapping (gst_nvdsstitch_memory_map_egl)
 * @note Only functional on aarch64 (Jetson) platforms
 *
 * @see gst_nvdsstitch_memory_unregister_cuda
 * @see gst_nvdsstitch_memory_map_egl
 */
gboolean gst_nvdsstitch_memory_register_cuda(GstNvdsStitchMemory *mem);

/**
 * @brief Unregister CUDA graphics resources
 *
 * Unregisters CUDA graphics resources, releasing CUDA references to EGL frames.
 * Call before EGL unmapping or when CUDA access no longer needed.
 *
 * @param[in,out] mem Memory structure to unregister
 *
 * @note Idempotent - safe to call even if not registered
 * @note Must be called before gst_nvdsstitch_memory_unmap_egl()
 *
 * @see gst_nvdsstitch_memory_register_cuda
 */
void gst_nvdsstitch_memory_unregister_cuda(GstNvdsStitchMemory *mem);

G_END_DECLS

#endif /* __GST_NVDSSTITCH_ALLOCATOR_H__ */
