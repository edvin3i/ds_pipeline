#ifndef __GST_NVDSSTITCH_ALLOCATOR_H__
#define __GST_NVDSSTITCH_ALLOCATOR_H__

#include <gst/gst.h>
#include <gst/gstallocator.h>
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

/* Memory structure for stitch plugin with full EGL support */
typedef struct {
    NvBufSurface *surf;

    /* EGL state flags */
    gboolean egl_mapped;
    gboolean cuda_registered;

#ifdef __aarch64__
    /* For SURFACE_ARRAY: CUDA resources for EGL interop */
    std::vector<CUgraphicsResource> cuda_resources;
    std::vector<CUeglFrame> egl_frames;
#endif

    /* Frame memory pointers */
    std::vector<void *> frame_memory_ptrs;

    /* Reference counting for safe memory management */
    gint ref_count;
    GMutex lock;
} GstNvdsStitchMemory;

struct _GstNvdsStitchAllocator {
    GstAllocator parent;

    guint width;
    guint height;
    guint gpu_id;

    /* EGL settings */
    gboolean use_egl;

    /* Statistics for debugging */
    guint total_allocated;
    guint total_freed;
};

/* API functions */
GstAllocator *gst_nvdsstitch_allocator_new(guint width, guint height, guint gpu_id);
GstNvdsStitchMemory *gst_nvdsstitch_buffer_get_memory(GstBuffer *buffer);

/* EGL management */
gboolean gst_nvdsstitch_memory_map_egl(GstNvdsStitchMemory *mem);
void gst_nvdsstitch_memory_unmap_egl(GstNvdsStitchMemory *mem);
gboolean gst_nvdsstitch_memory_register_cuda(GstNvdsStitchMemory *mem);
void gst_nvdsstitch_memory_unregister_cuda(GstNvdsStitchMemory *mem);

G_END_DECLS

#endif /* __GST_NVDSSTITCH_ALLOCATOR_H__ */