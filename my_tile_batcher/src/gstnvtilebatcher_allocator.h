#ifndef __GST_NVTILEBATCHER_ALLOCATOR_H__
#define __GST_NVTILEBATCHER_ALLOCATOR_H__

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <nvbufsurface.h>

#ifdef __aarch64__
#include <cudaEGL.h>
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVTILEBATCHER_ALLOCATOR (gst_nvtilebatcher_allocator_get_type())
G_DECLARE_FINAL_TYPE(GstNvTileBatcherAllocator, gst_nvtilebatcher_allocator, 
                     GST, NVTILEBATCHER_ALLOCATOR, GstAllocator)

typedef struct {
    NvBufSurface *surf;
    gboolean egl_mapped;
    
#ifdef __aarch64__
    CUgraphicsResource cuda_resources[6];
    CUeglFrame egl_frames[6];
    void* batch_cuda_ptr;
#endif
    
    gint ref_count;
    GMutex lock;
} GstNvTileBatcherMemory;

struct _GstNvTileBatcherAllocator {
    GstAllocator parent;
    guint gpu_id;
    guint total_allocated;
    guint total_freed;
};

GstAllocator *gst_nvtilebatcher_allocator_new(guint gpu_id);
GstNvTileBatcherMemory *gst_nvtilebatcher_buffer_get_memory(GstBuffer *buffer);

G_END_DECLS

#endif