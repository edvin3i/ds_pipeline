#ifndef __GST_NVDSVIRTUALCAM_ALLOCATOR_H__
#define __GST_NVDSVIRTUALCAM_ALLOCATOR_H__

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <nvbufsurface.h>
#include <vector>
#include <memory>

#ifdef __aarch64__
#include <cudaEGL.h>
#endif

G_BEGIN_DECLS

// ИЗМЕНЕНО: GstNvdsStitchAllocator -> GstNvdsVirtualCamAllocator
#define GST_TYPE_NVDSVIRTUALCAM_ALLOCATOR (gst_nvdsvirtualcam_allocator_get_type())
G_DECLARE_FINAL_TYPE(GstNvdsVirtualCamAllocator, gst_nvdsvirtualcam_allocator, 
                     GST, NVDSVIRTUALCAM_ALLOCATOR, GstAllocator)

// ИЗМЕНЕНО: GstNvdsStitchMemory -> GstNvdsVirtualCamMemory
typedef struct {
    NvBufSurface *surf;
    gboolean egl_mapped;
    gboolean cuda_registered;
    
#ifdef __aarch64__
    std::vector<CUgraphicsResource> cuda_resources;
    std::vector<CUeglFrame> egl_frames;
#endif
    
    std::vector<void *> frame_memory_ptrs;
    gint ref_count;
    GMutex lock;
} GstNvdsVirtualCamMemory;

struct _GstNvdsVirtualCamAllocator {
    GstAllocator parent;
    guint width;
    guint height;
    guint gpu_id;
    gboolean use_egl;
    guint total_allocated;
    guint total_freed;
};

// ИЗМЕНЕНО: все функции теперь используют новые имена
GstAllocator *gst_nvdsvirtualcam_allocator_new(guint width, guint height, guint gpu_id);
GstNvdsVirtualCamMemory *gst_nvdsvirtualcam_buffer_get_memory(GstBuffer *buffer);

// EGL управление - ИЗМЕНЕНО имена
gboolean gst_nvdsvirtualcam_memory_map_egl(GstNvdsVirtualCamMemory *mem);
void gst_nvdsvirtualcam_memory_unmap_egl(GstNvdsVirtualCamMemory *mem);
gboolean gst_nvdsvirtualcam_memory_register_cuda(GstNvdsVirtualCamMemory *mem);
void gst_nvdsvirtualcam_memory_unregister_cuda(GstNvdsVirtualCamMemory *mem);

G_END_DECLS

#endif /* __GST_NVDSVIRTUALCAM_ALLOCATOR_H__ */