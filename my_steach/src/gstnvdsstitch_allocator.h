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

/* Структура памяти для stitch плагина с полной поддержкой EGL */
typedef struct {
    NvBufSurface *surf;
    
    /* Флаги состояния EGL */
    gboolean egl_mapped;
    gboolean cuda_registered;
    
#ifdef __aarch64__
    /* Для SURFACE_ARRAY: CUDA ресурсы для EGL interop */
    std::vector<CUgraphicsResource> cuda_resources;
    std::vector<CUeglFrame> egl_frames;
#endif
    
    /* Указатели на память кадров */
    std::vector<void *> frame_memory_ptrs;
    
    /* Reference counting для безопасного управления */
    gint ref_count;
    GMutex lock;
} GstNvdsStitchMemory;

struct _GstNvdsStitchAllocator {
    GstAllocator parent;
    
    guint width;
    guint height;
    guint gpu_id;
    
    /* Настройки EGL */
    gboolean use_egl;
    
    /* Статистика для отладки */
    guint total_allocated;
    guint total_freed;
};

/* API функции */
GstAllocator *gst_nvdsstitch_allocator_new(guint width, guint height, guint gpu_id);
GstNvdsStitchMemory *gst_nvdsstitch_buffer_get_memory(GstBuffer *buffer);

/* EGL управление */
gboolean gst_nvdsstitch_memory_map_egl(GstNvdsStitchMemory *mem);
void gst_nvdsstitch_memory_unmap_egl(GstNvdsStitchMemory *mem);
gboolean gst_nvdsstitch_memory_register_cuda(GstNvdsStitchMemory *mem);
void gst_nvdsstitch_memory_unregister_cuda(GstNvdsStitchMemory *mem);

G_END_DECLS

#endif /* __GST_NVDSSTITCH_ALLOCATOR_H__ */