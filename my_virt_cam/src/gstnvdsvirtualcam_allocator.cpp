/*
 * gstnvdsvirtualcam_allocator.cpp - Allocator для nvdsvirtualcam плагина с полной поддержкой EGL
 * 
 * Этот allocator управляет GPU памятью для выходных буферов виртуальной камеры
 * и обеспечивает правильную работу с EGL на платформе Jetson
 */

#include "gstnvdsvirtualcam_allocator.h"
#include <nvbufsurface.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <atomic>

GST_DEBUG_CATEGORY_STATIC(gst_nvdsvirtualcam_allocator_debug);
#define GST_CAT_DEFAULT gst_nvdsvirtualcam_allocator_debug

#define GST_NVDSVIRTUALCAM_MEMORY_TYPE "nvdsvirtualcam"

// Внутренняя структура для GstMemory
typedef struct {
    GstMemory mem;
    GstNvdsVirtualCamMemory *virtualcam_mem;
} GstNvdsVirtualCamMem;

G_DEFINE_TYPE(GstNvdsVirtualCamAllocator, gst_nvdsvirtualcam_allocator, GST_TYPE_ALLOCATOR);

/* ============================================================================
 * Вспомогательные функции для работы с EGL
 * ============================================================================ */

/**
 * gst_nvdsvirtualcam_memory_map_egl:
 * @mem: Указатель на GstNvdsVirtualCamMemory
 * 
 * Выполняет EGL mapping для NvBufSurface. На Jetson это необходимо
 * для доступа к памяти через CUDA.
 * 
 * Returns: TRUE при успехе, FALSE при ошибке
 */
gboolean gst_nvdsvirtualcam_memory_map_egl(GstNvdsVirtualCamMemory *mem)
{
    g_return_val_if_fail(mem != NULL, FALSE);
    g_return_val_if_fail(mem->surf != NULL, FALSE);
    
    g_mutex_lock(&mem->lock);
    
    if (mem->egl_mapped) {
        GST_DEBUG("EGL already mapped for memory %p", mem);
        g_mutex_unlock(&mem->lock);
        return TRUE;
    }
    
#ifdef __aarch64__
    if (mem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        // Проверяем, не сделан ли уже маппинг
        gboolean already_mapped = TRUE;
        for (guint i = 0; i < mem->surf->numFilled; i++) {
            if (mem->surf->surfaceList[i].mappedAddr.eglImage == NULL) {
                already_mapped = FALSE;
                break;
            }
        }
        
        if (!already_mapped) {
            GST_DEBUG("Performing EGL mapping for surface %p", mem->surf);
            if (NvBufSurfaceMapEglImage(mem->surf, -1) != 0) {
                GST_ERROR("Failed to map EGL image for surface %p", mem->surf);
                g_mutex_unlock(&mem->lock);
                return FALSE;
            }
            GST_DEBUG("EGL mapping successful");
        } else {
            GST_DEBUG("Surface already has EGL mapping");
        }
        
        mem->egl_mapped = TRUE;
    } else {
        GST_DEBUG("Surface memory type %d doesn't require EGL mapping", 
                  mem->surf->memType);
    }
#else
    GST_DEBUG("EGL mapping not required on x86 platform");
#endif
    
    g_mutex_unlock(&mem->lock);
    return TRUE;
}

/**
 * gst_nvdsvirtualcam_memory_unmap_egl:
 * @mem: Указатель на GstNvdsVirtualCamMemory
 * 
 * Освобождает EGL mapping. Вызывается только при уничтожении памяти
 * и только если reference count = 0.
 */
void gst_nvdsvirtualcam_memory_unmap_egl(GstNvdsVirtualCamMemory *mem)
{
    g_return_if_fail(mem != NULL);
    
    g_mutex_lock(&mem->lock);
    
#ifdef __aarch64__
    if (mem->egl_mapped && mem->surf && mem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        // Проверяем reference count перед unmapping
        if (mem->ref_count <= 1) {
            GST_DEBUG("Unmapping EGL for surface %p (ref_count=%d)", 
                      mem->surf, mem->ref_count);
            NvBufSurfaceUnMapEglImage(mem->surf, -1);
            mem->egl_mapped = FALSE;
        } else {
            GST_DEBUG("Skipping EGL unmap, ref_count=%d", mem->ref_count);
        }
    }
#endif
    
    g_mutex_unlock(&mem->lock);
}

/**
 * gst_nvdsvirtualcam_memory_register_cuda:
 * @mem: Указатель на GstNvdsVirtualCamMemory
 * 
 * Регистрирует EGL images в CUDA для получения доступа к памяти.
 * Требует предварительного вызова gst_nvdsvirtualcam_memory_map_egl.
 * 
 * Returns: TRUE при успехе, FALSE при ошибке
 */
gboolean gst_nvdsvirtualcam_memory_register_cuda(GstNvdsVirtualCamMemory *mem)
{
    g_return_val_if_fail(mem != NULL, FALSE);
    
    g_mutex_lock(&mem->lock);
    
    if (mem->cuda_registered) {
        GST_DEBUG("CUDA already registered for memory %p", mem);
        g_mutex_unlock(&mem->lock);
        return TRUE;
    }
    
#ifdef __aarch64__
    if (mem->surf->memType == NVBUF_MEM_SURFACE_ARRAY && mem->egl_mapped) {
        GST_DEBUG("Registering %u EGL images in CUDA", mem->surf->numFilled);
        
        mem->cuda_resources.resize(mem->surf->numFilled);
        mem->egl_frames.resize(mem->surf->numFilled);
        
        gboolean success = TRUE;
        guint i;
        
        for (i = 0; i < mem->surf->numFilled; i++) {
            CUresult cu_result;
            
            // Регистрируем EGL image в CUDA
            cu_result = cuGraphicsEGLRegisterImage(&mem->cuda_resources[i],
                    mem->surf->surfaceList[i].mappedAddr.eglImage,
                    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            
            if (cu_result != CUDA_SUCCESS) {
                const char *error_name = NULL;
                cuGetErrorName(cu_result, &error_name);
                GST_ERROR("Failed to register EGL image %u in CUDA: %s", 
                          i, error_name ? error_name : "unknown error");
                success = FALSE;
                break;
            }
            
            // Получаем mapped EGL frame
            cu_result = cuGraphicsResourceGetMappedEglFrame(&mem->egl_frames[i],
                    mem->cuda_resources[i], 0, 0);
            
            if (cu_result != CUDA_SUCCESS) {
                const char *error_name = NULL;
                cuGetErrorName(cu_result, &error_name);
                GST_ERROR("Failed to get mapped EGL frame %u: %s", 
                          i, error_name ? error_name : "unknown error");
                cuGraphicsUnregisterResource(mem->cuda_resources[i]);
                success = FALSE;
                break;
            }
            
            // Сохраняем указатель на память
            mem->frame_memory_ptrs[i] = (void*)mem->egl_frames[i].frame.pPitch[0];
            
            GST_DEBUG("Registered EGL image %u: ptr=%p, pitch=%u", i,
                    mem->frame_memory_ptrs[i], 
                    (guint)mem->surf->surfaceList[i].planeParams.pitch[0]);
        }
        
        if (!success) {
            // Очистка при ошибке
            for (guint j = 0; j < i; j++) {
                cuGraphicsUnregisterResource(mem->cuda_resources[j]);
            }
            mem->cuda_resources.clear();
            mem->egl_frames.clear();
            g_mutex_unlock(&mem->lock);
            return FALSE;
        }
        
        mem->cuda_registered = TRUE;
        GST_DEBUG("Successfully registered %u frames in CUDA", mem->surf->numFilled);
    } else {
        GST_WARNING("Cannot register CUDA: memType=%d, egl_mapped=%d",
                    mem->surf->memType, mem->egl_mapped);
    }
#else
    // На x86 просто используем прямые указатели
    for (guint i = 0; i < mem->surf->numFilled; i++) {
        mem->frame_memory_ptrs[i] = mem->surf->surfaceList[i].dataPtr;
    }
    mem->cuda_registered = TRUE;
#endif
    
    g_mutex_unlock(&mem->lock);
    return TRUE;
}

/**
 * gst_nvdsvirtualcam_memory_unregister_cuda:
 * @mem: Указатель на GstNvdsVirtualCamMemory
 * 
 * Освобождает CUDA регистрацию для EGL images.
 */
void gst_nvdsvirtualcam_memory_unregister_cuda(GstNvdsVirtualCamMemory *mem)
{
    g_return_if_fail(mem != NULL);
    
    g_mutex_lock(&mem->lock);
    
#ifdef __aarch64__
    if (mem->cuda_registered) {
        GST_DEBUG("Unregistering %zu CUDA resources", mem->cuda_resources.size());
        
        for (size_t i = 0; i < mem->cuda_resources.size(); i++) {
            CUresult result = cuGraphicsUnregisterResource(mem->cuda_resources[i]);
            if (result != CUDA_SUCCESS) {
                const char *error_name = NULL;
                cuGetErrorName(result, &error_name);
                GST_WARNING("Failed to unregister CUDA resource %zu: %s",
                           i, error_name ? error_name : "unknown error");
            }
        }
        
        mem->cuda_resources.clear();
        mem->egl_frames.clear();
        mem->cuda_registered = FALSE;
        GST_DEBUG("CUDA resources unregistered");
    }
#endif
    
    g_mutex_unlock(&mem->lock);
}

/* ============================================================================
 * Внутренние функции для создания и уничтожения памяти
 * ============================================================================ */

/**
 * create_virtualcam_memory:
 * Создает новую структуру GstNvdsVirtualCamMemory с NvBufSurface
 */
static GstNvdsVirtualCamMemory *
create_virtualcam_memory(guint width, guint height, guint gpu_id)
{
    GstNvdsVirtualCamMemory *virtualcam_mem = g_new0(GstNvdsVirtualCamMemory, 1);
    NvBufSurfaceCreateParams create_params;
    
    memset(&create_params, 0, sizeof(create_params));
    
    // Настройка параметров для создания surface
    create_params.gpuId = gpu_id;
    create_params.width = width;
    create_params.height = height;
    create_params.size = 0;  // Автоматический расчет
    create_params.isContiguous = 1;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    
#ifdef __aarch64__
    // На Jetson используем SURFACE_ARRAY для zero-copy с EGL
    create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
    GST_DEBUG("Creating SURFACE_ARRAY buffer for Jetson");
#else
    // На x86 используем обычную CUDA память
    create_params.memType = NVBUF_MEM_CUDA_DEVICE;
    GST_DEBUG("Creating CUDA_DEVICE buffer for x86");
#endif
    
    // Создаем NvBufSurface
    if (NvBufSurfaceCreate(&virtualcam_mem->surf, 1, &create_params) != 0) {
        GST_ERROR("Failed to create NvBufSurface: %ux%u on GPU %u",
                  width, height, gpu_id);
        g_free(virtualcam_mem);
        return NULL;
    }
    
    virtualcam_mem->surf->numFilled = 1;
    virtualcam_mem->surf->batchSize = 1;
    
    // Инициализация полей
    virtualcam_mem->egl_mapped = FALSE;
    virtualcam_mem->cuda_registered = FALSE;
    virtualcam_mem->ref_count = 1;
    g_mutex_init(&virtualcam_mem->lock);
    
    // Резервируем память для указателей
    virtualcam_mem->frame_memory_ptrs.resize(1);
    
#ifdef __aarch64__
    // На Jetson сразу делаем EGL mapping для нашего буфера
    if (create_params.memType == NVBUF_MEM_SURFACE_ARRAY) {
        if (!gst_nvdsvirtualcam_memory_map_egl(virtualcam_mem)) {
            GST_WARNING("Failed initial EGL mapping - will retry later");
            // Не считаем это критической ошибкой
        }
    }
#else
    // На dGPU просто сохраняем указатель
    virtualcam_mem->frame_memory_ptrs[0] = virtualcam_mem->surf->surfaceList[0].dataPtr;
#endif
    
    // Логирование информации о созданном буфере
    GST_DEBUG("Created virtualcam memory %p: %dx%d, pitch=%u, dataSize=%u, memType=%d",
              virtualcam_mem,
              width, height,
              virtualcam_mem->surf->surfaceList[0].planeParams.pitch[0],
              virtualcam_mem->surf->surfaceList[0].dataSize,
              virtualcam_mem->surf->memType);
    
    // Проверка выравнивания pitch
    guint pitch = virtualcam_mem->surf->surfaceList[0].planeParams.pitch[0];
    if (pitch % 32 != 0) {
        GST_WARNING("Pitch %u is not aligned to 32 bytes - may cause performance issues", 
                    pitch);
    }
    
    return virtualcam_mem;
}

/**
 * destroy_virtualcam_memory:
 * Освобождает все ресурсы, связанные с GstNvdsVirtualCamMemory
 */
static void
destroy_virtualcam_memory(GstNvdsVirtualCamMemory *mem)
{
    if (!mem) return;
    
    GST_DEBUG("Destroying virtualcam memory %p, ref_count=%d", mem, mem->ref_count);
    
    // Освобождаем CUDA ресурсы
    if (mem->cuda_registered) {
        gst_nvdsvirtualcam_memory_unregister_cuda(mem);
    }
    
    // Освобождаем EGL маппинг
    if (mem->egl_mapped) {
        gst_nvdsvirtualcam_memory_unmap_egl(mem);
    }
    
    // Уничтожаем surface
    if (mem->surf) {
        GST_DEBUG("Destroying NvBufSurface %p", mem->surf);
        NvBufSurfaceDestroy(mem->surf);
        mem->surf = NULL;
    }
    
    g_mutex_clear(&mem->lock);
    g_free(mem);
}

/* ============================================================================
 * Реализация методов GstAllocator
 * ============================================================================ */

/**
 * gst_nvdsvirtualcam_allocator_alloc:
 * Основная функция выделения памяти
 */
static GstMemory *
gst_nvdsvirtualcam_allocator_alloc(GstAllocator *allocator, gsize size,
                                   GstAllocationParams *params)
{
    GstNvdsVirtualCamAllocator *vcam_allocator = GST_NVDSVIRTUALCAM_ALLOCATOR(allocator);
    GstNvdsVirtualCamMem *mem = g_new0(GstNvdsVirtualCamMem, 1);
    
    GST_DEBUG("Allocating buffer: %ux%u on GPU %u",
              vcam_allocator->width, vcam_allocator->height,
              vcam_allocator->gpu_id);
    
    // Создаем virtualcam memory
    mem->virtualcam_mem = create_virtualcam_memory(
        vcam_allocator->width,
        vcam_allocator->height,
        vcam_allocator->gpu_id
    );
    
    if (!mem->virtualcam_mem) {
        GST_ERROR("Failed to create virtualcam memory");
        g_free(mem);
        return NULL;
    }
    
    // Инициализируем GstMemory
    gst_memory_init(GST_MEMORY_CAST(mem), 
                    static_cast<GstMemoryFlags>(0), 
                    allocator, 
                    NULL,
                    sizeof(NvBufSurface), 
                    0, 
                    0, 
                    sizeof(NvBufSurface));
    
    g_atomic_int_inc(&vcam_allocator->total_allocated);
    
    GST_DEBUG("Allocated buffer %u (total: %u)",
              g_atomic_int_get(&vcam_allocator->total_allocated),
              g_atomic_int_get(&vcam_allocator->total_allocated));
    
    // Параметры не используются в нашей реализации
    (void)size;
    (void)params;
    
    return GST_MEMORY_CAST(mem);
}

/**
 * gst_nvdsvirtualcam_allocator_free:
 * Освобождение памяти
 */
static void
gst_nvdsvirtualcam_allocator_free(GstAllocator *allocator, GstMemory *memory)
{
    GstNvdsVirtualCamAllocator *vcam_allocator = GST_NVDSVIRTUALCAM_ALLOCATOR(allocator);
    GstNvdsVirtualCamMem *mem = (GstNvdsVirtualCamMem *)memory;
    
    GST_DEBUG("Freeing memory %p", memory);
    
    if (mem->virtualcam_mem) {
        destroy_virtualcam_memory(mem->virtualcam_mem);
        mem->virtualcam_mem = NULL;
    }
    
    g_atomic_int_inc(&vcam_allocator->total_freed);
    
    GST_DEBUG("Freed buffer %u (allocated: %u, freed: %u)",
              g_atomic_int_get(&vcam_allocator->total_freed),
              g_atomic_int_get(&vcam_allocator->total_allocated),
              g_atomic_int_get(&vcam_allocator->total_freed));
    
    g_free(mem);
}

/**
 * gst_nvdsvirtualcam_allocator_map:
 * Маппинг памяти для доступа
 */
static gpointer
gst_nvdsvirtualcam_allocator_map(GstMemory *mem, gsize maxsize, GstMapFlags flags)
{
    GstNvdsVirtualCamMem *vcam_mem = (GstNvdsVirtualCamMem *)mem;
    
    GST_DEBUG("Mapping memory %p (flags: %d)", mem, flags);
    
    if (!vcam_mem->virtualcam_mem || !vcam_mem->virtualcam_mem->surf) {
        GST_ERROR("Invalid memory structure");
        return NULL;
    }
    
    // Возвращаем указатель на NvBufSurface
    return vcam_mem->virtualcam_mem->surf;
}

/**
 * gst_nvdsvirtualcam_allocator_unmap:
 * Unmapping памяти
 */
static void
gst_nvdsvirtualcam_allocator_unmap(GstMemory *mem)
{
    GST_DEBUG("Unmapping memory %p", mem);
    // В нашем случае unmap не требует действий
}

/* ============================================================================
 * Инициализация класса и объекта
 * ============================================================================ */

static void
gst_nvdsvirtualcam_allocator_class_init(GstNvdsVirtualCamAllocatorClass *klass)
{
    GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS(klass);
    
    allocator_class->alloc = gst_nvdsvirtualcam_allocator_alloc;
    allocator_class->free = gst_nvdsvirtualcam_allocator_free;
    
    GST_DEBUG_CATEGORY_INIT(gst_nvdsvirtualcam_allocator_debug, "nvdsvirtualcamallocator", 0,
                           "NVIDIA DeepStream Virtual Camera Allocator");
}

static void
gst_nvdsvirtualcam_allocator_init(GstNvdsVirtualCamAllocator *allocator)
{
    GstAllocator *alloc = GST_ALLOCATOR_CAST(allocator);
    
    alloc->mem_type = GST_NVDSVIRTUALCAM_MEMORY_TYPE;
    alloc->mem_map = (GstMemoryMapFunction)gst_nvdsvirtualcam_allocator_map;
    alloc->mem_unmap = gst_nvdsvirtualcam_allocator_unmap;
    
    // Устанавливаем флаг custom allocator
    GST_OBJECT_FLAG_SET(allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
    
    // Инициализация полей
    allocator->use_egl = TRUE;
    allocator->total_allocated = 0;
    allocator->total_freed = 0;
}

/* ============================================================================
 * Публичные API функции
 * ============================================================================ */

/**
 * gst_nvdsvirtualcam_allocator_new:
 * @width: Ширина буфера
 * @height: Высота буфера  
 * @gpu_id: ID GPU устройства
 * 
 * Создает новый allocator для nvdsvirtualcam плагина
 * 
 * Returns: (transfer full): Новый GstAllocator
 */
GstAllocator *
gst_nvdsvirtualcam_allocator_new(guint width, guint height, guint gpu_id)
{
    GstNvdsVirtualCamAllocator *allocator;
    
    g_return_val_if_fail(width > 0, NULL);
    g_return_val_if_fail(height > 0, NULL);
    
    allocator = (GstNvdsVirtualCamAllocator *)
        g_object_new(GST_TYPE_NVDSVIRTUALCAM_ALLOCATOR, NULL);
    
    allocator->width = width;
    allocator->height = height;
    allocator->gpu_id = gpu_id;
    
    GST_INFO("Created allocator for %ux%u on GPU %u", width, height, gpu_id);
    
    // Проверяем доступность GPU
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || gpu_id >= (guint)device_count) {
        GST_WARNING("GPU %u may not be available (device_count=%d, error=%s)",
                    gpu_id, device_count, cudaGetErrorString(cuda_err));
    }
    
    return GST_ALLOCATOR_CAST(allocator);
}

/**
 * gst_nvdsvirtualcam_buffer_get_memory:
 * @buffer: GstBuffer для получения памяти
 * 
 * Получает GstNvdsVirtualCamMemory из GstBuffer
 * 
 * Returns: (transfer none) (nullable): Указатель на GstNvdsVirtualCamMemory или NULL
 */
GstNvdsVirtualCamMemory *
gst_nvdsvirtualcam_buffer_get_memory(GstBuffer *buffer)
{
    GstMemory *mem;
    
    g_return_val_if_fail(buffer != NULL, NULL);
    
    if (gst_buffer_n_memory(buffer) == 0) {
        GST_WARNING("Buffer has no memory blocks");
        return NULL;
    }
    
    mem = gst_buffer_peek_memory(buffer, 0);
    if (!mem) {
        GST_WARNING("Failed to get memory from buffer");
        return NULL;
    }
    
    if (!gst_memory_is_type(mem, GST_NVDSVIRTUALCAM_MEMORY_TYPE)) {
        GST_WARNING("Memory is not of type %s (actual: %s)", 
                    GST_NVDSVIRTUALCAM_MEMORY_TYPE, mem->allocator->mem_type);
        return NULL;
    }
    
    GstNvdsVirtualCamMem *vcam_mem = (GstNvdsVirtualCamMem *)mem;
    return vcam_mem->virtualcam_mem;
}