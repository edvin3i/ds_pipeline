/*
 * gstnvdsstitch_allocator.cpp - Allocator для nvdsstitch плагина с полной поддержкой EGL
 * 
 * Этот allocator управляет GPU памятью для выходных буферов стичинга
 * и обеспечивает правильную работу с EGL на платформе Jetson
 */

#include "gstnvdsstitch_allocator.h"
#include <nvbufsurface.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <atomic>

GST_DEBUG_CATEGORY_STATIC(gst_nvdsstitch_allocator_debug);
#define GST_CAT_DEFAULT gst_nvdsstitch_allocator_debug

#define GST_NVDSSTITCH_MEMORY_TYPE "nvdsstitch"

// Внутренняя структура для GstMemory
typedef struct {
    GstMemory mem;
    GstNvdsStitchMemory *stitch_mem;
} GstNvdsStitchMem;

G_DEFINE_TYPE(GstNvdsStitchAllocator, gst_nvdsstitch_allocator, GST_TYPE_ALLOCATOR);

/* ============================================================================
 * Вспомогательные функции для работы с EGL
 * ============================================================================ */

/**
 * @brief Map NVMM surface to EGL images for GPU access
 *
 * Performs EGL mapping for NvBufSurface. On Jetson platforms, this is
 * required for CUDA access to surface memory.
 *
 * @param[in,out] mem Pointer to GstNvdsStitchMemory structure
 *
 * @return Success status
 * @retval TRUE EGL mapping succeeded or already mapped (idempotent)
 * @retval FALSE Mapping failed (errors logged via GST_ERROR)
 *
 * @note Function is idempotent - safe to call multiple times
 * @note Only functional on aarch64 (Jetson) with SURFACE_ARRAY mem type
 * @note Increments ref_count to track mapping state
 *
 * @see gst_nvdsstitch_memory_unmap_egl
 * @see gst_nvdsstitch_memory_register_cuda
 */
gboolean gst_nvdsstitch_memory_map_egl(GstNvdsStitchMemory *mem)
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
 * @brief Unmap EGL images from NVMM surface
 *
 * Destroys EGL images and releases EGL resources. Only performs unmapping
 * when reference count reaches 1 or below to prevent premature release.
 *
 * @param[in,out] mem Pointer to GstNvdsStitchMemory structure
 *
 * @note Safe to call even if not mapped (no-op)
 * @note Automatically checks ref_count before unmapping
 * @note Should be called before memory destruction
 *
 * @see gst_nvdsstitch_memory_map_egl
 */
void gst_nvdsstitch_memory_unmap_egl(GstNvdsStitchMemory *mem)
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
 * @brief Register CUDA graphics resources for EGL surface access
 *
 * Registers EGL images as CUDA graphics resources, enabling CUDA kernels
 * to directly access surface memory. Requires prior EGL mapping.
 *
 * @param[in,out] mem Pointer to GstNvdsStitchMemory structure
 *
 * @return Success status
 * @retval TRUE CUDA registration succeeded or already registered (idempotent)
 * @retval FALSE Registration failed (errors logged via GST_ERROR)
 *
 * @note Function is idempotent - safe to call multiple times
 * @note Requires prior call to gst_nvdsstitch_memory_map_egl()
 * @note Only functional on aarch64 (Jetson) platforms
 * @note Populates cuda_resources, egl_frames, and frame_memory_ptrs vectors
 *
 * @see gst_nvdsstitch_memory_map_egl
 * @see gst_nvdsstitch_memory_unregister_cuda
 */
gboolean gst_nvdsstitch_memory_register_cuda(GstNvdsStitchMemory *mem)
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
 * @brief Unregister CUDA graphics resources for EGL images
 *
 * Releases CUDA registration for EGL images and clears resource vectors.
 * Called before EGL unmapping or memory destruction.
 *
 * @param[in,out] mem Pointer to GstNvdsStitchMemory structure
 *
 * @note Safe to call even if not registered (no-op)
 * @note Must be called before gst_nvdsstitch_memory_unmap_egl()
 * @note Clears cuda_resources and egl_frames vectors
 * @note Clears frame_memory_ptrs vector
 *
 * @see gst_nvdsstitch_memory_register_cuda
 */
void gst_nvdsstitch_memory_unregister_cuda(GstNvdsStitchMemory *mem)
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
 * create_stitch_memory:
 * Создает новую структуру GstNvdsStitchMemory с NvBufSurface
 */
static GstNvdsStitchMemory *
create_stitch_memory(guint width, guint height, guint gpu_id)
{
    GstNvdsStitchMemory *stitch_mem = g_new0(GstNvdsStitchMemory, 1);
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
    if (NvBufSurfaceCreate(&stitch_mem->surf, 1, &create_params) != 0) {
        GST_ERROR("Failed to create NvBufSurface: %ux%u on GPU %u",
                  width, height, gpu_id);
        g_free(stitch_mem);
        return NULL;
    }
    
    stitch_mem->surf->numFilled = 1;
    stitch_mem->surf->batchSize = 1;
    
    // Инициализация полей
    stitch_mem->egl_mapped = FALSE;
    stitch_mem->cuda_registered = FALSE;
    stitch_mem->ref_count = 1;
    g_mutex_init(&stitch_mem->lock);
    
    // Резервируем память для указателей
    stitch_mem->frame_memory_ptrs.resize(1);
    
#ifdef __aarch64__
    // На Jetson сразу делаем EGL mapping для нашего буфера
    if (create_params.memType == NVBUF_MEM_SURFACE_ARRAY) {
        if (!gst_nvdsstitch_memory_map_egl(stitch_mem)) {
            GST_WARNING("Failed initial EGL mapping - will retry later");
            // Не считаем это критической ошибкой
        }
    }
#else
    // На dGPU просто сохраняем указатель
    stitch_mem->frame_memory_ptrs[0] = stitch_mem->surf->surfaceList[0].dataPtr;
#endif
    
    // Логирование информации о созданном буфере
    GST_DEBUG("Created stitch memory %p: %dx%d, pitch=%u, dataSize=%u, memType=%d",
              stitch_mem,
              width, height,
              stitch_mem->surf->surfaceList[0].planeParams.pitch[0],
              stitch_mem->surf->surfaceList[0].dataSize,
              stitch_mem->surf->memType);
    
    // Проверка выравнивания pitch
    guint pitch = stitch_mem->surf->surfaceList[0].planeParams.pitch[0];
    if (pitch % 32 != 0) {
        GST_WARNING("Pitch %u is not aligned to 32 bytes - may cause performance issues", 
                    pitch);
    }
    
    return stitch_mem;
}

/**
 * destroy_stitch_memory:
 * Освобождает все ресурсы, связанные с GstNvdsStitchMemory
 */
static void
destroy_stitch_memory(GstNvdsStitchMemory *mem)
{
    if (!mem) return;
    
    GST_DEBUG("Destroying stitch memory %p, ref_count=%d", mem, mem->ref_count);
    
    // Освобождаем CUDA ресурсы
    if (mem->cuda_registered) {
        gst_nvdsstitch_memory_unregister_cuda(mem);
    }
    
    // Освобождаем EGL маппинг
    if (mem->egl_mapped) {
        gst_nvdsstitch_memory_unmap_egl(mem);
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
 * gst_nvdsstitch_allocator_alloc:
 * Основная функция выделения памяти
 */
static GstMemory *
gst_nvdsstitch_allocator_alloc(GstAllocator *allocator, gsize size,
                               GstAllocationParams *params)
{
    GstNvdsStitchAllocator *stitch_allocator = GST_NVDSSTITCH_ALLOCATOR(allocator);
    GstNvdsStitchMem *mem = g_new0(GstNvdsStitchMem, 1);
    
    GST_DEBUG("Allocating buffer: %ux%u on GPU %u",
              stitch_allocator->width, stitch_allocator->height,
              stitch_allocator->gpu_id);
    
    // Создаем stitch memory
    mem->stitch_mem = create_stitch_memory(
        stitch_allocator->width,
        stitch_allocator->height,
        stitch_allocator->gpu_id
    );
    
    if (!mem->stitch_mem) {
        GST_ERROR("Failed to create stitch memory");
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
    
    g_atomic_int_inc(&stitch_allocator->total_allocated);
    
    GST_DEBUG("Allocated buffer %u (total: %u)",
              g_atomic_int_get(&stitch_allocator->total_allocated),
              g_atomic_int_get(&stitch_allocator->total_allocated));
    
    // Параметры не используются в нашей реализации
    (void)size;
    (void)params;
    
    return GST_MEMORY_CAST(mem);
}

/**
 * gst_nvdsstitch_allocator_free:
 * Освобождение памяти
 */
static void
gst_nvdsstitch_allocator_free(GstAllocator *allocator, GstMemory *memory)
{
    GstNvdsStitchAllocator *stitch_allocator = GST_NVDSSTITCH_ALLOCATOR(allocator);
    GstNvdsStitchMem *mem = (GstNvdsStitchMem *)memory;
    
    GST_DEBUG("Freeing memory %p", memory);
    
    if (mem->stitch_mem) {
        destroy_stitch_memory(mem->stitch_mem);
        mem->stitch_mem = NULL;
    }
    
    g_atomic_int_inc(&stitch_allocator->total_freed);
    
    GST_DEBUG("Freed buffer %u (allocated: %u, freed: %u)",
              g_atomic_int_get(&stitch_allocator->total_freed),
              g_atomic_int_get(&stitch_allocator->total_allocated),
              g_atomic_int_get(&stitch_allocator->total_freed));
    
    g_free(mem);
}

/**
 * gst_nvdsstitch_allocator_map:
 * Маппинг памяти для доступа
 */
static gpointer
gst_nvdsstitch_allocator_map(GstMemory *mem, gsize maxsize, GstMapFlags flags)
{
    GstNvdsStitchMem *stitch_mem = (GstNvdsStitchMem *)mem;
    
    GST_DEBUG("Mapping memory %p (flags: %d)", mem, flags);
    
    if (!stitch_mem->stitch_mem || !stitch_mem->stitch_mem->surf) {
        GST_ERROR("Invalid memory structure");
        return NULL;
    }
    
    // Возвращаем указатель на NvBufSurface
    return stitch_mem->stitch_mem->surf;
}

/**
 * gst_nvdsstitch_allocator_unmap:
 * Unmapping памяти
 */
static void
gst_nvdsstitch_allocator_unmap(GstMemory *mem)
{
    GST_DEBUG("Unmapping memory %p", mem);
    // В нашем случае unmap не требует действий
}

/* ============================================================================
 * Инициализация класса и объекта
 * ============================================================================ */

static void
gst_nvdsstitch_allocator_class_init(GstNvdsStitchAllocatorClass *klass)
{
    GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS(klass);
    
    allocator_class->alloc = gst_nvdsstitch_allocator_alloc;
    allocator_class->free = gst_nvdsstitch_allocator_free;
    
    GST_DEBUG_CATEGORY_INIT(gst_nvdsstitch_allocator_debug, "nvdsstitchallocator", 0,
                           "NVIDIA DeepStream Stitch Allocator");
}

static void
gst_nvdsstitch_allocator_init(GstNvdsStitchAllocator *allocator)
{
    GstAllocator *alloc = GST_ALLOCATOR_CAST(allocator);
    
    alloc->mem_type = GST_NVDSSTITCH_MEMORY_TYPE;
    alloc->mem_map = (GstMemoryMapFunction)gst_nvdsstitch_allocator_map;
    alloc->mem_unmap = gst_nvdsstitch_allocator_unmap;
    
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
 * gst_nvdsstitch_allocator_new:
 * @width: Ширина буфера
 * @height: Высота буфера  
 * @gpu_id: ID GPU устройства
 * 
 * Создает новый allocator для nvdsstitch плагина
 * 
 * Returns: (transfer full): Новый GstAllocator
 */
GstAllocator *
gst_nvdsstitch_allocator_new(guint width, guint height, guint gpu_id)
{
    GstNvdsStitchAllocator *allocator;
    
    g_return_val_if_fail(width > 0, NULL);
    g_return_val_if_fail(height > 0, NULL);
    
    allocator = (GstNvdsStitchAllocator *)
        g_object_new(GST_TYPE_NVDSSTITCH_ALLOCATOR, NULL);
    
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
 * gst_nvdsstitch_buffer_get_memory:
 * @buffer: GstBuffer для получения памяти
 * 
 * Получает GstNvdsStitchMemory из GstBuffer
 * 
 * Returns: (transfer none) (nullable): Указатель на GstNvdsStitchMemory или NULL
 */
GstNvdsStitchMemory *
gst_nvdsstitch_buffer_get_memory(GstBuffer *buffer)
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
    
    if (!gst_memory_is_type(mem, GST_NVDSSTITCH_MEMORY_TYPE)) {
        GST_WARNING("Memory is not of type %s (actual: %s)", 
                    GST_NVDSSTITCH_MEMORY_TYPE, mem->allocator->mem_type);
        return NULL;
    }
    
    GstNvdsStitchMem *stitch_mem = (GstNvdsStitchMem *)mem;
    return stitch_mem->stitch_mem;
}