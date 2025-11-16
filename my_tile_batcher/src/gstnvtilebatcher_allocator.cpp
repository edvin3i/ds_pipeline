// gstnvtilebatcher_allocator.cpp - Версия только для Jetson
#include "gstnvtilebatcher_allocator.h"
#include <cuda_runtime_api.h>
#include <cudaEGL.h>
#include <cstring>
#include <cstdio>

GST_DEBUG_CATEGORY_STATIC(gst_nvtilebatcher_allocator_debug);
#define GST_CAT_DEFAULT gst_nvtilebatcher_allocator_debug

#define GST_NVTILEBATCHER_MEMORY_TYPE "nvtilebatcher"
#define TILE_WIDTH 1024
#define TILE_HEIGHT 1024
#define TILES_PER_BATCH 6

typedef struct {
    GstMemory mem;
    GstNvTileBatcherMemory *batch_mem;
} GstNvTileBatcherMem;

G_DEFINE_TYPE(GstNvTileBatcherAllocator, gst_nvtilebatcher_allocator, GST_TYPE_ALLOCATOR);

/* ============================================================================
 * Создание batch памяти для Jetson
 * ============================================================================ */
static GstNvTileBatcherMemory *
create_batch_memory(guint gpu_id)
{
    GstNvTileBatcherMemory *batch_mem = g_new0(GstNvTileBatcherMemory, 1);
    NvBufSurfaceCreateParams create_params;
    
    memset(&create_params, 0, sizeof(create_params));
    
    // Параметры для Jetson
    create_params.gpuId = gpu_id;
    create_params.width = TILE_WIDTH;
    create_params.height = TILE_HEIGHT;
    create_params.size = 0;  // Автоматический расчет
    create_params.isContiguous = 1;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = NVBUF_MEM_SURFACE_ARRAY;  // Всегда для Jetson
    
    GST_INFO("Creating batch surface for Jetson: %dx%d x%d tiles, GPU %u",
             TILE_WIDTH, TILE_HEIGHT, TILES_PER_BATCH, gpu_id);
    
    // Создаем batch surface
    if (NvBufSurfaceCreate(&batch_mem->surf, TILES_PER_BATCH, &create_params) != 0) {
        GST_ERROR("Failed to create batch surface for %d tiles", TILES_PER_BATCH);
        g_free(batch_mem);
        return NULL;
    }
    
    // Проверяем тип памяти
    if (batch_mem->surf->memType != NVBUF_MEM_SURFACE_ARRAY) {
        GST_ERROR("Unexpected memory type %d, expected SURFACE_ARRAY(%d)", 
                  batch_mem->surf->memType, NVBUF_MEM_SURFACE_ARRAY);
        NvBufSurfaceDestroy(batch_mem->surf);
        g_free(batch_mem);
        return NULL;
    }
    
    // Критически важно: устанавливаем numFilled и batchSize
    batch_mem->surf->numFilled = 6;
    batch_mem->surf->batchSize = 6;
    
    // Инициализируем поля
    batch_mem->egl_mapped = FALSE;
    batch_mem->ref_count = 1;
    g_mutex_init(&batch_mem->lock);
    
    // Мапим EGL images для всех тайлов
    GST_DEBUG("Mapping EGL images for %d tiles", 6);
    
    if (NvBufSurfaceMapEglImage(batch_mem->surf, -1) != 0) {
        GST_ERROR("Failed to map EGL images for batch");
        NvBufSurfaceDestroy(batch_mem->surf);
        g_mutex_clear(&batch_mem->lock);
        g_free(batch_mem);
        return NULL;
    }
    
    batch_mem->egl_mapped = TRUE;
    
    // Проверяем каждый тайл
    gboolean all_mapped = TRUE;
    for (int i = 0; i < TILES_PER_BATCH; i++) {
        void* egl_image = batch_mem->surf->surfaceList[i].mappedAddr.eglImage;
        if (!egl_image) {
            GST_ERROR("No EGL image for tile %d after mapping", i);
            all_mapped = FALSE;
        } else {
            GST_DEBUG("Tile %d: EGL image = %p, size = %u bytes, pitch = %u", 
                      i, egl_image,
                      batch_mem->surf->surfaceList[i].dataSize,
                      batch_mem->surf->surfaceList[i].planeParams.pitch[0]);
        }
    }
    
    if (!all_mapped) {
        GST_ERROR("Failed to map all EGL images");
        NvBufSurfaceUnMapEglImage(batch_mem->surf, -1);
        NvBufSurfaceDestroy(batch_mem->surf);
        g_mutex_clear(&batch_mem->lock);
        g_free(batch_mem);
        return NULL;
    }
    
    // Проверка выравнивания pitch
    guint pitch = batch_mem->surf->surfaceList[0].planeParams.pitch[0];
    guint expected_pitch = TILE_WIDTH * 4;  // RGBA = 4 bytes per pixel
    
    if (pitch < expected_pitch) {
        GST_ERROR("Pitch %u is less than expected %u", pitch, expected_pitch);
        NvBufSurfaceUnMapEglImage(batch_mem->surf, -1);
        NvBufSurfaceDestroy(batch_mem->surf);
        g_mutex_clear(&batch_mem->lock);
        g_free(batch_mem);
        return NULL;
    }
    
    if (pitch % 64 != 0) {
        GST_WARNING("Pitch %u is not aligned to 64 bytes - may impact performance", pitch);
    }
    
    GST_INFO("✓ Created batch memory: %p, %dx%d x%d tiles, pitch=%u, numFilled=%d", 
              batch_mem, TILE_WIDTH, TILE_HEIGHT, TILES_PER_BATCH,
              pitch, batch_mem->surf->numFilled);
    
    return batch_mem;
}

/* ============================================================================
 * Уничтожение batch памяти
 * ============================================================================ */
static void
destroy_batch_memory(GstNvTileBatcherMemory *mem)
{
    if (!mem) return;
    
    GST_DEBUG("Destroying batch memory %p (ref_count=%d)", mem, mem->ref_count);
    
    // Освобождаем EGL маппинг
    if (mem->egl_mapped && mem->surf) {
        GST_DEBUG("Unmapping EGL images for batch");
        NvBufSurfaceUnMapEglImage(mem->surf, -1);
        mem->egl_mapped = FALSE;
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
 * GstAllocator методы
 * ============================================================================ */
static GstMemory *
gst_nvtilebatcher_allocator_alloc(GstAllocator *allocator, gsize size,
                                  GstAllocationParams *params)
{
    GstNvTileBatcherAllocator *batch_allocator = GST_NVTILEBATCHER_ALLOCATOR(allocator);
    GstNvTileBatcherMem *mem = g_new0(GstNvTileBatcherMem, 1);
    
    (void)size;
    (void)params;
    
    GST_DEBUG("Allocating batch buffer on GPU %u", batch_allocator->gpu_id);
    
    // Устанавливаем CUDA устройство
    cudaError_t cuda_err = cudaSetDevice(batch_allocator->gpu_id);
    if (cuda_err != cudaSuccess) {
        GST_ERROR("Failed to set CUDA device %u: %s", 
                  batch_allocator->gpu_id, cudaGetErrorString(cuda_err));
        g_free(mem);
        return NULL;
    }
    
    // Создаем batch память
    mem->batch_mem = create_batch_memory(batch_allocator->gpu_id);
    if (!mem->batch_mem) {
        GST_ERROR("Failed to create batch memory");
        g_free(mem);
        return NULL;
    }
    
    // Инициализируем GstMemory
    gst_memory_init(GST_MEMORY_CAST(mem), 
                    (GstMemoryFlags)0,
                    allocator,
                    NULL,
                    sizeof(NvBufSurface),
                    0,
                    0,
                    sizeof(NvBufSurface));
    
    g_atomic_int_inc(&batch_allocator->total_allocated);
    
    GST_INFO("Allocated batch buffer %u (total allocated: %u)", 
             g_atomic_int_get(&batch_allocator->total_allocated),
             g_atomic_int_get(&batch_allocator->total_allocated));
    
    return GST_MEMORY_CAST(mem);
}

static void
gst_nvtilebatcher_allocator_free(GstAllocator *allocator, GstMemory *memory)
{
    GstNvTileBatcherAllocator *batch_allocator = GST_NVTILEBATCHER_ALLOCATOR(allocator);
    GstNvTileBatcherMem *mem = (GstNvTileBatcherMem *)memory;
    
    GST_DEBUG("Freeing batch memory %p", memory);
    
    if (mem->batch_mem) {
        destroy_batch_memory(mem->batch_mem);
        mem->batch_mem = NULL;
    }
    
    g_atomic_int_inc(&batch_allocator->total_freed);
    
    GST_DEBUG("Freed buffer (total: allocated=%u, freed=%u)",
              g_atomic_int_get(&batch_allocator->total_allocated),
              g_atomic_int_get(&batch_allocator->total_freed));
    
    g_free(mem);
}

static gpointer
gst_nvtilebatcher_allocator_map(GstMemory *mem, gsize maxsize, GstMapFlags flags)
{
    GstNvTileBatcherMem *batch_mem = (GstNvTileBatcherMem *)mem;
    
    (void)maxsize;
    (void)flags;
    
    GST_DEBUG("Mapping memory %p", mem);
    
    if (!batch_mem->batch_mem || !batch_mem->batch_mem->surf) {
        GST_ERROR("Invalid batch memory structure in map");
        return NULL;
    }
    
    // Возвращаем указатель на NvBufSurface
    return batch_mem->batch_mem->surf;
}

static void
gst_nvtilebatcher_allocator_unmap(GstMemory *mem)
{
    GST_DEBUG("Unmapping memory %p", mem);
    // Для NvBufSurface unmap не требует действий
    (void)mem;
}

/* ============================================================================
 * Инициализация класса
 * ============================================================================ */
static void
gst_nvtilebatcher_allocator_class_init(GstNvTileBatcherAllocatorClass *klass)
{
    GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS(klass);
    
    allocator_class->alloc = gst_nvtilebatcher_allocator_alloc;
    allocator_class->free = gst_nvtilebatcher_allocator_free;
    
    GST_DEBUG_CATEGORY_INIT(gst_nvtilebatcher_allocator_debug, 
                           "nvtilebatcherallocator", 0,
                           "NVIDIA Tile Batcher Allocator for Jetson");
}

static void
gst_nvtilebatcher_allocator_init(GstNvTileBatcherAllocator *allocator)
{
    GstAllocator *alloc = GST_ALLOCATOR_CAST(allocator);
    
    alloc->mem_type = GST_NVTILEBATCHER_MEMORY_TYPE;
    alloc->mem_map = (GstMemoryMapFunction)gst_nvtilebatcher_allocator_map;
    alloc->mem_unmap = gst_nvtilebatcher_allocator_unmap;
    
    // Устанавливаем флаг custom allocator
    GST_OBJECT_FLAG_SET(allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
    
    allocator->total_allocated = 0;
    allocator->total_freed = 0;
}

/* ============================================================================
 * Публичные API функции
 * ============================================================================ */
GstAllocator *
gst_nvtilebatcher_allocator_new(guint gpu_id)
{
    GstNvTileBatcherAllocator *allocator;
    
    allocator = (GstNvTileBatcherAllocator *)
        g_object_new(GST_TYPE_NVTILEBATCHER_ALLOCATOR, NULL);
    
    allocator->gpu_id = gpu_id;
    
    GST_INFO("Created batch allocator for Jetson (GPU %u, SURFACE_ARRAY mode)", gpu_id);
    
    // Проверяем доступность GPU
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess) {
        GST_ERROR("Failed to get CUDA device count: %s", cudaGetErrorString(cuda_err));
        g_object_unref(allocator);
        return NULL;
    }
    
    if (gpu_id >= (guint)device_count) {
        GST_ERROR("GPU %u not available (device_count=%d)", gpu_id, device_count);
        g_object_unref(allocator);
        return NULL;
    }
    
    // Проверяем возможности устройства
    cudaDeviceProp prop;
    cuda_err = cudaGetDeviceProperties(&prop, gpu_id);
    if (cuda_err == cudaSuccess) {
        GST_INFO("GPU %u: %s, compute capability %d.%d", 
                 gpu_id, prop.name, prop.major, prop.minor);
    }
    
    return GST_ALLOCATOR_CAST(allocator);
}

GstNvTileBatcherMemory *
gst_nvtilebatcher_buffer_get_memory(GstBuffer *buffer)
{
    GstMemory *mem;
    
    g_return_val_if_fail(buffer != NULL, NULL);
    
    if (gst_buffer_n_memory(buffer) == 0) {
        GST_WARNING("Buffer has no memory blocks");
        return NULL;
    }
    
    mem = gst_buffer_peek_memory(buffer, 0);
    if (!mem) {
        GST_WARNING("Failed to peek memory from buffer");
        return NULL;
    }
    
    if (!gst_memory_is_type(mem, GST_NVTILEBATCHER_MEMORY_TYPE)) {
        GST_WARNING("Memory is not of type %s (actual: %s)", 
                    GST_NVTILEBATCHER_MEMORY_TYPE, 
                    mem->allocator ? mem->allocator->mem_type : "unknown");
        return NULL;
    }
    
    GstNvTileBatcherMem *batch_mem = (GstNvTileBatcherMem *)mem;
    
    if (!batch_mem->batch_mem) {
        GST_WARNING("batch_mem->batch_mem is NULL");
        return NULL;
    }
    
    return batch_mem->batch_mem;
}