/**
 * gstnvdsstitch.cpp - Плагин для панорамной склейки 360°
 * 
 * Использует LUT карты и веса для создания эквиректангулярной проекции
 */

#include "gstnvdsstitch.h"
#include "nvdsstitch_config.h"
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <cuda_runtime_api.h>
#include <gstnvdsbufferpool.h>
#include "gstnvdsstitch_allocator.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cstring>
#include <cstdio>
#include "cuda_stitch_kernel.h"

#ifndef PACKAGE
#define PACKAGE "nvdsstitch"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvds_stitch_debug);
#define GST_CAT_DEFAULT gst_nvds_stitch_debug

// Макросы для логирования
#define LOG_ERROR(obj, fmt, ...) GST_ERROR_OBJECT(obj, "❌ " fmt, ##__VA_ARGS__)
#define LOG_WARNING(obj, fmt, ...) GST_WARNING_OBJECT(obj, "⚠️ " fmt, ##__VA_ARGS__)
#define LOG_INFO(obj, fmt, ...) GST_INFO_OBJECT(obj, "ℹ️ " fmt, ##__VA_ARGS__)
#define LOG_DEBUG(obj, fmt, ...) GST_DEBUG_OBJECT(obj, "🔍 " fmt, ##__VA_ARGS__)

// Параметры очистки кэша
#define CACHE_CLEANUP_INTERVAL 150
#define CACHE_ENTRY_TTL 300
#define CACHE_MAX_SIZE 50

// ============================================================================
// ОБРАБОТКА ОШИБОК И УСТОЙЧИВОСТЬ
// ============================================================================

// Макрос для проверки NULL указателей
#define CHECK_NULL_RET(ptr, obj, msg) \
    do { \
        if (!(ptr)) { \
            LOG_ERROR(obj, "NULL pointer: %s", msg); \
            return FALSE; \
        } \
    } while(0)

#define CHECK_NULL_RET_FLOW(ptr, obj, msg) \
    do { \
        if (!(ptr)) { \
            LOG_ERROR(obj, "NULL pointer: %s", msg); \
            return GST_FLOW_ERROR; \
        } \
    } while(0)

// Функция очистки CUDA состояния при ошибке
static void reset_cuda_state(GstNvdsStitch *stitch) {
    if (!stitch) return;

    if (stitch->cuda_stream) {
        cudaStreamSynchronize(stitch->cuda_stream);
    }

    // Очистить последнюю ошибку CUDA
    cudaGetLastError();

    LOG_DEBUG(stitch, "CUDA state reset");
}

extern "C" {
GstBufferPool* gst_nvds_buffer_pool_new(void) {
    return gst_buffer_pool_new();
}

GType gst_nvds_buffer_pool_get_type(void) {
    return gst_buffer_pool_get_type();
}
}

// Forward declarations
static gboolean get_or_register_egl_resource(GstNvdsStitch *stitch,
                                            gpointer egl_image,
                                            gboolean is_write,
                                            CUgraphicsResource *resource,
                                            CUeglFrame *frame);

// Pad templates
static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=RGBA"));

static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=RGBA"));

G_DEFINE_TYPE(GstNvdsStitch, gst_nvds_stitch, GST_TYPE_BASE_TRANSFORM);

// Properties
enum {
    PROP_0,
    PROP_LEFT_SOURCE_ID,
    PROP_RIGHT_SOURCE_ID,
    PROP_GPU_ID,
    PROP_USE_EGL,
    PROP_PANORAMA_WIDTH,
    PROP_PANORAMA_HEIGHT
};

/* ============================================================================
 * EGL Cache Management
 * ============================================================================ */

typedef struct {
    gpointer egl_image;
    gboolean is_write;
} EGLCacheKey;

extern "C" {
    cudaError_t update_color_correction_simple(
        const unsigned char* left_frame,
        const unsigned char* right_frame,
        const float* weight_left,    // Добавить
        const float* weight_right,   // Добавить
        int width,
        int height,
        int pitch,
        cudaStream_t stream
    );
}

// Forward declarations
extern "C" cudaError_t init_color_correction();

static guint egl_cache_key_hash(gconstpointer key)
{
    const EGLCacheKey *k = (const EGLCacheKey *)key;
    return GPOINTER_TO_UINT(k->egl_image) ^ (k->is_write ? 0xDEADBEEF : 0);
}

static gboolean egl_cache_key_equal(gconstpointer a, gconstpointer b)
{
    const EGLCacheKey *ka = (const EGLCacheKey *)a;
    const EGLCacheKey *kb = (const EGLCacheKey *)b;
    return ka->egl_image == kb->egl_image && ka->is_write == kb->is_write;
}

static void egl_cache_key_free(gpointer key)
{
    g_free(key);
}

static void egl_cache_entry_free(gpointer entry)
{
    g_free(entry);
}

/* ============================================================================
 * Промежуточные буферы для панорамы
 * ============================================================================ */

static gboolean setup_intermediate_buffer_pool(GstNvdsStitch *stitch)
{
    LOG_INFO(stitch, "Setting up intermediate buffer pool for panorama");
    
    stitch->intermediate_pool = gst_nvds_buffer_pool_new();
    if (!stitch->intermediate_pool) {
        LOG_ERROR(stitch, "Failed to create intermediate buffer pool");
        return FALSE;
    }
    
    GstAllocator *allocator = gst_nvdsstitch_allocator_new(
        NvdsStitchConfig::INPUT_WIDTH,
        NvdsStitchConfig::INPUT_HEIGHT,
        stitch->gpu_id
    );
    
    if (!allocator) {
        LOG_ERROR(stitch, "Failed to create intermediate allocator");
        gst_object_unref(stitch->intermediate_pool);
        stitch->intermediate_pool = NULL;
        return FALSE;
    }
    
    GstStructure *config = gst_buffer_pool_get_config(stitch->intermediate_pool);
    
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",
        "width", G_TYPE_INT, NvdsStitchConfig::INPUT_WIDTH,
        "height", G_TYPE_INT, NvdsStitchConfig::INPUT_HEIGHT,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    gst_caps_set_features(caps, 0, gst_caps_features_new("memory:NVMM", NULL));
    
    gst_buffer_pool_config_set_params(config, caps, sizeof(NvBufSurface), 2, 2);
    gst_buffer_pool_config_set_allocator(config, allocator, NULL);
    
    gst_structure_set(config,
        "memtype", G_TYPE_UINT, NvdsStitchConfig::POOL_MEMTYPE,
        "gpu-id", G_TYPE_UINT, stitch->gpu_id,
        "batch-size", G_TYPE_UINT, 1,
        NULL);
    
    gst_caps_unref(caps);
    gst_object_unref(allocator);
    
    if (!gst_buffer_pool_set_config(stitch->intermediate_pool, config)) {
        LOG_ERROR(stitch, "Failed to set intermediate buffer pool config");
        gst_object_unref(stitch->intermediate_pool);
        stitch->intermediate_pool = NULL;
        return FALSE;
    }
    
    if (!gst_buffer_pool_set_active(stitch->intermediate_pool, TRUE)) {
        LOG_ERROR(stitch, "Failed to activate intermediate buffer pool");
        gst_object_unref(stitch->intermediate_pool);
        stitch->intermediate_pool = NULL;
        return FALSE;
    }
    
    GstFlowReturn flow_ret;
    
    flow_ret = gst_buffer_pool_acquire_buffer(stitch->intermediate_pool, 
                                             &stitch->intermediate_left, NULL);
    if (flow_ret != GST_FLOW_OK) {
        LOG_ERROR(stitch, "Failed to acquire left intermediate buffer");
        return FALSE;
    }
    
    flow_ret = gst_buffer_pool_acquire_buffer(stitch->intermediate_pool, 
                                             &stitch->intermediate_right, NULL);
    if (flow_ret != GST_FLOW_OK) {
        LOG_ERROR(stitch, "Failed to acquire right intermediate buffer");
        gst_buffer_unref(stitch->intermediate_left);
        stitch->intermediate_left = NULL;
        return FALSE;
    }
    
    GstMapInfo map_info;
    
    if (!gst_buffer_map(stitch->intermediate_left, &map_info, GST_MAP_READWRITE)) {
        LOG_ERROR(stitch, "Failed to map left intermediate buffer");
        return FALSE;
    }
    stitch->intermediate_left_surf = (NvBufSurface *)map_info.data;
    gst_buffer_unmap(stitch->intermediate_left, &map_info);
    
    if (!gst_buffer_map(stitch->intermediate_right, &map_info, GST_MAP_READWRITE)) {
        LOG_ERROR(stitch, "Failed to map right intermediate buffer");
        return FALSE;
    }
    stitch->intermediate_right_surf = (NvBufSurface *)map_info.data;
    gst_buffer_unmap(stitch->intermediate_right, &map_info);
    
#ifdef __aarch64__
    if (stitch->intermediate_left_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        if (NvBufSurfaceMapEglImage(stitch->intermediate_left_surf, -1) != 0) {
            LOG_ERROR(stitch, "Failed to map EGL for left intermediate buffer");
            return FALSE;
        }
        if (NvBufSurfaceMapEglImage(stitch->intermediate_right_surf, -1) != 0) {
            LOG_ERROR(stitch, "Failed to map EGL for right intermediate buffer");
            return FALSE;
        }
        
        CUgraphicsResource resource;
        CUeglFrame frame;
        
        if (!get_or_register_egl_resource(stitch, 
            stitch->intermediate_left_surf->surfaceList[0].mappedAddr.eglImage,
            FALSE, &resource, &frame)) {
            LOG_ERROR(stitch, "Failed to register left intermediate buffer");
            return FALSE;
        }
        
        if (!get_or_register_egl_resource(stitch,
            stitch->intermediate_right_surf->surfaceList[0].mappedAddr.eglImage,
            FALSE, &resource, &frame)) {
            LOG_ERROR(stitch, "Failed to register right intermediate buffer");
            return FALSE;
        }
        
        LOG_INFO(stitch, "Intermediate buffers registered in EGL/CUDA successfully");
    }
#endif
    
    LOG_INFO(stitch, "Intermediate buffer pool setup complete for panorama");
    return TRUE;
}

static gboolean setup_fixed_output_pool(GstNvdsStitch *stitch)
{
    LOG_INFO(stitch, "Setting up fixed output buffer pool for panorama (%dx%d)",
             stitch->output_width, stitch->output_height);
    
    g_mutex_init(&stitch->output_pool_fixed.mutex);
    stitch->output_pool_fixed.current_index = 0;
    
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        GstFlowReturn flow_ret;
        
        flow_ret = gst_buffer_pool_acquire_buffer(stitch->output_pool, 
                                                 &stitch->output_pool_fixed.buffers[i], 
                                                 NULL);
        if (flow_ret != GST_FLOW_OK) {
            LOG_ERROR(stitch, "Failed to acquire output buffer %d", i);
            return FALSE;
        }
        
        GstMapInfo map_info;
        if (!gst_buffer_map(stitch->output_pool_fixed.buffers[i], &map_info, GST_MAP_READWRITE)) {
            LOG_ERROR(stitch, "Failed to map output buffer %d", i);
            return FALSE;
        }
        
        stitch->output_pool_fixed.surfaces[i] = (NvBufSurface *)map_info.data;
        gst_buffer_unmap(stitch->output_pool_fixed.buffers[i], &map_info);
        
#ifdef __aarch64__
        if (stitch->output_pool_fixed.surfaces[i]->memType == NVBUF_MEM_SURFACE_ARRAY) {
            if (NvBufSurfaceMapEglImage(stitch->output_pool_fixed.surfaces[i], -1) != 0) {
                LOG_ERROR(stitch, "Failed to map EGL for output buffer %d", i);
                return FALSE;
            }
            
            CUgraphicsResource resource;
            CUeglFrame frame;
            
            if (!get_or_register_egl_resource(stitch, 
                stitch->output_pool_fixed.surfaces[i]->surfaceList[0].mappedAddr.eglImage,
                TRUE, &resource, &frame)) {
                LOG_ERROR(stitch, "Failed to register output buffer %d", i);
                return FALSE;
            }
            
            stitch->output_pool_fixed.registered[i] = TRUE;
            LOG_INFO(stitch, "Output buffer %d registered in EGL/CUDA successfully", i);
        }
#else
        stitch->output_pool_fixed.registered[i] = TRUE;
#endif
    }
    
    LOG_INFO(stitch, "Fixed output pool ready with %d pre-registered buffers", FIXED_OUTPUT_POOL_SIZE);
    return TRUE;
}

/* ============================================================================
 * Cache Management
 * ============================================================================ */

static void cleanup_stale_cache_entries(GstNvdsStitch *stitch)
{
    GHashTableIter iter;
    gpointer key, value;
    GList *keys_to_remove = NULL;
    guint removed_count = 0;
    
    g_mutex_lock(&stitch->egl_lock);
    
    guint cache_size = g_hash_table_size(stitch->egl_resource_cache);
    guint ttl_threshold = (cache_size > CACHE_MAX_SIZE) ? 50 : CACHE_ENTRY_TTL;
    
    g_hash_table_iter_init(&iter, stitch->egl_resource_cache);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        EGLResourceCacheEntry *entry = (EGLResourceCacheEntry *)value;
        guint64 frames_since_use = stitch->current_frame_number - entry->last_access_frame;
        
        if (frames_since_use > ttl_threshold) {
            keys_to_remove = g_list_prepend(keys_to_remove, key);
            
            if (entry->registered) {
#ifdef __aarch64__
                cuGraphicsUnregisterResource(entry->cuda_resource);
#endif
            }
            removed_count++;
        }
    }
    
    for (GList *l = keys_to_remove; l != NULL; l = l->next) {
        g_hash_table_remove(stitch->egl_resource_cache, l->data);
    }
    g_list_free(keys_to_remove);
    
    if (removed_count > 0) {
        LOG_DEBUG(stitch, "Cache cleanup: removed %u entries", removed_count);
    }
    
    g_mutex_unlock(&stitch->egl_lock);
}

static gboolean find_frame_indices(GstNvdsStitch *stitch, 
                                  NvDsBatchMeta *batch_meta,
                                  FrameIndices *indices)
{
    // if (stitch->cached_indices.left_index >= 0 && 
    //     stitch->cached_indices.right_index >= 0) {
    //     *indices = stitch->cached_indices;
    //     return TRUE;
    // }
    
    indices->left_index = -1;
    indices->right_index = -1;
    gint frame_index = 0;
    
    for (GList *l = batch_meta->frame_meta_list; l != NULL; l = l->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l->data;
        if (frame_meta) {
            if (frame_meta->source_id == stitch->left_source_id) {
                indices->left_index = frame_index;
            } else if (frame_meta->source_id == stitch->right_source_id) {
                indices->right_index = frame_index;
            }
        }
        frame_index++;
        
        if (indices->left_index >= 0 && indices->right_index >= 0) {
            stitch->cached_indices = *indices;
            return TRUE;
        }
    }
    
    return (indices->left_index >= 0 && indices->right_index >= 0);
}

static gboolean get_or_register_egl_resource(GstNvdsStitch *stitch,
                                            gpointer egl_image,
                                            gboolean is_write,
                                            CUgraphicsResource *resource,
                                            CUeglFrame *frame)
{
#ifdef __aarch64__
    EGLCacheKey lookup_key = { egl_image, is_write };
    EGLResourceCacheEntry *cache_entry;
    
    g_mutex_lock(&stitch->egl_lock);
    
    cache_entry = (EGLResourceCacheEntry *)
        g_hash_table_lookup(stitch->egl_resource_cache, &lookup_key);
    
    if (cache_entry && cache_entry->registered) {
        cache_entry->last_access_frame = stitch->current_frame_number;
        *resource = cache_entry->cuda_resource;
        *frame = cache_entry->egl_frame;
        stitch->egl_cache_hits++;
        g_mutex_unlock(&stitch->egl_lock);
        return TRUE;
    }
    
    g_mutex_unlock(&stitch->egl_lock);
    
    CUresult cu_result = cuGraphicsEGLRegisterImage(resource, egl_image, 
        is_write ? CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD : 
                   CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
    
    if (cu_result != CUDA_SUCCESS) {
        LOG_ERROR(stitch, "Failed to register EGL image: %d", cu_result);
        return FALSE;
    }
    
    cu_result = cuGraphicsResourceGetMappedEglFrame(frame, *resource, 0, 0);
    if (cu_result != CUDA_SUCCESS) {
        LOG_ERROR(stitch, "Failed to get mapped frame: %d", cu_result);
        cuGraphicsUnregisterResource(*resource);
        return FALSE;
    }
    
    g_mutex_lock(&stitch->egl_lock);
    
    if (!cache_entry) {
        EGLCacheKey *new_key = g_new(EGLCacheKey, 1);
        new_key->egl_image = egl_image;
        new_key->is_write = is_write;
        
        cache_entry = g_new0(EGLResourceCacheEntry, 1);
        g_hash_table_insert(stitch->egl_resource_cache, new_key, cache_entry);
    }
    
    cache_entry->egl_image = egl_image;
    cache_entry->cuda_resource = *resource;
    cache_entry->egl_frame = *frame;
    cache_entry->registered = TRUE;
    cache_entry->last_access_frame = stitch->current_frame_number;
    
    stitch->egl_register_count++;
    g_mutex_unlock(&stitch->egl_lock);
#endif
    
    return TRUE;
}

/* ============================================================================
 * Buffer Pool Management
 * ============================================================================ */

static gboolean setup_output_buffer_pool(GstNvdsStitch *stitch)
{
    if (stitch->output_pool && stitch->pool_configured) {
        return TRUE;
    }
    
    LOG_INFO(stitch, "Setting up output buffer pool for panorama (%dx%d)",
             stitch->output_width, stitch->output_height);

    if (!stitch->output_pool) {
        stitch->output_pool = gst_nvds_buffer_pool_new();
        if (!stitch->output_pool) {
            LOG_ERROR(stitch, "Failed to create buffer pool");
            return FALSE;
        }
    }

    GstAllocator *allocator = gst_nvdsstitch_allocator_new(
        stitch->output_width,
        stitch->output_height,
        stitch->gpu_id
    );
    
    if (!allocator) {
        LOG_ERROR(stitch, "Failed to create allocator");
        gst_object_unref(stitch->output_pool);
        stitch->output_pool = NULL;
        return FALSE;
    }
    
    GstStructure *config = gst_buffer_pool_get_config(stitch->output_pool);
    
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",
        "width", G_TYPE_INT, stitch->output_width,
        "height", G_TYPE_INT, stitch->output_height,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    gst_caps_set_features(caps, 0, gst_caps_features_new("memory:NVMM", NULL));
    
    gst_buffer_pool_config_set_params(config, caps, sizeof(NvBufSurface), 
                                      NvdsStitchConfig::POOL_MIN_BUFFERS, 
                                      NvdsStitchConfig::POOL_MAX_BUFFERS);
    gst_buffer_pool_config_set_allocator(config, allocator, NULL);
    
    gst_structure_set(config,
        "memtype", G_TYPE_UINT, NvdsStitchConfig::POOL_MEMTYPE,
        "gpu-id", G_TYPE_UINT, stitch->gpu_id,
        "batch-size", G_TYPE_UINT, 1,
        NULL);
    
    gst_caps_unref(caps);
    gst_object_unref(allocator);
    
    if (!gst_buffer_pool_set_config(stitch->output_pool, config)) {
        LOG_ERROR(stitch, "Failed to set buffer pool config");
        gst_object_unref(stitch->output_pool);
        stitch->output_pool = NULL;
        return FALSE;
    }
    
    if (!gst_buffer_pool_set_active(stitch->output_pool, TRUE)) {
        LOG_ERROR(stitch, "Failed to activate buffer pool");
        gst_object_unref(stitch->output_pool);
        stitch->output_pool = NULL;
        return FALSE;
    }
    
    stitch->pool_configured = TRUE;
    LOG_INFO(stitch, "Output buffer pool configured successfully for panorama");
    return TRUE;
}

/* ============================================================================
 * Stitching Methods
 * ============================================================================ */

static gboolean copy_to_intermediate_buffers(GstNvdsStitch *stitch, 
                                            NvBufSurface *input_surface,
                                            const FrameIndices *indices)
{
    NvBufSurfTransformParams transform_params;
    NvBufSurfTransformRect src_rect, dst_rect;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransform_Error err;
    
    if (!stitch || !input_surface || !indices) {
        LOG_ERROR(stitch, "Invalid parameters to copy_to_intermediate_buffers");
        return FALSE;
    }
    
    if (indices->left_index < 0 || indices->left_index >= (gint)input_surface->numFilled ||
        indices->right_index < 0 || indices->right_index >= (gint)input_surface->numFilled) {
        LOG_ERROR(stitch, "Invalid frame indices: left=%d, right=%d, numFilled=%d",
                  indices->left_index, indices->right_index, input_surface->numFilled);
        return FALSE;
    }
    
    memset(&transform_params, 0, sizeof(transform_params));
    memset(&transform_config_params, 0, sizeof(transform_config_params));
    
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;

    // Используем VIC для копирования буферов, освобождая GPU для stitching (+18% FPS)
    transform_config_params.compute_mode = NvBufSurfTransformCompute_VIC;
    transform_config_params.gpu_id = stitch->gpu_id;
    
    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        LOG_ERROR(stitch, "Failed to set transform session params: %d", err);
        return FALSE;
    }
    
    // Копируем левый кадр
    NvBufSurface temp_left_surface;
    memcpy(&temp_left_surface, input_surface, sizeof(NvBufSurface));
    temp_left_surface.surfaceList = &input_surface->surfaceList[indices->left_index];
    temp_left_surface.numFilled = 1;
    temp_left_surface.batchSize = 1;
    
    guint src_width = input_surface->surfaceList[indices->left_index].width;
    guint src_height = input_surface->surfaceList[indices->left_index].height;
    guint dst_width = stitch->intermediate_left_surf->surfaceList[0].width;
    guint dst_height = stitch->intermediate_left_surf->surfaceList[0].height;
    
    src_rect = {0, 0, src_width, src_height};
    dst_rect = {0, 0, dst_width, dst_height};
    
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    
    err = NvBufSurfTransform(&temp_left_surface, stitch->intermediate_left_surf, 
                             &transform_params);
    
    if (err != NvBufSurfTransformError_Success) {
        LOG_ERROR(stitch, "Failed to copy left frame: %d", err);
        return FALSE;
    }
    
    // Копируем правый кадр
    NvBufSurface temp_right_surface;
    memcpy(&temp_right_surface, input_surface, sizeof(NvBufSurface));
    temp_right_surface.surfaceList = &input_surface->surfaceList[indices->right_index];
    temp_right_surface.numFilled = 1;
    temp_right_surface.batchSize = 1;
    
    err = NvBufSurfTransform(&temp_right_surface, stitch->intermediate_right_surf, 
                             &transform_params);
    
    if (err != NvBufSurfTransformError_Success) {
        LOG_ERROR(stitch, "Failed to copy right frame: %d", err);
        return FALSE;
    }
    
    return TRUE;
}

// Замените функцию panorama_stitch_frames на эту (начиная со строки ~565):

static gboolean panorama_stitch_frames(GstNvdsStitch *stitch, 
                                       NvBufSurface *output_surface)
{
    if (!stitch->warp_maps_loaded) {
        LOG_WARNING(stitch, "Panorama LUT maps not loaded");
        return FALSE;
    }
    
    LOG_DEBUG(stitch, "Using panorama kernel for 360° stitching");
    
    NvBufSurfaceParams *left_params = &stitch->intermediate_left_surf->surfaceList[0];
    NvBufSurfaceParams *right_params = &stitch->intermediate_right_surf->surfaceList[0];
    NvBufSurfaceParams *output_params = &output_surface->surfaceList[0];
    
    cudaError_t err;
    
    stitch->kernel_config.input_width = left_params->width;
    stitch->kernel_config.input_height = left_params->height;
    stitch->kernel_config.input_pitch = left_params->planeParams.pitch[0];
    stitch->kernel_config.output_width = output_params->width;
    stitch->kernel_config.output_height = output_params->height;
    stitch->kernel_config.output_pitch = output_params->planeParams.pitch[0];

    // Цветокоррекция БЕЗ синхронизации - асинхронно
    if (stitch->current_frame_number % 30 == 0) {
        update_color_correction_simple(
            (const unsigned char*)left_params->dataPtr,
            (const unsigned char*)right_params->dataPtr,
            stitch->weight_left_gpu,
            stitch->weight_right_gpu,
            output_params->width,
            output_params->height,
            output_params->planeParams.pitch[0],
            stitch->cuda_stream
        );
        // НЕ ЖДЁМ результат цветокоррекции!
        LOG_DEBUG(stitch, "Color correction update launched at frame %lu", 
                  stitch->current_frame_number);
    }

    // Запускаем kernel (с VIC оптимизацией для буферов)
    err = launch_panorama_kernel(
        (const unsigned char*)left_params->dataPtr,
        (const unsigned char*)right_params->dataPtr,
        (unsigned char*)output_params->dataPtr,
        stitch->warp_left_x_gpu,
        stitch->warp_left_y_gpu,
        stitch->warp_right_x_gpu,
        stitch->warp_right_y_gpu,
        stitch->weight_left_gpu,
        stitch->weight_right_gpu,
        &stitch->kernel_config,
        stitch->cuda_stream
    );
    
    if (err != cudaSuccess) {
        LOG_ERROR(stitch, "Panorama kernel failed: %s", cudaGetErrorString(err));
        reset_cuda_state(stitch);  // ✅ Очистка CUDA состояния
        return FALSE;
    }

    // НЕ синхронизируем здесь - сделаем это позже через событие
    return TRUE;
}

#ifdef __aarch64__
static gboolean panorama_stitch_frames_egl(GstNvdsStitch *stitch, 
                                           gint output_buffer_idx)
{
    stitch->current_frame_number++;
    
    if (stitch->current_frame_number % CACHE_CLEANUP_INTERVAL == 0) {
        cleanup_stale_cache_entries(stitch);
    }
    
    gboolean success = FALSE;
    cudaError_t err;
    CUgraphicsResource resources[3] = {NULL, NULL, NULL};
    CUeglFrame frames[3];
    
    if (!get_or_register_egl_resource(stitch, 
        stitch->intermediate_left_surf->surfaceList[0].mappedAddr.eglImage,
        FALSE, &resources[0], &frames[0])) {
        LOG_ERROR(stitch, "Failed to get left intermediate resource from cache");
        return FALSE;
    }
    
    if (!get_or_register_egl_resource(stitch,
        stitch->intermediate_right_surf->surfaceList[0].mappedAddr.eglImage,
        FALSE, &resources[1], &frames[1])) {
        LOG_ERROR(stitch, "Failed to get right intermediate resource from cache");
        return FALSE;
    }
    
    if (!get_or_register_egl_resource(stitch,
        stitch->output_pool_fixed.surfaces[output_buffer_idx]->surfaceList[0].mappedAddr.eglImage,
        TRUE, &resources[2], &frames[2])) {
        LOG_ERROR(stitch, "Failed to get output buffer %d from cache", output_buffer_idx);
        return FALSE;
    }
    
    NvBufSurface *output_surface = stitch->output_pool_fixed.surfaces[output_buffer_idx];
    
    stitch->kernel_config.input_width = stitch->intermediate_left_surf->surfaceList[0].width;
    stitch->kernel_config.input_height = stitch->intermediate_left_surf->surfaceList[0].height;
    stitch->kernel_config.input_pitch = stitch->intermediate_left_surf->surfaceList[0].planeParams.pitch[0];
    stitch->kernel_config.output_width = output_surface->surfaceList[0].width;
    stitch->kernel_config.output_height = output_surface->surfaceList[0].height;
    stitch->kernel_config.output_pitch = output_surface->surfaceList[0].planeParams.pitch[0];
    
    // Цветокоррекция БЕЗ синхронизации
    if (stitch->current_frame_number % 30 == 0) {
        update_color_correction_simple(
            (const unsigned char*)frames[0].frame.pPitch[0],
            (const unsigned char*)frames[1].frame.pPitch[0],
            stitch->weight_left_gpu,
            stitch->weight_right_gpu,
            output_surface->surfaceList[0].width,
            output_surface->surfaceList[0].height,
            output_surface->surfaceList[0].planeParams.pitch[0],
            stitch->cuda_stream
        );
        // НЕ ЖДЁМ результат!
        LOG_DEBUG(stitch, "EGL: Color correction update launched at frame %lu", 
                  stitch->current_frame_number);
    }

    // Запускаем kernel (с VIC оптимизацией для буферов)
    err = launch_panorama_kernel(
        (const unsigned char*)frames[0].frame.pPitch[0],
        (const unsigned char*)frames[1].frame.pPitch[0],
        (unsigned char*)frames[2].frame.pPitch[0],
        stitch->warp_left_x_gpu,
        stitch->warp_left_y_gpu,
        stitch->warp_right_x_gpu,
        stitch->warp_right_y_gpu,
        stitch->weight_left_gpu,
        stitch->weight_right_gpu,
        &stitch->kernel_config,
        stitch->cuda_stream
    );
    
    if (err == cudaSuccess) {
        success = TRUE;

        if (stitch->current_frame_number % 300 == 0) {
            LOG_INFO(stitch, "✅ Panorama stitching: frame %lu processed",
                     stitch->current_frame_number);
        }
    } else {
        LOG_ERROR(stitch, "Panorama kernel failed: %s", cudaGetErrorString(err));
        reset_cuda_state(stitch);  // ✅ Очистка CUDA состояния
    }

    // НЕ синхронизируем здесь - сделаем это через событие
    return success;
}
#else

static gboolean panorama_stitch_frames_egl(GstNvdsStitch *stitch, 
                                           gint output_buffer_idx)
{
    NvBufSurface *output_surface = stitch->output_pool_fixed.surfaces[output_buffer_idx];
    return panorama_stitch_frames(stitch, output_surface);
}
#endif

/* ============================================================================
 * Main Processing Function
 * ============================================================================ */

static GstFlowReturn gst_nvds_stitch_submit_input_buffer(GstBaseTransform *btrans, 
                                                      gboolean discont G_GNUC_UNUSED, 
                                                      GstBuffer *inbuf)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(btrans);
    GstFlowReturn flow_ret = GST_FLOW_OK;

    stitch->current_input = inbuf;
    
    if (!stitch->pool_configured) {
        if (!setup_output_buffer_pool(stitch)) {
            LOG_ERROR(stitch, "Failed to setup output buffer pool");
            return GST_FLOW_ERROR;
        }
        
        if (!setup_intermediate_buffer_pool(stitch)) {
            LOG_ERROR(stitch, "Failed to setup intermediate buffer pool");
            return GST_FLOW_ERROR;
        }
        
        if (!setup_fixed_output_pool(stitch)) {
            LOG_ERROR(stitch, "Failed to setup fixed output buffer pool");
            return GST_FLOW_ERROR;
        }
    }
    
    cudaSetDevice(stitch->gpu_id);
    
    GstMapInfo in_map;
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        LOG_ERROR(stitch, "Failed to map input buffer");
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    NvBufSurface *input_surface = (NvBufSurface *)in_map.data;
    CHECK_NULL_RET_FLOW(input_surface, stitch, "input_surface from map");

    if (input_surface->numFilled != 2) {
        LOG_WARNING(stitch, "Incomplete batch: expected 2 frames, got %d)",
                    input_surface->numFilled);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (!batch_meta) {
        LOG_ERROR(stitch, "Failed to get batch metadata");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    FrameIndices indices;
    if (!find_frame_indices(stitch, batch_meta, &indices)) {
        LOG_ERROR(stitch, "Failed to find frame indices");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    if (!copy_to_intermediate_buffers(stitch, input_surface, &indices)) {
        LOG_WARNING(stitch, "Failed to copy buffers");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    g_mutex_lock(&stitch->output_pool_fixed.mutex);
    gint buf_idx = stitch->output_pool_fixed.current_index;
    GstBuffer *pool_buf = stitch->output_pool_fixed.buffers[buf_idx];
    NvBufSurface *output_surface = stitch->output_pool_fixed.surfaces[buf_idx];

    // ✅ Проверка NULL для критических указателей
    if (!pool_buf || !output_surface) {
        LOG_ERROR(stitch, "NULL in output pool: pool_buf=%p, output_surface=%p",
                  pool_buf, output_surface);
        g_mutex_unlock(&stitch->output_pool_fixed.mutex);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    GstBuffer *output_buf = gst_buffer_new();
    GstMemory *mem = gst_buffer_peek_memory(pool_buf, 0);
    gst_buffer_append_memory(output_buf, gst_memory_ref(mem));
    
    stitch->output_pool_fixed.current_index = (buf_idx + 1) % FIXED_OUTPUT_POOL_SIZE;
    g_mutex_unlock(&stitch->output_pool_fixed.mutex);
    
    output_surface->numFilled = 1;
    
    gboolean stitch_success = FALSE;
    
    if (stitch->use_egl) {
#ifdef __aarch64__
        if (stitch->warp_maps_loaded && 
            stitch->intermediate_left_surf->memType == NVBUF_MEM_SURFACE_ARRAY &&
            stitch->output_pool_fixed.registered[buf_idx]) {
            stitch_success = panorama_stitch_frames_egl(stitch, buf_idx);
        } else
#endif
        if (stitch->warp_maps_loaded) {
            stitch_success = panorama_stitch_frames(stitch, output_surface);
        }
    }
    
    gst_buffer_unmap(inbuf, &in_map);
    
    if (!stitch_success) {
        LOG_WARNING(stitch, "Panorama stitching failed");
        gst_buffer_unref(output_buf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_OK;
    }
    
    // ВАЖНО: Синхронизация через событие ТОЛЬКО для основного kernel
    if (stitch->cuda_stream && stitch->frame_complete_event) {
        // Записываем событие после завершения основного kernel
        cudaError_t err = cudaEventRecord(stitch->frame_complete_event, stitch->cuda_stream);
        if (err != cudaSuccess) {
            LOG_WARNING(stitch, "Failed to record CUDA event: %s", cudaGetErrorString(err));
        }

        // Ждём ТОЛЬКО завершения этого конкретного kernel
        err = cudaEventSynchronize(stitch->frame_complete_event);
        if (err != cudaSuccess) {
            LOG_WARNING(stitch, "Failed to synchronize CUDA event: %s", cudaGetErrorString(err));
        }
    }
    
    GST_BUFFER_PTS(output_buf) = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DTS(output_buf) = GST_BUFFER_DTS(inbuf);
    GST_BUFFER_DURATION(output_buf) = GST_BUFFER_DURATION(inbuf);
    
    gst_buffer_copy_into(output_buf, inbuf, 
                        (GstBufferCopyFlags)(GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_META), 
                        0, -1);
    
    NvDsBatchMeta *output_batch_meta = gst_buffer_get_nvds_batch_meta(output_buf);
    if (output_batch_meta) {
        output_batch_meta->num_frames_in_batch = 1;
        output_batch_meta->max_frames_in_batch = 1;
        
        while (g_list_length(output_batch_meta->frame_meta_list) > 1) {
            GList *last = g_list_last(output_batch_meta->frame_meta_list);
            output_batch_meta->frame_meta_list = 
                g_list_remove_link(output_batch_meta->frame_meta_list, last);
            g_list_free(last);
        }
        
        NvDsFrameMeta *frame_meta =
            (NvDsFrameMeta *)output_batch_meta->frame_meta_list->data;
        if (frame_meta) {
            frame_meta->source_id = 99;
            frame_meta->source_frame_width = stitch->output_width;
            frame_meta->source_frame_height = stitch->output_height;
            frame_meta->surface_index = 0;
            frame_meta->num_surfaces_per_frame = 1;
        }
    }
    
    // Отправляем буфер в pipeline
    flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(btrans), output_buf);
    
    gst_buffer_unref(inbuf);
    
    stitch->current_frame_number++;
    if (stitch->current_frame_number % 300 == 0) {
        LOG_INFO(stitch, "Panorama: Processed %lu frames", 
                stitch->current_frame_number);
    }
    
    stitch->last_flow_ret = flow_ret;
    
    return flow_ret;
}

static GstFlowReturn 
gst_nvds_stitch_generate_output(GstBaseTransform *btrans, GstBuffer **outbuf)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(btrans);
    *outbuf = NULL;
    return stitch->last_flow_ret;
}

/* ============================================================================
 * Caps and Configuration
 * ============================================================================ */

static GstCaps* gst_nvds_stitch_transform_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
                                               GstCaps *caps,
                                               GstCaps *filter)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(trans);
    GstCaps *other_caps = gst_caps_copy(caps);
    GstStructure *structure = gst_caps_get_structure(other_caps, 0);

    if (direction == GST_PAD_SINK) {
        gst_structure_set(structure,
                         "width", G_TYPE_INT, stitch->output_width,
                         "height", G_TYPE_INT, stitch->output_height,
                         NULL);
    }
    
    if (filter) {
        GstCaps *intersect = gst_caps_intersect(other_caps, filter);
        gst_caps_unref(other_caps);
        other_caps = intersect;
    }
    
    return other_caps;
}

/* ============================================================================
 * Lifecycle Methods
 * ============================================================================ */

static gboolean gst_nvds_stitch_start(GstBaseTransform *trans)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(trans);
    
    LOG_INFO(stitch, "Starting nvdsstitch plugin - PANORAMA MODE");

    // ВАЛИДАЦИЯ: проверяем что размеры панорамы установлены через properties
    if (stitch->output_width == 0 || stitch->output_height == 0) {
        LOG_ERROR(stitch, "❌ ОШИБКА: panorama-width и panorama-height ОБЯЗАТЕЛЬНЫ!");
        LOG_ERROR(stitch, "   Добавьте в pipeline: panorama-width=6528 panorama-height=1800");
        return FALSE;
    }

    if (cudaSetDevice(stitch->gpu_id) != cudaSuccess) {
        LOG_ERROR(stitch, "Failed to set CUDA device %d", stitch->gpu_id);
        return FALSE;
    }
    
    if (cudaStreamCreate(&stitch->cuda_stream) != cudaSuccess) {
        LOG_ERROR(stitch, "Failed to create CUDA stream");
        return FALSE;
    }

    if (cudaEventCreateWithFlags(&stitch->frame_complete_event,
                                 cudaEventDisableTiming) != cudaSuccess) {
        LOG_ERROR(stitch, "Failed to create CUDA event for frame synchronization");
        return FALSE;
    }

    if (init_color_correction() != cudaSuccess) {
        LOG_ERROR(stitch, "Failed to initialize color correction");
        return FALSE;
    }
    
    // Загружаем LUT карты и веса для панорамы
    std::string left_x_path = NvdsStitchConfig::getWarpLeftXPath();
    std::string left_y_path = NvdsStitchConfig::getWarpLeftYPath();
    std::string right_x_path = NvdsStitchConfig::getWarpRightXPath();
    std::string right_y_path = NvdsStitchConfig::getWarpRightYPath();
    std::string weight_left_path = NvdsStitchConfig::getWeightLeftPath();
    std::string weight_right_path = NvdsStitchConfig::getWeightRightPath();
    
    LOG_INFO(stitch, "Loading panorama LUT maps from %s (size: %dx%d)",
             NvdsStitchConfig::WARP_MAPS_DIR,
             stitch->output_width, stitch->output_height);

    cudaError_t err = load_panorama_luts(
        left_x_path.c_str(),
        left_y_path.c_str(),
        right_x_path.c_str(),
        right_y_path.c_str(),
        weight_left_path.c_str(),
        weight_right_path.c_str(),
        &stitch->warp_left_x_gpu,
        &stitch->warp_left_y_gpu,
        &stitch->warp_right_x_gpu,
        &stitch->warp_right_y_gpu,
        &stitch->weight_left_gpu,
        &stitch->weight_right_gpu,
        stitch->output_width,   // Используем динамический размер из property!
        stitch->output_height   // Используем динамический размер из property!
    );

    stitch->warp_maps_loaded = (err == cudaSuccess);
    if (stitch->warp_maps_loaded) {
        LOG_INFO(stitch, "Panorama LUT maps loaded successfully");
        stitch->kernel_config.warp_width = stitch->output_width;   // Динамический размер!
        stitch->kernel_config.warp_height = stitch->output_height; // Динамический размер!
    } else {
        LOG_ERROR(stitch, "Failed to load panorama LUT maps");
        return FALSE;
    }
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    LOG_INFO(stitch, "=== Panorama Stitch Configuration ===");
    LOG_INFO(stitch, "Output: %dx%d (Equirectangular)",
             stitch->output_width, stitch->output_height);
    LOG_INFO(stitch, "GPU: %d", stitch->gpu_id);
    LOG_INFO(stitch, "=====================================");
    
    return TRUE;
}

static gboolean gst_nvds_stitch_stop(GstBaseTransform *trans)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(trans);
    
    LOG_INFO(stitch, "Stopping nvdsstitch plugin");
    
    if (stitch->cuda_stream) {
        cudaStreamSynchronize(stitch->cuda_stream);
    }
    
    if (stitch->warp_maps_loaded) {
        LOG_INFO(stitch, "Freeing panorama LUT maps");
        free_panorama_luts(
            stitch->warp_left_x_gpu,
            stitch->warp_left_y_gpu,
            stitch->warp_right_x_gpu,
            stitch->warp_right_y_gpu,
            stitch->weight_left_gpu,
            stitch->weight_right_gpu
        );
        stitch->warp_maps_loaded = FALSE;
    }
    
    if (stitch->egl_resource_cache) {
        g_hash_table_destroy(stitch->egl_resource_cache);
        stitch->egl_resource_cache = NULL;
    }
    
    if (stitch->cuda_stream) {
        cudaStreamDestroy(stitch->cuda_stream);
        stitch->cuda_stream = NULL;
    }

    if (stitch->frame_complete_event) {
        cudaEventDestroy(stitch->frame_complete_event);
        stitch->frame_complete_event = NULL;
    }

    if (stitch->intermediate_pool) {
        gst_buffer_pool_set_active(stitch->intermediate_pool, FALSE);
        stitch->intermediate_left = NULL;
        stitch->intermediate_right = NULL;
        stitch->intermediate_left_surf = NULL;
        stitch->intermediate_right_surf = NULL;
        gst_object_unref(stitch->intermediate_pool);
        stitch->intermediate_pool = NULL;
    }
    
    g_mutex_lock(&stitch->output_pool_fixed.mutex);
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        stitch->output_pool_fixed.buffers[i] = NULL;
        stitch->output_pool_fixed.surfaces[i] = NULL;
        stitch->output_pool_fixed.registered[i] = FALSE;
    }
    g_mutex_unlock(&stitch->output_pool_fixed.mutex);
    g_mutex_clear(&stitch->output_pool_fixed.mutex);
    
    if (stitch->output_pool) {
        gst_buffer_pool_set_active(stitch->output_pool, FALSE);
        gst_object_unref(stitch->output_pool);
        stitch->output_pool = NULL;
    }
    
    g_mutex_clear(&stitch->egl_lock);
    
    stitch->pool_configured = FALSE;
    stitch->current_frame_number = 0;
    
    LOG_INFO(stitch, "nvdsstitch plugin stopped successfully");
    
    return TRUE;
}

/* ============================================================================
 * Properties
 * ============================================================================ */

static void gst_nvds_stitch_set_property(GObject *object, guint prop_id,
                                         const GValue *value, GParamSpec *pspec)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(object);
    
    switch (prop_id) {
        case PROP_LEFT_SOURCE_ID:
            stitch->left_source_id = g_value_get_uint(value);
            stitch->cached_indices.left_index = -1;
            break;
        case PROP_RIGHT_SOURCE_ID:
            stitch->right_source_id = g_value_get_uint(value);
            stitch->cached_indices.right_index = -1;
            break;
        case PROP_GPU_ID:
            stitch->gpu_id = g_value_get_uint(value);
            break;
        case PROP_USE_EGL:
            stitch->use_egl = g_value_get_boolean(value);
            break;
        case PROP_PANORAMA_WIDTH:
            stitch->output_width = g_value_get_uint(value);
            stitch->kernel_config.output_width = stitch->output_width;
            stitch->kernel_config.warp_width = stitch->output_width;
            stitch->kernel_config.output_pitch = NvdsStitchConfig::calculatePitch(stitch->output_width);
            break;
        case PROP_PANORAMA_HEIGHT:
            stitch->output_height = g_value_get_uint(value);
            stitch->kernel_config.output_height = stitch->output_height;
            stitch->kernel_config.warp_height = stitch->output_height;
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvds_stitch_get_property(GObject *object, guint prop_id,
                                         GValue *value, GParamSpec *pspec)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(object);
    
    switch (prop_id) {
        case PROP_LEFT_SOURCE_ID:
            g_value_set_uint(value, stitch->left_source_id);
            break;
        case PROP_RIGHT_SOURCE_ID:
            g_value_set_uint(value, stitch->right_source_id);
            break;
        case PROP_GPU_ID:
            g_value_set_uint(value, stitch->gpu_id);
            break;
        case PROP_USE_EGL:
            g_value_set_boolean(value, stitch->use_egl);
            break;
        case PROP_PANORAMA_WIDTH:
            g_value_set_uint(value, stitch->output_width);
            break;
        case PROP_PANORAMA_HEIGHT:
            g_value_set_uint(value, stitch->output_height);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvds_stitch_finalize(GObject *object)
{
    GstNvdsStitch *stitch = GST_NVDS_STITCH(object);

    LOG_INFO(stitch, "Finalizing nvdsstitch");

    // Очистка texture resources

    G_OBJECT_CLASS(gst_nvds_stitch_parent_class)->finalize(object);
}

/* ============================================================================
 * Class Initialization
 * ============================================================================ */

static void gst_nvds_stitch_class_init(GstNvdsStitchClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gst_element_class_set_static_metadata(gstelement_class,
        "NvDsStitch Panorama", "Video/Filter",
        "NVIDIA DeepStream Panorama Stitching Plugin", "NVIDIA");

    gst_element_class_add_static_pad_template(gstelement_class, &sink_template);
    gst_element_class_add_static_pad_template(gstelement_class, &src_template);

    gobject_class->set_property = gst_nvds_stitch_set_property;
    gobject_class->get_property = gst_nvds_stitch_get_property;
    gobject_class->finalize = gst_nvds_stitch_finalize;

    g_object_class_install_property(gobject_class, PROP_LEFT_SOURCE_ID,
        g_param_spec_uint("left-source-id", "Left Source ID",
                         "Source ID for left frame", 0, G_MAXUINT, 
                         NvdsStitchConfig::LEFT_SOURCE_ID,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_RIGHT_SOURCE_ID,
        g_param_spec_uint("right-source-id", "Right Source ID",
                         "Source ID for right frame", 0, G_MAXUINT, 
                         NvdsStitchConfig::RIGHT_SOURCE_ID,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID",
                         "GPU device ID", 0, G_MAXUINT, 
                         NvdsStitchConfig::GPU_ID,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_USE_EGL,
        g_param_spec_boolean("use-egl", "Use EGL",
                            "Use EGL interop on Jetson",
                            TRUE,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_WIDTH,
        g_param_spec_uint("panorama-width", "Panorama Width",
                         "Output panorama width (REQUIRED!)", 0, 10000,
                         0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО передавать через properties!
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_HEIGHT,
        g_param_spec_uint("panorama-height", "Panorama Height",
                         "Output panorama height (REQUIRED!)", 0, 10000,
                         0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО передавать через properties!
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gstbasetransform_class->submit_input_buffer = GST_DEBUG_FUNCPTR(gst_nvds_stitch_submit_input_buffer);
    gstbasetransform_class->generate_output = GST_DEBUG_FUNCPTR(gst_nvds_stitch_generate_output);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_nvds_stitch_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_nvds_stitch_stop);
    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvds_stitch_transform_caps);
}

static void gst_nvds_stitch_init(GstNvdsStitch *stitch)
{
    stitch->left_source_id = NvdsStitchConfig::LEFT_SOURCE_ID;
    stitch->right_source_id = NvdsStitchConfig::RIGHT_SOURCE_ID;
    stitch->output_width = 0;   // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
    stitch->output_height = 0;  // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
    stitch->gpu_id = NvdsStitchConfig::GPU_ID;
    
    // Для панорамы не нужны crop параметры
    stitch->crop_top = 0;
    stitch->crop_bottom = 0;
    stitch->crop_sides = 0;
    stitch->overlap = 0;
    
    stitch->current_input = NULL;
    stitch->output_pool = NULL;
    stitch->pool_configured = FALSE;

    stitch->intermediate_pool = NULL;
    stitch->intermediate_left = NULL;
    stitch->intermediate_right = NULL;
    stitch->intermediate_left_surf = NULL;
    stitch->intermediate_right_surf = NULL;

    stitch->warp_left_x_gpu = NULL;
    stitch->warp_left_y_gpu = NULL;
    stitch->warp_right_x_gpu = NULL;
    stitch->warp_right_y_gpu = NULL;
    stitch->weight_left_gpu = NULL;
    stitch->weight_right_gpu = NULL;
    stitch->warp_maps_loaded = FALSE;
    stitch->cuda_stream = NULL;
    stitch->frame_complete_event = NULL;

    // Инициализация входных параметров (константы)
    stitch->kernel_config.input_width = NvdsStitchConfig::INPUT_WIDTH;
    stitch->kernel_config.input_height = NvdsStitchConfig::INPUT_HEIGHT;
    stitch->kernel_config.input_pitch = NvdsStitchConfig::getInputPitch();

    // Выходные параметры устанавливаются через properties в set_property()!

    stitch->last_flow_ret = GST_FLOW_OK;
    
    stitch->use_egl = TRUE;
    stitch->egl_resource_cache = g_hash_table_new_full(
        egl_cache_key_hash, egl_cache_key_equal, 
        egl_cache_key_free, egl_cache_entry_free);
    g_mutex_init(&stitch->egl_lock);
    
    stitch->egl_map_count = 0;
    stitch->egl_register_count = 0;
    stitch->egl_cache_hits = 0;
    stitch->current_frame_number = 0;
    
    stitch->cached_indices.left_index = -1;
    stitch->cached_indices.right_index = -1;
    
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(stitch), FALSE);
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(stitch), FALSE);
}

static gboolean nvds_stitch_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvds_stitch_debug, "nvdsstitch", 0, 
                           "NVIDIA DeepStream Panorama Stitch Plugin Debug");
    
    return gst_element_register(plugin, "nvdsstitch", GST_RANK_PRIMARY, 
                               GST_TYPE_NVDS_STITCH);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsstitch,
    "NVIDIA DeepStream Panorama Stitching Plugin",
    nvds_stitch_plugin_init,
    "1.0",
    "Proprietary",
    "nvdsstitch",
    "https://developer.nvidia.com/"
)
