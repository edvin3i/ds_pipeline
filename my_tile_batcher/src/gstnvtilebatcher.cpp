// gstnvtilebatcher.cpp - Версия только для Jetson (ИСПРАВЛЕНО)
#include "gstnvtilebatcher.h"
#include "gstnvtilebatcher_allocator.h"
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"
#include <string.h>
#include <cstdio>

GST_DEBUG_CATEGORY_STATIC(gst_nvtilebatcher_debug);
#define GST_CAT_DEFAULT gst_nvtilebatcher_debug

/* External CUDA functions */
extern "C" {
    int cuda_init_tile_positions(int positions[][2]);
    int cuda_set_tile_pointers(void** tile_ptrs);
    int cuda_extract_tiles(void* src_gpu,
                          int src_width, int src_height,
                          int src_pitch, int tile_pitch,
                          cudaStream_t stream);
    int cuda_extract_tiles_nv12(void* src_y_gpu, void* src_uv_gpu,
                               int src_width, int src_height,
                               int src_pitch_y, int src_pitch_uv,
                               int tile_pitch,
                               cudaStream_t stream);
}

/* Properties */
enum {
    PROP_0,
    PROP_GPU_ID,
    PROP_SILENT,
    PROP_PANORAMA_WIDTH,
    PROP_PANORAMA_HEIGHT,
    PROP_TILE_OFFSET_Y
};

/* Pad templates */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format={ RGBA, NV12 }")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw(memory:NVMM), "
        "format=RGBA, "
        "width=1024, "
        "height=1024"
    )
);

#define gst_nvtilebatcher_parent_class parent_class
G_DEFINE_TYPE(GstNvTileBatcher, gst_nvtilebatcher, GST_TYPE_BASE_TRANSFORM);

/* ============================================================================
 * Tile Position Calculation
 * ============================================================================ */

static void
calculate_tile_positions(GstNvTileBatcher *batcher)
{
    // X позиции тайлов фиксированные (отступ слева 192px, затем 6 тайлов по 1024px)
    const gint tile_x_positions[TILES_PER_BATCH] = {192, 1216, 2240, 3264, 4288, 5312};

    // Y позиция - используем tile_offset_y (рассчитанный из field_mask.png)
    // БЫЛО: gint tile_y = (batcher->panorama_height - TILE_HEIGHT) / 2; // симметричное центрирование = 304
    // СТАЛО: используем property tile_offset_y = 434 (из field_mask.png)
    gint tile_y = batcher->tile_offset_y;

    for (int i = 0; i < TILES_PER_BATCH; i++) {
        batcher->tile_positions[i].tile_id = i;
        batcher->tile_positions[i].x = tile_x_positions[i];
        batcher->tile_positions[i].y = tile_y;
    }

    GST_INFO_OBJECT(batcher, "Calculated tile positions for panorama %ux%u: y_offset=%d (from property)",
                    batcher->panorama_width, batcher->panorama_height, tile_y);
}

/* ============================================================================
 * EGL Cache Management для входных буферов
 * ============================================================================ */

static void
egl_cache_entry_free(gpointer data)
{
    EGLResourceCacheEntry *entry = (EGLResourceCacheEntry*)data;
    if (entry && entry->registered) {
        cuGraphicsUnregisterResource(entry->cuda_resource);
        GST_DEBUG("Unregistered EGL resource %p", entry->egl_image);
    }
    g_free(entry);
}



static gboolean
get_or_register_egl_resource(GstNvTileBatcher *batcher,
                            gpointer egl_image,
                            gboolean is_write,
                            CUgraphicsResource *resource,
                            CUeglFrame *frame)
{
    EGLResourceCacheEntry *cache_entry;
    
    g_mutex_lock(&batcher->egl_cache_mutex);
    
    cache_entry = (EGLResourceCacheEntry*)g_hash_table_lookup(batcher->egl_cache, egl_image);
    
    if (cache_entry && cache_entry->registered) {
        cache_entry->last_access_frame = batcher->frame_counter;
        *resource = cache_entry->cuda_resource;
        *frame = cache_entry->egl_frame;
        g_mutex_unlock(&batcher->egl_cache_mutex);
        GST_DEBUG_OBJECT(batcher, "Using cached EGL resource for %p", egl_image);
        return TRUE;
    }
    
    g_mutex_unlock(&batcher->egl_cache_mutex);
    
    // Регистрируем новый ресурс
    CUresult cu_result = cuGraphicsEGLRegisterImage(resource, egl_image,
        is_write ? CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD : 
                   CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
    
    if (cu_result != CUDA_SUCCESS) {
        GST_ERROR_OBJECT(batcher, "Failed to register EGL image %p: %d", 
                        egl_image, cu_result);
        return FALSE;
    }
    
    cu_result = cuGraphicsResourceGetMappedEglFrame(frame, *resource, 0, 0);
    if (cu_result != CUDA_SUCCESS) {
        GST_ERROR_OBJECT(batcher, "Failed to get mapped frame for %p: %d", 
                        egl_image, cu_result);
        cuGraphicsUnregisterResource(*resource);
        return FALSE;
    }
    
    // Добавляем в кеш
    g_mutex_lock(&batcher->egl_cache_mutex);
    
    if (!cache_entry) {
        cache_entry = g_new0(EGLResourceCacheEntry, 1);
        g_hash_table_insert(batcher->egl_cache, egl_image, cache_entry);
    }
    
    cache_entry->egl_image = egl_image;
    cache_entry->cuda_resource = *resource;
    cache_entry->egl_frame = *frame;
    cache_entry->registered = TRUE;
    cache_entry->last_access_frame = batcher->frame_counter;
    
    g_mutex_unlock(&batcher->egl_cache_mutex);
    
    GST_DEBUG_OBJECT(batcher, "Registered new EGL resource for %p", egl_image);
    
    return TRUE;
}

/* ============================================================================
 * Fixed Output Pool Setup
 * ============================================================================ */

static gboolean
setup_fixed_output_pool(GstNvTileBatcher *batcher)
{
    GST_INFO_OBJECT(batcher, "Setting up fixed output buffer pool for Jetson");
    
    g_mutex_init(&batcher->output_pool_fixed.mutex);
    batcher->output_pool_fixed.current_index = 0;
    
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        GstFlowReturn flow_ret;
        
        flow_ret = gst_buffer_pool_acquire_buffer(batcher->output_pool, 
                                                 &batcher->output_pool_fixed.buffers[i], 
                                                 NULL);
        if (flow_ret != GST_FLOW_OK) {
            GST_ERROR_OBJECT(batcher, "Failed to acquire output buffer %d: %s", 
                            i, gst_flow_get_name(flow_ret));
            return FALSE;
        }
        
        GstMapInfo map_info;
        if (!gst_buffer_map(batcher->output_pool_fixed.buffers[i], &map_info, GST_MAP_READWRITE)) {
            GST_ERROR_OBJECT(batcher, "Failed to map output buffer %d", i);
            return FALSE;
        }
        
        batcher->output_pool_fixed.surfaces[i] = (NvBufSurface *)map_info.data;
        gst_buffer_unmap(batcher->output_pool_fixed.buffers[i], &map_info);
        
        NvBufSurface *surface = batcher->output_pool_fixed.surfaces[i];
        
        if (surface->memType != NVBUF_MEM_SURFACE_ARRAY) {
            GST_ERROR_OBJECT(batcher, "Output buffer %d has wrong memory type: %d", 
                            i, surface->memType);
            return FALSE;
        }
        
        if (surface->batchSize != TILES_PER_BATCH) {
            GST_WARNING_OBJECT(batcher, "Buffer %d: batchSize=%d, expected %d", 
                              i, surface->batchSize, TILES_PER_BATCH);
            surface->batchSize = 6;
            surface->numFilled = 6;
        }
        
        gboolean egl_mapped = TRUE;
        for (int j = 0; j < TILES_PER_BATCH; j++) {
            if (!surface->surfaceList[j].mappedAddr.eglImage) {
                GST_DEBUG_OBJECT(batcher, "Buffer %d, tile %d needs EGL mapping", i, j);
                egl_mapped = FALSE;
                break;
            }
        }
        
        if (!egl_mapped) {
            if (NvBufSurfaceMapEglImage(surface, -1) != 0) {
                GST_ERROR_OBJECT(batcher, "Failed to map EGL for output buffer %d", i);
                return FALSE;
            }
        }
        
        for (int j = 0; j < TILES_PER_BATCH; j++) {
            void* egl_image = surface->surfaceList[j].mappedAddr.eglImage;
            if (!egl_image) {
                GST_ERROR_OBJECT(batcher, "No EGL image for tile %d in buffer %d", j, i);
                return FALSE;
            }
            
            CUresult cu_result = cuGraphicsEGLRegisterImage(
                &batcher->output_pool_fixed.cuda_resources[i][j],
                egl_image,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
            );
            
            if (cu_result != CUDA_SUCCESS) {
                GST_ERROR_OBJECT(batcher, "Failed to register tile %d in buffer %d: %d", 
                                j, i, cu_result);
                return FALSE;
            }
            
            cu_result = cuGraphicsResourceGetMappedEglFrame(
                &batcher->output_pool_fixed.egl_frames[i][j],
                batcher->output_pool_fixed.cuda_resources[i][j],
                0, 0
            );
            
            if (cu_result != CUDA_SUCCESS) {
                GST_ERROR_OBJECT(batcher, "Failed to get mapped frame for tile %d in buffer %d: %d", 
                                j, i, cu_result);
                return FALSE;
            }
            
            void* tile_ptr = (void*)batcher->output_pool_fixed.egl_frames[i][j].frame.pPitch[0];
            if (!tile_ptr) {
                GST_ERROR_OBJECT(batcher, "Got NULL pointer for tile %d in buffer %d", j, i);
                return FALSE;
            }
            
            GST_DEBUG_OBJECT(batcher, "Buffer %d, tile %d: ptr=%p, pitch=%d", 
                            i, j, tile_ptr,
                            surface->surfaceList[j].planeParams.pitch[0]);
        }
        
        batcher->output_pool_fixed.registered[i] = TRUE;
        GST_INFO_OBJECT(batcher, "Output buffer %d registered with %d tiles", i, TILES_PER_BATCH);
    }
    
    GST_INFO_OBJECT(batcher, "Fixed output pool ready with %d buffers", FIXED_OUTPUT_POOL_SIZE);
    return TRUE;
}

/* ============================================================================
 * Buffer Pool Setup
 * ============================================================================ */

static gboolean
setup_output_buffer_pool(GstNvTileBatcher *batcher)
{
    if (batcher->pool_configured) {
        return TRUE;
    }
    
    GST_INFO_OBJECT(batcher, "Setting up output buffer pool for Jetson");
    
    batcher->output_pool = gst_buffer_pool_new();
    if (!batcher->output_pool) {
        GST_ERROR_OBJECT(batcher, "Failed to create buffer pool");
        return FALSE;
    }
    
    GstAllocator *allocator = gst_nvtilebatcher_allocator_new(batcher->gpu_id);
    if (!allocator) {
        GST_ERROR_OBJECT(batcher, "Failed to create allocator");
        gst_object_unref(batcher->output_pool);
        batcher->output_pool = NULL;
        return FALSE;
    }
    
    GstStructure *config = gst_buffer_pool_get_config(batcher->output_pool);
    
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",
        "width", G_TYPE_INT, TILE_WIDTH,
        "height", G_TYPE_INT, TILE_HEIGHT,
        "framerate", GST_TYPE_FRACTION, 0, 1,
        NULL);
    gst_caps_set_features(caps, 0, gst_caps_features_new("memory:NVMM", NULL));
    
    gst_buffer_pool_config_set_params(config, caps, sizeof(NvBufSurface), 
                                     FIXED_OUTPUT_POOL_SIZE + 2, 
                                     FIXED_OUTPUT_POOL_SIZE + 4);
    gst_buffer_pool_config_set_allocator(config, allocator, NULL);
    
    gst_structure_set(config,
        "memtype", G_TYPE_UINT, NVBUF_MEM_SURFACE_ARRAY,
        "gpu-id", G_TYPE_UINT, batcher->gpu_id,
        "batch-size", G_TYPE_UINT, TILES_PER_BATCH,
        NULL);
    
    gst_caps_unref(caps);
    gst_object_unref(allocator);
    
    if (!gst_buffer_pool_set_config(batcher->output_pool, config)) {
        GST_ERROR_OBJECT(batcher, "Failed to set buffer pool config");
        gst_object_unref(batcher->output_pool);
        batcher->output_pool = NULL;
        return FALSE;
    }
    
    if (!gst_buffer_pool_set_active(batcher->output_pool, TRUE)) {
        GST_ERROR_OBJECT(batcher, "Failed to activate buffer pool");
        gst_object_unref(batcher->output_pool);
        batcher->output_pool = NULL;
        return FALSE;
    }
    
    batcher->pool_configured = TRUE;
    GST_INFO_OBJECT(batcher, "Output buffer pool configured successfully");
    return TRUE;
}

/* ============================================================================
 * Metadata - ИСПРАВЛЕНО для DS 7.1
 * ============================================================================ */

static void process_and_update_metadata(GstNvTileBatcher *batcher,
                                        GstBuffer *input_buffer,
                                        GstBuffer *output_buffer,
                                        NvBufSurface *output_surface)
{
    (void)output_surface;  // Unused parameter

    // НЕ КОПИРУЕМ metadata! Создаём новый batch_meta с нуля
    // Это избегает проблем с двойным освобождением user_meta

    // Получаем информацию из входного буфера (если есть)
    guint panorama_source_id = 0;
    guint64 panorama_buf_pts = GST_BUFFER_PTS(input_buffer);
    guint64 panorama_ntp_timestamp = 0;

    NvDsBatchMeta *input_batch_meta = gst_buffer_get_nvds_batch_meta(input_buffer);
    if (input_batch_meta && input_batch_meta->frame_meta_list) {
        NvDsFrameMeta *orig_frame = (NvDsFrameMeta *)input_batch_meta->frame_meta_list->data;
        panorama_source_id = orig_frame->source_id;
        panorama_ntp_timestamp = orig_frame->ntp_timestamp;
    }

    // Создаём НОВЫЙ batch_meta для output_buffer
    NvDsBatchMeta *batch_meta = nvds_create_batch_meta(TILES_PER_BATCH);
    if (!batch_meta) {
        GST_ERROR_OBJECT(batcher, "Failed to create batch metadata");
        return;
    }

    // Устанавливаем параметры batch
    batch_meta->max_frames_in_batch = TILES_PER_BATCH;
    batch_meta->num_frames_in_batch = TILES_PER_BATCH;

    // Блокируем для потокобезопасности
    g_rec_mutex_lock(&batch_meta->meta_mutex);
    
    // Создаём frame_meta для каждого тайла
    for (int i = 0; i < TILES_PER_BATCH; i++) {
        NvDsFrameMeta *frame_meta = NULL;

        // Всегда используем пул DeepStream (если пула нет - это ошибка)
        if (!batch_meta->frame_meta_pool) {
            GST_ERROR_OBJECT(batcher,
                "No frame_meta_pool available in batch_meta for tile %d", i);
            g_rec_mutex_unlock(&batch_meta->meta_mutex);
            return;
        }

        frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
        if (!frame_meta) {
            GST_WARNING_OBJECT(batcher,
                "Failed to acquire frame_meta from pool for tile %d", i);
            continue;  // Пропускаем этот тайл
        }
        
        frame_meta->base_meta.batch_meta = batch_meta;
        frame_meta->source_id = panorama_source_id;
        frame_meta->batch_id = i;
        frame_meta->frame_num = batcher->frame_counter;
        frame_meta->buf_pts = panorama_buf_pts;  // ВАЖНО: копируем timestamp!
        frame_meta->ntp_timestamp = panorama_ntp_timestamp;
        frame_meta->source_frame_width = TILE_WIDTH;
        frame_meta->source_frame_height = TILE_HEIGHT;
        frame_meta->surface_index = i;
        frame_meta->num_surfaces_per_frame = 1;

        // Добавляем user_meta с информацией о позиции тайла
        NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

        if (user_meta) {
            TileRegionInfo *tile_info = g_new0(TileRegionInfo, 1);
            tile_info->tile_id = i;
            tile_info->panorama_x = batcher->tile_positions[i].x;
            tile_info->panorama_y = batcher->tile_positions[i].y;
            tile_info->tile_width = TILE_WIDTH;
            tile_info->tile_height = TILE_HEIGHT;

            user_meta->user_meta_data = tile_info;
            user_meta->base_meta.meta_type = NVDS_USER_META;
            user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)tile_region_info_copy;
            user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)tile_region_info_free;

            nvds_add_user_meta_to_frame(frame_meta, user_meta);
        }
        
        batch_meta->frame_meta_list = g_list_append(batch_meta->frame_meta_list, frame_meta);
        // batch_meta->num_frames_in_batch++;
    }
    
    g_rec_mutex_unlock(&batch_meta->meta_mutex);

    // Прикрепляем batch_meta к output_buffer
    NvDsMeta *meta = gst_buffer_add_nvds_meta(output_buffer, batch_meta, NULL, nvds_batch_meta_copy_func, nvds_batch_meta_release_func);

    if (!meta) {
        GST_ERROR_OBJECT(batcher, "❌ FAILED to add nvds_meta to output_buffer!");
        nvds_destroy_batch_meta(batch_meta);
        return;
    }

    // ВАЖНО: Устанавливаем meta_type для batch_meta
    meta->meta_type = NVDS_BATCH_GST_META;

    batch_meta->base_meta.batch_meta = batch_meta;
    batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
    batch_meta->base_meta.release_func = nvds_batch_meta_release_func;

    GST_INFO_OBJECT(batcher, "✅ Created new batch_meta: %d tiles, source_id=%u",
                     batch_meta->num_frames_in_batch, panorama_source_id);

    // Проверка что metadata действительно прикреплена
    NvDsBatchMeta *check_meta = gst_buffer_get_nvds_batch_meta(output_buffer);
    if (!check_meta) {
        GST_ERROR_OBJECT(batcher, "❌ FAILED to get batch_meta from output_buffer!");
    } else {
        GST_INFO_OBJECT(batcher, "✅ Verified: batch_meta is attached, num_frames=%u", check_meta->num_frames_in_batch);
    }
}

/* ============================================================================
 * Main Processing Function
 * ============================================================================ */

static GstFlowReturn
gst_nvtilebatcher_submit_input_buffer(GstBaseTransform *btrans,
                                      gboolean discont G_GNUC_UNUSED,
                                      GstBuffer *inbuf)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(btrans);
    GstFlowReturn flow_ret = GST_FLOW_OK;

    // Валидация: проверяем что panorama размеры заданы через properties
    if (batcher->panorama_width == 0 || batcher->panorama_height == 0) {
        GST_ERROR_OBJECT(batcher, "❌ ОШИБКА: panorama-width и panorama-height ОБЯЗАТЕЛЬНЫ!");
        GST_ERROR_OBJECT(batcher, "   Добавьте в pipeline: panorama-width=6528 panorama-height=1800");
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Инициализация пулов если нужно
    if (!batcher->pool_configured) {
        // Позиции тайлов уже рассчитаны в gst_nvtilebatcher_start()
        if (!setup_output_buffer_pool(batcher)) {
            GST_ERROR_OBJECT(batcher, "Failed to setup output buffer pool");
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }

        if (!setup_fixed_output_pool(batcher)) {
            GST_ERROR_OBJECT(batcher, "Failed to setup fixed output buffer pool");
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }
    }
    
    cudaSetDevice(batcher->gpu_id);
    
    // Маппим входной буфер
    GstMapInfo in_map;
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(batcher, "Failed to map input buffer");
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    NvBufSurface *input_surface = (NvBufSurface *)in_map.data;

    // Проверяем тип памяти
    if (input_surface->memType != NVBUF_MEM_SURFACE_ARRAY) {
        GST_ERROR_OBJECT(batcher, "Input surface is not SURFACE_ARRAY type: %d",
                        input_surface->memType);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Валидация размеров входного буфера (должна быть панорама из properties)
    if (input_surface->surfaceList[0].width != batcher->panorama_width ||
        input_surface->surfaceList[0].height != batcher->panorama_height) {
        GST_ERROR_OBJECT(batcher,
            "Invalid input buffer size: %dx%d (expected %dx%d)",
            input_surface->surfaceList[0].width,
            input_surface->surfaceList[0].height,
            batcher->panorama_width, batcher->panorama_height);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Validate format (accept RGBA or NV12)
    NvBufSurfaceColorFormat input_format = input_surface->surfaceList[0].colorFormat;
    if (input_format != NVBUF_COLOR_FORMAT_RGBA &&
        input_format != NVBUF_COLOR_FORMAT_NV12) {
        GST_ERROR_OBJECT(batcher,
            "Unsupported input format: %d (expected RGBA=%d or NV12=%d)",
            input_format, NVBUF_COLOR_FORMAT_RGBA, NVBUF_COLOR_FORMAT_NV12);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    GST_DEBUG_OBJECT(batcher, "Input format: %s",
                     input_format == NVBUF_COLOR_FORMAT_NV12 ? "NV12" : "RGBA");

    // Получаем CUDA указатель через EGL
    void* src_ptr = NULL;
    if (!input_surface->surfaceList[0].mappedAddr.eglImage) {
        if (NvBufSurfaceMapEglImage(input_surface, 0) != 0) {
            GST_ERROR_OBJECT(batcher, "Failed to map EGL image for input");
            gst_buffer_unmap(inbuf, &in_map);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }

        // Проверяем, что маппинг действительно произошёл
        if (!input_surface->surfaceList[0].mappedAddr.eglImage) {
            GST_ERROR_OBJECT(batcher, "EGL image is NULL after successful mapping");
            gst_buffer_unmap(inbuf, &in_map);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }
    }
    
    CUgraphicsResource resource;
    CUeglFrame frame;
    if (!get_or_register_egl_resource(batcher,
                                     input_surface->surfaceList[0].mappedAddr.eglImage,
                                     FALSE, &resource, &frame)) {
        GST_ERROR_OBJECT(batcher, "Failed to get input CUDA pointer");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    src_ptr = (void*)frame.frame.pPitch[0];
    
    // Получаем выходной буфер из пула (защищено мьютексом)
    g_mutex_lock(&batcher->output_pool_fixed.mutex);
    gint buf_idx = batcher->output_pool_fixed.current_index;
    GstBuffer *pool_buf = batcher->output_pool_fixed.buffers[buf_idx];
    NvBufSurface *output_surface = batcher->output_pool_fixed.surfaces[buf_idx];

    // Устанавливаем параметры batch
    output_surface->batchSize = TILES_PER_BATCH;
    output_surface->numFilled = TILES_PER_BATCH;

    // Создаём новый GstBuffer с reference на память из пула
    // NOTE: GstMemory reference counting защищает буфер от переиспользования
    // пока output_buf существует, поэтому безопасно отпускать mutex здесь
    GstBuffer *output_buf = gst_buffer_new();
    GstMemory *mem = gst_buffer_peek_memory(pool_buf, 0);
    gst_buffer_append_memory(output_buf, gst_memory_ref(mem));

    // Сохраняем указатели на тайлы для CUDA (пока под мьютексом)
    void* tile_pointers[TILES_PER_BATCH];
    for (int i = 0; i < TILES_PER_BATCH; i++) {
        tile_pointers[i] = (void*)batcher->output_pool_fixed.egl_frames[buf_idx][i].frame.pPitch[0];
        if (!tile_pointers[i]) {
            g_mutex_unlock(&batcher->output_pool_fixed.mutex);
            GST_ERROR_OBJECT(batcher, "NULL pointer for tile %d", i);
            gst_buffer_unmap(inbuf, &in_map);
            gst_buffer_unref(output_buf);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }
    }

    // Двигаем указатель на следующий буфер
    batcher->output_pool_fixed.current_index = (buf_idx + 1) % FIXED_OUTPUT_POOL_SIZE;
    g_mutex_unlock(&batcher->output_pool_fixed.mutex);

    // Устанавливаем указатели для CUDA kernel
    if (cuda_set_tile_pointers(tile_pointers) != 0) {
        GST_ERROR_OBJECT(batcher, "Failed to set tile pointers");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(output_buf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Dispatch CUDA kernel based on input format
    int result = -1;
    if (input_format == NVBUF_COLOR_FORMAT_NV12) {
        // NV12 input: Extract Y and UV plane pointers
        unsigned char* src_y_ptr = (unsigned char*)src_ptr;

        // UV plane offset: Y_size = pitch[0] × height
        unsigned char* src_uv_ptr = src_y_ptr +
            (input_surface->surfaceList[0].planeParams.pitch[0] *
             input_surface->surfaceList[0].planeParams.height[0]);

        int pitch_y = input_surface->surfaceList[0].planeParams.pitch[0];
        int pitch_uv = input_surface->surfaceList[0].planeParams.pitch[1];

        GST_DEBUG_OBJECT(batcher,
            "Extracting %d tiles from NV12 panorama %dx%d (Y=%p UV=%p pitch_y=%d pitch_uv=%d)",
            TILES_PER_BATCH,
            input_surface->surfaceList[0].width,
            input_surface->surfaceList[0].height,
            src_y_ptr, src_uv_ptr, pitch_y, pitch_uv);

        result = cuda_extract_tiles_nv12(
            src_y_ptr,
            src_uv_ptr,
            input_surface->surfaceList[0].width,
            input_surface->surfaceList[0].height,
            pitch_y,
            pitch_uv,
            output_surface->surfaceList[0].planeParams.pitch[0],
            batcher->cuda_stream
        );
    } else {
        // RGBA input: Original kernel (backward compatible)
        GST_DEBUG_OBJECT(batcher, "Extracting %d tiles from RGBA panorama %dx%d",
                         TILES_PER_BATCH,
                         input_surface->surfaceList[0].width,
                         input_surface->surfaceList[0].height);

        result = cuda_extract_tiles(
            src_ptr,
            input_surface->surfaceList[0].width,
            input_surface->surfaceList[0].height,
            input_surface->surfaceList[0].planeParams.pitch[0],
            output_surface->surfaceList[0].planeParams.pitch[0],
            batcher->cuda_stream
        );
    }
    
    gst_buffer_unmap(inbuf, &in_map);
    
    if (result != 0) {
        GST_ERROR_OBJECT(batcher, "CUDA extraction failed: %d", result);
        gst_buffer_unref(output_buf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Синхронизация CUDA
    if (batcher->frame_complete_event) {
        cudaError_t cuda_err = cudaEventRecord(batcher->frame_complete_event, batcher->cuda_stream);
        if (cuda_err != cudaSuccess) {
            GST_ERROR_OBJECT(batcher, "CUDA event record failed: %s",
                             cudaGetErrorString(cuda_err));
            gst_buffer_unref(output_buf);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }

        cuda_err = cudaEventSynchronize(batcher->frame_complete_event);
        if (cuda_err != cudaSuccess) {
            GST_ERROR_OBJECT(batcher, "CUDA event synchronization failed: %s",
                             cudaGetErrorString(cuda_err));
            gst_buffer_unref(output_buf);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }
    }
    
    // ВАЖНО: Копируем timestamp и flags ДО работы с метаданными
    GST_BUFFER_PTS(output_buf) = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DTS(output_buf) = GST_BUFFER_DTS(inbuf);
    GST_BUFFER_DURATION(output_buf) = GST_BUFFER_DURATION(inbuf);
    GST_BUFFER_OFFSET(output_buf) = GST_BUFFER_OFFSET(inbuf);
    GST_BUFFER_OFFSET_END(output_buf) = GST_BUFFER_OFFSET_END(inbuf);
    
    // Обновляем метаданные для 6 тайлов
    process_and_update_metadata(batcher, inbuf, output_buf, output_surface);

    // Логируем только при первых вызовах
    static int metadata_log_count = 0;
    if (metadata_log_count < 3) {
        GST_INFO_OBJECT(batcher, "✅ Batch created: %d tiles ready for nvinfer (frame #%lu)",
                       output_surface->batchSize, batcher->frame_counter);
        metadata_log_count++;
    }
    
    // Отправляем буфер downstream
    flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(btrans), output_buf);
    
    // Освобождаем входной буфер
    gst_buffer_unref(inbuf);
    
    // Обновляем счётчики
    batcher->frame_counter++;
    batcher->batch_counter++;
    
    if (!batcher->silent && batcher->frame_counter % 300 == 0) {
        GST_INFO_OBJECT(batcher, "Processed %lu frames, %lu batches", 
                       batcher->frame_counter, batcher->batch_counter);
    }
    
    batcher->last_flow_ret = flow_ret;
    
    return flow_ret;
}

static GstFlowReturn
gst_nvtilebatcher_generate_output(GstBaseTransform *btrans, GstBuffer **outbuf)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(btrans);
    *outbuf = NULL;
    return batcher->last_flow_ret;
}

/* ============================================================================
 * Start/Stop
 * ============================================================================ */

static gboolean
gst_nvtilebatcher_start(GstBaseTransform *trans)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(trans);
    
    GST_INFO_OBJECT(batcher, "Starting nvtilebatcher for Jetson (GPU %u)", 
                    batcher->gpu_id);
    
    cudaError_t cuda_err = cudaSetDevice(batcher->gpu_id);
    if (cuda_err != cudaSuccess) {
        GST_ERROR_OBJECT(batcher, "Failed to set CUDA device %u: %s",
                        batcher->gpu_id, cudaGetErrorString(cuda_err));
        return FALSE;
    }
    
    cuda_err = cudaStreamCreateWithFlags(&batcher->cuda_stream, cudaStreamNonBlocking);
    if (cuda_err != cudaSuccess) {
        GST_ERROR_OBJECT(batcher, "Failed to create CUDA stream: %s",
                        cudaGetErrorString(cuda_err));
        return FALSE;
    }
    
    cudaEventCreateWithFlags(&batcher->frame_complete_event, cudaEventDisableTiming);

    // КРИТИЧЕСКИ ВАЖНО: Вычисляем позиции тайлов ДО копирования в CUDA!
    calculate_tile_positions(batcher);

    // Инициализируем позиции тайлов
    int positions[6][2];
    for (int i = 0; i < TILES_PER_BATCH; i++) {
        positions[i][0] = batcher->tile_positions[i].x;
        positions[i][1] = batcher->tile_positions[i].y;
    }
    
    if (cuda_init_tile_positions(positions) != 0) {
        GST_ERROR_OBJECT(batcher, "Failed to initialize CUDA tile positions");
        cudaStreamDestroy(batcher->cuda_stream);
        return FALSE;
    }
    
    // Инициализируем EGL кеш
    g_mutex_init(&batcher->egl_cache_mutex);
    batcher->egl_cache = g_hash_table_new_full(
        g_direct_hash,
        g_direct_equal,
        NULL,
        egl_cache_entry_free
    );
    
    batcher->frame_counter = 0;
    batcher->batch_counter = 0;
    
    GST_INFO_OBJECT(batcher, "nvtilebatcher started successfully");
    
    return TRUE;
}

static gboolean
gst_nvtilebatcher_stop(GstBaseTransform *trans)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(trans);
    
    GST_INFO_OBJECT(batcher, "Stopping nvtilebatcher");
    
    if (batcher->cuda_stream) {
        cudaStreamSynchronize(batcher->cuda_stream);
    }
    
    // Очистка фиксированного пула
    g_mutex_lock(&batcher->output_pool_fixed.mutex);
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        if (batcher->output_pool_fixed.registered[i]) {
            for (int j = 0; j < TILES_PER_BATCH; j++) {
                cuGraphicsUnregisterResource(
                    batcher->output_pool_fixed.cuda_resources[i][j]);
            }
        }
        batcher->output_pool_fixed.buffers[i] = NULL;
        batcher->output_pool_fixed.surfaces[i] = NULL;
        batcher->output_pool_fixed.registered[i] = FALSE;
    }
    g_mutex_unlock(&batcher->output_pool_fixed.mutex);
    g_mutex_clear(&batcher->output_pool_fixed.mutex);
    
    if (batcher->output_pool) {
        gst_buffer_pool_set_active(batcher->output_pool, FALSE);
        gst_object_unref(batcher->output_pool);
        batcher->output_pool = NULL;
    }
    
    if (batcher->egl_cache) {
        g_hash_table_destroy(batcher->egl_cache);
        batcher->egl_cache = NULL;
    }
    g_mutex_clear(&batcher->egl_cache_mutex);
    
    if (batcher->frame_complete_event) {
        cudaEventDestroy(batcher->frame_complete_event);
        batcher->frame_complete_event = NULL;
    }
    
    if (batcher->cuda_stream) {
        cudaStreamDestroy(batcher->cuda_stream);
        batcher->cuda_stream = NULL;
    }
    
    batcher->pool_configured = FALSE;
    
    GST_INFO_OBJECT(batcher, "nvtilebatcher stopped successfully");
    
    return TRUE;
}

/* ============================================================================
 * Other Functions
 * ============================================================================ */

static GstCaps*
gst_nvtilebatcher_transform_caps(GstBaseTransform *trans,
                                 GstPadDirection direction,
                                 GstCaps *caps,
                                 GstCaps *filter)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(trans);
    (void)caps;

    GstCaps *result;

    if (direction == GST_PAD_SINK) {
        result = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "RGBA",
            "width", G_TYPE_INT, TILE_WIDTH,
            "height", G_TYPE_INT, TILE_HEIGHT,
            NULL);
        gst_caps_set_features(result, 0,
            gst_caps_features_from_string("memory:NVMM"));
    } else {
        // Используем динамические размеры из properties
        // Если properties ещё не установлены (caps negotiation до set_property),
        // используем ANY для ширины/высоты
        if (batcher->panorama_width == 0 || batcher->panorama_height == 0) {
            result = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "RGBA",
                NULL);
        } else {
            result = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "RGBA",
                "width", G_TYPE_INT, batcher->panorama_width,
                "height", G_TYPE_INT, batcher->panorama_height,
                NULL);
        }
        gst_caps_set_features(result, 0,
            gst_caps_features_from_string("memory:NVMM"));
    }

    if (filter) {
        GstCaps *tmp = gst_caps_intersect_full(result, filter,
                                              GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }

    return result;
}

static void
gst_nvtilebatcher_set_property(GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *pspec)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(object);

    switch (prop_id) {
        case PROP_GPU_ID:
            batcher->gpu_id = g_value_get_uint(value);
            break;
        case PROP_SILENT:
            batcher->silent = g_value_get_boolean(value);
            break;
        case PROP_PANORAMA_WIDTH:
            batcher->panorama_width = g_value_get_uint(value);
            GST_INFO_OBJECT(batcher, "Panorama width set to %u", batcher->panorama_width);
            break;
        case PROP_PANORAMA_HEIGHT:
            batcher->panorama_height = g_value_get_uint(value);
            GST_INFO_OBJECT(batcher, "Panorama height set to %u", batcher->panorama_height);
            break;
        case PROP_TILE_OFFSET_Y:
            batcher->tile_offset_y = g_value_get_uint(value);
            GST_INFO_OBJECT(batcher, "Tile offset Y set to %u", batcher->tile_offset_y);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvtilebatcher_get_property(GObject *object, guint prop_id,
                               GValue *value, GParamSpec *pspec)
{
    GstNvTileBatcher *batcher = GST_NVTILEBATCHER(object);

    switch (prop_id) {
        case PROP_GPU_ID:
            g_value_set_uint(value, batcher->gpu_id);
            break;
        case PROP_SILENT:
            g_value_set_boolean(value, batcher->silent);
            break;
        case PROP_PANORAMA_WIDTH:
            g_value_set_uint(value, batcher->panorama_width);
            break;
        case PROP_PANORAMA_HEIGHT:
            g_value_set_uint(value, batcher->panorama_height);
            break;
        case PROP_TILE_OFFSET_Y:
            g_value_set_uint(value, batcher->tile_offset_y);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvtilebatcher_finalize(GObject *object)
{
    GST_DEBUG_OBJECT(object, "Finalizing nvtilebatcher");
    G_OBJECT_CLASS(gst_nvtilebatcher_parent_class)->finalize(object);
}

static void
gst_nvtilebatcher_class_init(GstNvTileBatcherClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    
    gobject_class->set_property = gst_nvtilebatcher_set_property;
    gobject_class->get_property = gst_nvtilebatcher_get_property;
    gobject_class->finalize = gst_nvtilebatcher_finalize;
    
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID",
            "GPU device ID", 0, 7, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_SILENT,
        g_param_spec_boolean("silent", "Silent",
            "Disable info messages", FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_WIDTH,
        g_param_spec_uint("panorama-width", "Panorama Width",
            "Input panorama width (REQUIRED!)", 0, 10000,
            0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО передавать через properties!
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_HEIGHT,
        g_param_spec_uint("panorama-height", "Panorama Height",
            "Input panorama height (REQUIRED!)", 0, 3000,
            0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО передавать через properties!
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_TILE_OFFSET_Y,
        g_param_spec_uint("tile-offset-y", "Tile Offset Y",
            "Vertical offset for tiles (calculated from field_mask.png)", 0, 3000,
            434,  // Дефолт 434 - рассчитан из field_mask.png
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    transform_class->start = gst_nvtilebatcher_start;
    transform_class->stop = gst_nvtilebatcher_stop;
    transform_class->submit_input_buffer = gst_nvtilebatcher_submit_input_buffer;
    transform_class->generate_output = gst_nvtilebatcher_generate_output;
    transform_class->transform_caps = gst_nvtilebatcher_transform_caps;
    
    gst_element_class_set_static_metadata(element_class,
        "NVIDIA Tile Batcher for Jetson",
        "Filter/Video",
        "Creates batch of 6 tiles from panorama",
        "NVIDIA");
    
    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);
    
    GST_DEBUG_CATEGORY_INIT(gst_nvtilebatcher_debug, "nvtilebatcher", 0, 
                           "NVIDIA Tile Batcher for Jetson");
}

static void
gst_nvtilebatcher_init(GstNvTileBatcher *batcher)
{
    batcher->gpu_id = 0;
    batcher->silent = FALSE;
    batcher->panorama_width = 0;   // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
    batcher->panorama_height = 0;  // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
    batcher->tile_offset_y = 434;  // Дефолт - рассчитан из field_mask.png

    // Позиции тайлов будут рассчитаны динамически в submit_input_buffer()
    // после того, как panorama_width/height будут установлены через properties
    memset(batcher->tile_positions, 0, sizeof(TilePosition) * TILES_PER_BATCH);

    batcher->output_pool = NULL;
    batcher->pool_configured = FALSE;
    batcher->cuda_stream = NULL;
    batcher->frame_complete_event = NULL;
    batcher->frame_counter = 0;
    batcher->batch_counter = 0;
    batcher->last_flow_ret = GST_FLOW_OK;
    batcher->egl_cache = NULL;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(batcher), FALSE);
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(batcher), FALSE);
}

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvtilebatcher", GST_RANK_NONE, 
                               GST_TYPE_NVTILEBATCHER);
}

#ifndef PACKAGE
#define PACKAGE "nvtilebatcher"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvtilebatcher,
    "NVIDIA Tile Batcher for Jetson",
    plugin_init,
    "1.0",
    "Proprietary",
    "NVIDIA",
    "http://nvidia.com"
)