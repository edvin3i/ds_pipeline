#ifndef __GST_NVTILEBATCHER_H__
#define __GST_NVTILEBATCHER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime_api.h>
#include <cudaEGL.h>
#include "nvbufsurface.h"
#include "nvdsmeta.h"

G_BEGIN_DECLS

#define GST_TYPE_NVTILEBATCHER (gst_nvtilebatcher_get_type())
#define GST_NVTILEBATCHER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVTILEBATCHER, GstNvTileBatcher))

typedef struct _GstNvTileBatcher GstNvTileBatcher;
typedef struct _GstNvTileBatcherClass GstNvTileBatcherClass;

/* Константы конфигурации */
#define TILES_PER_BATCH 6
#define TILE_WIDTH 1024
#define TILE_HEIGHT 1024
// УДАЛЕНО: PANORAMA_WIDTH, PANORAMA_HEIGHT - теперь передаются через properties!
#define FIXED_OUTPUT_POOL_SIZE 4
#define MAX_EGL_CACHE_SIZE 32  // Maximum EGL cache entries before LRU eviction

/* Позиции тайлов */
typedef struct {
    guint tile_id;
    gint x;
    gint y;
} TilePosition;

// ПРИМЕЧАНИЕ: Позиции тайлов теперь рассчитываются динамически на основе
// panorama_width и panorama_height в методе calculate_tile_positions()
// X позиции: 192, 1216, 2240, 3264, 4288, 5312 (отступ слева 192px)
// Y позиция: (panorama_height - TILE_HEIGHT) / 2 (вертикальное центрирование)

/* EGL кеш для входных буферов */
typedef struct {
    gpointer egl_image;
    CUgraphicsResource cuda_resource;
    CUeglFrame egl_frame;
    gboolean registered;
    guint64 last_access_frame;
} EGLResourceCacheEntry;

typedef struct {
    guint tile_id;          // Номер тайла (0-5)
    guint panorama_x;       // X координата в панораме
    guint panorama_y;       // Y координата в панораме
    guint tile_width;       // Ширина тайла (1024)
    guint tile_height;      // Высота тайла (1024)
} TileRegionInfo;

/* Тип для user_meta */

#define NVDS_USER_FRAME_META_TILE_INFO (NVDS_START_USER_META + 1)
/* Функции для copy/release user_meta */
static gpointer tile_region_info_copy(gpointer data, gpointer user_data)
{
    (void)user_data;
    TileRegionInfo *src = (TileRegionInfo *)data;
    TileRegionInfo *dst = (TileRegionInfo *)g_malloc0(sizeof(TileRegionInfo));
    if (src && dst) {
        memcpy(dst, src, sizeof(TileRegionInfo));
    }
    return dst;
}

static void tile_region_info_free(gpointer data, gpointer user_data)
{
    (void)user_data;
    // Free the allocated TileRegionInfo structure
    // This is required because the data is allocated with g_new0()/g_malloc0()
    if (data) {
        g_free(data);
    }
}

struct _GstNvTileBatcher {
    GstBaseTransform base_transform;

    /* Properties */
    guint gpu_id;
    gboolean silent;
    guint panorama_width;   // Размер панорамы - передаётся через properties!
    guint panorama_height;  // Размер панорамы - передаётся через properties!
    guint tile_offset_y;    // Вертикальный offset тайлов (рассчитан из field_mask.png)

    /* Позиции тайлов */
    TilePosition tile_positions[TILES_PER_BATCH];
    
    /* Output buffer pool */
    GstBufferPool *output_pool;
    gboolean pool_configured;
    
    /* Фиксированный пул выходных буферов */
    struct {
        GstBuffer* buffers[FIXED_OUTPUT_POOL_SIZE];
        NvBufSurface* surfaces[FIXED_OUTPUT_POOL_SIZE];
        CUgraphicsResource cuda_resources[FIXED_OUTPUT_POOL_SIZE][TILES_PER_BATCH];
        CUeglFrame egl_frames[FIXED_OUTPUT_POOL_SIZE][TILES_PER_BATCH];
        gboolean registered[FIXED_OUTPUT_POOL_SIZE];
        gint current_index;
        GMutex mutex;
    } output_pool_fixed;
    
    /* CUDA */
    cudaStream_t cuda_stream;
    cudaEvent_t frame_complete_event;
    
    /* EGL кеш для входных буферов */
    GHashTable *egl_cache;
    GMutex egl_cache_mutex;
    
    /* Счётчики и статистика */
    guint64 frame_counter;
    guint64 batch_counter;
    GstFlowReturn last_flow_ret;
};

struct _GstNvTileBatcherClass {
    GstBaseTransformClass base_transform_class;
};

GType gst_nvtilebatcher_get_type(void);

G_END_DECLS

#endif