// gstnvdsvirtualcam.h - Заголовочный файл плагина виртуальной камеры
#ifndef __GST_NVDS_VIRTUAL_CAM_H__
#define __GST_NVDS_VIRTUAL_CAM_H__

#include "gstnvdsvirtualcam_allocator.h"
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime_api.h>
#include <nvbufsurface.h>
#include "cuda_virtual_cam_kernel.h"

#include <cmath>
// Определяем M_PI если его нет
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#ifdef __aarch64__
#include <cudaEGL.h>
#include <unordered_map>
#include <memory>
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVDS_VIRTUAL_CAM (gst_nvds_virtual_cam_get_type())
G_DECLARE_FINAL_TYPE(GstNvdsVirtualCam, gst_nvds_virtual_cam, GST, NVDS_VIRTUAL_CAM, GstBaseTransform)

// Размер фиксированного пула буферов
#define FIXED_OUTPUT_POOL_SIZE 8

// Основная структура плагина
struct _GstNvdsVirtualCam {
   GstBaseTransform element;
   
   // ========== ПАРАМЕТРЫ КАМЕРЫ ==========
   gfloat yaw;           // Поворот влево-вправо (-110 до +110)
   gfloat pitch;         // Наклон вверх-вниз (-30 до +30)
   gfloat roll;          // Наклон вбок (-15 до +15)
   gfloat fov;           // Угол обзора (30-90)
   GMutex properties_mutex;  // Защита от race condition при изменении параметров
   
   // ========== РЕЖИМЫ РАБОТЫ ==========
   gboolean auto_follow;     // Следить за объектом из метаданных
   gfloat smooth_factor;     // Плавность переходов (0.0-1.0)
   
   // ========== КОНФИГУРАЦИЯ ==========
   guint gpu_id;
   guint output_width;       // Размер выходного изображения (1920)
   guint output_height;      // Размер выходного изображения (1080)
   guint input_width;        // Размер входной панорамы (6528)
   guint input_height;       // Размер входной панорамы (1800)
   
   // ========== CUDA РЕСУРСЫ ==========
   cudaStream_t cuda_stream;
   
   // Предвычисленные лучи камеры (зависят только от FOV)
   gfloat *rays_gpu;
   gboolean rays_computed;
   gfloat last_fov;          // Для проверки изменения FOV
   
   // LUT карты для remap (зависят от углов)
   gfloat *remap_u_gpu;
   gfloat *remap_v_gpu;
   
   // Конфигурация для kernels
   VirtualCamConfig kernel_config;
   
   // ========== КЭШИРОВАНИЕ ==========
   struct {
       gfloat last_yaw;
       gfloat last_pitch;
       gfloat last_roll;
       gboolean valid;
       GMutex mutex;  // Защита от race condition при многопоточности
   } lut_cache;
   
   // ========== УПРАВЛЕНИЕ БУФЕРАМИ ==========
   GstBufferPool *output_pool;
   gboolean pool_configured;
   
   // Фиксированный пул выходных буферов для оптимизации
   // ИЗМЕНЕНО: используем GstNvdsStitchMemory вместо GstNvDsPreProcessMemory
   struct {
       GstBuffer* buffers[FIXED_OUTPUT_POOL_SIZE];
       GstNvdsVirtualCamMemory* memories[FIXED_OUTPUT_POOL_SIZE];  // <-- ИЗМЕНЕНО ТИП
       gint current_index;
       GMutex mutex;
       gboolean initialized;
   } output_pool_fixed;
   
   // ========== СОСТОЯНИЕ ==========
   guint64 frame_count;      // Счетчик обработанных кадров
   GstFlowReturn last_flow_ret;  // Результат последнего push
   
   // ========== TRACKING (для auto_follow) ==========
   gfloat target_yaw;        // Целевые углы для плавного перехода
   gfloat target_pitch;
   gfloat target_fov;        // целевой FOV для плавного изменения
   gboolean tracking_active;
   guint tracked_object_id;  // ID отслеживаемого объекта

   // Параметры автозума
   gfloat s_target;          // <-- ДОБАВИТЬ: целевой размер объекта (0.035 = 3.5%)
   gfloat ball_angular_size; // <-- ДОБАВИТЬ: угловой размер мяча в радианах
   
   // ========== МЕТАДАННЫЕ ==========
   gboolean add_virtual_cam_meta;  // Добавлять ли метаданные с параметрами камеры
   
   // ========== PERFORMANCE ==========
   guint64 last_perf_log_frame;
   guint64 total_processing_time;    // ИЗМЕНЕНО на guint64
   guint64 max_processing_time;      // ИЗМЕНЕНО на guint64
   guint64 min_processing_time;      // ИЗМЕНЕНО на guint64

#ifdef __aarch64__
   // ========== КЕШИРОВАНИЕ EGL РЕСУРСОВ ==========
   // Структура для хранения зарегистрированных EGL буферов
   struct EGLCacheEntry {
       CUgraphicsResource cuda_resource;  // Зарегистрированный ресурс
       CUeglFrame egl_frame;              // Mapped EGL frame
       void* cuda_ptr;                    // CUDA указатель для использования
       guint64 last_used_frame;          // Номер кадра последнего использования
       gboolean is_registered;            // Флаг успешной регистрации
   };
   
   // Кеш: EGLImage address → CUDA resources
   // Используем указатель на EGLImage как ключ
   std::unordered_map<void*, std::unique_ptr<EGLCacheEntry>> egl_input_cache;
   
   // Максимальный размер кеша (обычно = размер buffer pool)
   static constexpr size_t MAX_EGL_CACHE_SIZE = 10;
   
   // Мьютекс для потокобезопасности кеша
   GMutex egl_cache_mutex;
   
   // Статистика кеша для отладки
   guint64 egl_cache_hits;
   guint64 egl_cache_misses;

   // ========== ПАРАМЕТРЫ МЯЧА ==========
   gfloat ball_x;                // Позиция мяча X на панораме (в пикселях)
   gfloat ball_y;                // Позиция мяча Y на панораме (в пикселях)
   gfloat ball_actual_radius;    // Реальный размер мяча в пикселях
   gfloat target_ball_size;      // Желаемый размер мяча на экране (0.01-0.1)

   // Для умной обработки границ
   gfloat safe_fov_limit;        // Текущий максимальный безопасный FOV
   gboolean ball_near_edge;      // Флаг близости к краю
#endif
};



// Структура для метаданных виртуальной камеры (опционально)
typedef struct {
   GstMeta meta;
   
   // Параметры виртуальной камеры
   gfloat yaw;
   gfloat pitch;
   gfloat roll;
   gfloat fov;
   
   // Информация о преобразовании
   guint input_width;
   guint input_height;
   guint output_width;
   guint output_height;
   
   // Флаги
   gboolean tracking_active;
   guint tracked_object_id;
   
   // Временные метки
   guint64 processing_time;  // ИЗМЕНЕНО на guint64
} GstNvdsVirtualCamMeta;

// API для метаданных
GType gst_nvds_virtual_cam_meta_api_get_type(void);
const GstMetaInfo *gst_nvds_virtual_cam_meta_get_info(void);

#define GST_NVDS_VIRTUAL_CAM_META_API_TYPE (gst_nvds_virtual_cam_meta_api_get_type())
#define GST_NVDS_VIRTUAL_CAM_META_INFO (gst_nvds_virtual_cam_meta_get_info())

// Макросы для работы с метаданными
#define gst_buffer_get_nvds_virtual_cam_meta(b) \
   ((GstNvdsVirtualCamMeta*)gst_buffer_get_meta((b), GST_NVDS_VIRTUAL_CAM_META_API_TYPE))

#define gst_buffer_add_nvds_virtual_cam_meta(b) \
   ((GstNvdsVirtualCamMeta*)gst_buffer_add_meta((b), GST_NVDS_VIRTUAL_CAM_META_INFO, NULL))

G_END_DECLS

#endif /* __GST_NVDS_VIRTUAL_CAM_H__ */
