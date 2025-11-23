// gstnvdsvirtualcam.cpp - Реализация плагина виртуальной камеры для Jetson
#include "gstnvdsvirtualcam.h"
#include "nvdsvirtualcam_config.h"
#include "cuda_virtual_cam_kernel.h"
#include "gstnvdsvirtualcam_allocator.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <cuda_runtime_api.h>
#include <cudaEGL.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <gstnvdsmeta.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <unordered_map>


#include "gstnvdsmeta.h"

// Use config namespace for cleaner code
using namespace NvdsVirtualCamConfig;

// Определение типа CUeglImage если он не определен
#ifndef CUeglImage
typedef void* CUeglImage;
#endif

#ifndef PACKAGE
#define PACKAGE "nvdsvirtualcam"
#endif

#ifndef GST_CAPS_FEATURE_MEMORY_NVMM
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvds_virtual_cam_debug);
#define GST_CAT_DEFAULT gst_nvds_virtual_cam_debug

// Макросы для логирования
#define LOG_ERROR(obj, fmt, ...) GST_ERROR_OBJECT(obj, fmt, ##__VA_ARGS__)
#define LOG_WARNING(obj, fmt, ...) GST_WARNING_OBJECT(obj, fmt, ##__VA_ARGS__)
#define LOG_INFO(obj, fmt, ...) GST_INFO_OBJECT(obj, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(obj, fmt, ...) GST_DEBUG_OBJECT(obj, fmt, ##__VA_ARGS__)


// Forward declarations для функций обработки метаданных
static void pano_xy_to_yaw_pitch(gfloat x, gfloat y, gint pano_w, gint pano_h,
                                 gfloat* yaw, gfloat* pitch);

static void smooth_camera_tracking(GstNvdsVirtualCam *vcam);

static void
update_camera_from_ball(GstNvdsVirtualCam *vcam);


static void 
gst_nvds_virtual_cam_get_property(GObject *object, guint prop_id,
                                  GValue *value, GParamSpec *pspec);

// Pad templates - используем хардкод размеров так как G_STRINGIFY не работает с constexpr
static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw(memory:NVMM), "
                                          "format={ RGBA, NV12 }, "
                                          "width=(int)[1,10000], "     // Динамическая ширина
                                          "height=(int)[1,3000], "     // Динамическая высота
                                          "framerate=(fraction)[0/1,MAX]"));

// Src template - выдаем ТОЛЬКО 1920x1080
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw(memory:NVMM), "
                                          "format=RGBA, "
                                          "width=(int)1920, "       // Фиксированный размер
                                          "height=(int)1080, "      // Фиксированный размер
                                          "framerate=(fraction)[0/1,MAX]"));

G_DEFINE_TYPE(GstNvdsVirtualCam, gst_nvds_virtual_cam, GST_TYPE_BASE_TRANSFORM);

// Properties
enum {
    PROP_0,
    PROP_YAW,
    PROP_PITCH,
    PROP_ROLL,
    PROP_FOV,
    PROP_GPU_ID,
    PROP_OUTPUT_WIDTH,
    PROP_OUTPUT_HEIGHT,
    PROP_PANORAMA_WIDTH,
    PROP_PANORAMA_HEIGHT,
    PROP_AUTO_FOLLOW,
    PROP_SMOOTH_FACTOR,
    PROP_S_TARGET,
    PROP_BALL_X,
    PROP_BALL_Y,
    PROP_BALL_ACTUAL_RADIUS,
    PROP_TARGET_BALL_SIZE
};  






/* ============================================================================
// Функция преобразования координат панорамы в углы камеры
 * ============================================================================ */
static void
pano_xy_to_yaw_pitch(gfloat x, gfloat y, gint pano_w, gint pano_h,
                     gfloat *out_yaw, gfloat *out_pitch)
{
    // ============================================================================
    // ПРЕОБРАЗОВАНИЕ ПИКСЕЛЬНЫХ КООРДИНАТ В СФЕРИЧЕСКИЕ УГЛЫ
    // ============================================================================
    // Эта функция конвертирует координаты мяча на панораме (x, y в пикселях)
    // в сферические углы камеры (yaw, pitch в градусах).
    //
    // Диапазоны углов панорамы:
    const gfloat LON_MIN = NvdsVirtualCamConfig::LON_MIN;  // -90° (левый край)
    const gfloat LON_MAX = NvdsVirtualCamConfig::LON_MAX;  // +90° (правый край)
    const gfloat LAT_MIN = NvdsVirtualCamConfig::LAT_MIN;  // -27° (НИЗ панорамы)
    const gfloat LAT_MAX = NvdsVirtualCamConfig::LAT_MAX;  // +27° (ВЕРХ панорамы)

    // X -> yaw (горизонтальный угол поворота камеры влево-вправо)
    // x=0 → -90° (крайний левый), x=pano_w-1 → +90° (крайний правый)
    gfloat norm_x = x / (pano_w - 1);
    *out_yaw = LON_MIN + norm_x * (LON_MAX - LON_MIN);

    // Y -> pitch (вертикальный угол наклона камеры вверх-вниз)
    // ВАЖНО: формула была ИСПРАВЛЕНА (была инвертирована!)
    // y=0 (верх изображения) → LAT_MAX (+27°), y=pano_h-1 (низ изображения) → LAT_MIN (-27°)
    // Старая НЕПРАВИЛЬНАЯ формула: *out_pitch = LAT_MIN - norm_y * (LAT_MIN - LAT_MAX)
    gfloat norm_y = y / (pano_h - 1);
    *out_pitch = LAT_MAX - norm_y * (LAT_MAX - LAT_MIN);

    // ОТЛАДКА: выводим преобразование координат
    // static int xy_log_counter = 0;
    // if (xy_log_counter++ % 30 == 0) {
    //     g_print("🔍 XY→ANGLE: ball_y=%.0f → pitch=%.1f° (LAT_MIN=%.1f°, LAT_MAX=%.1f°)\n",
    //             y, *out_pitch, LAT_MIN, LAT_MAX);
    // }
}

// Функция плавного слежения камеры
// ============================================================================
// ФУНКЦИЯ: smooth_camera_tracking
// ============================================================================
// Применяет плавное сглаживание к движению камеры.
// Это делает переходы камеры более плавными, без резких рывков.
//
// ЛОГИКА:
// 1. Вычисляет разницу между текущей и целевой позицией (target - current)
// 2. Если разница больше мертвой зоны → применяет частичное изменение
// 3. Если разница меньше мертвой зоны → не двигается (избегает микродвижений)
//
// ПАРАМЕТРЫ:
// - smooth_factor (обычно 0.3): как быстро камера догоняет цель
//   - 0.3 = камера движется на 30% от расстояния до цели каждый кадр
//   - 1.0 = мгновенное движение (без сглаживания)
//   - 0.1 = очень медленное плавное движение
// - DEAD_ZONE = 0.1°: порог для yaw/pitch (игнорируем изменения < 0.1°)
// - FOV_DEAD_ZONE = 0.5°: порог для FOV (игнорируем изменения < 0.5°)
//
// ВАЖНО: Границы применены до этой функции в update_virtual_camera(),
// поэтому здесь мы просто плавно двигаемся к уже ограниченным target значениям.
static void smooth_camera_tracking(GstNvdsVirtualCam *vcam)
{
    // ============================================================================
    // МЕРТВЫЕ ЗОНЫ (Dead Zones)
    // ============================================================================
    // Мертвая зона предотвращает микродвижения камеры (дрожание/шум).
    // Если разница между текущим и целевым значением меньше порога,
    // камера остается неподвижной.
    const gfloat DEAD_ZONE = 0.1f;      // Градусы для yaw/pitch
    const gfloat FOV_DEAD_ZONE = 0.5f;  // Градусы для FOV

    // ============================================================================
    // ВЫЧИСЛЕНИЕ РАЗНИЦЫ
    // ============================================================================
    // Разница = куда нужно двигаться (target - current)
    // Положительная разница → движение в положительном направлении
    // Отрицательная разница → движение в отрицательном направлении
    gfloat yaw_diff = vcam->target_yaw - vcam->yaw;
    gfloat pitch_diff = vcam->target_pitch - vcam->pitch;
    gfloat fov_diff = vcam->target_fov - vcam->fov;

    // ОТЛАДКА: вывод текущего состояния камеры (каждые 30 кадров)
    // static int log_counter = 0;
    // if (log_counter++ % 30 == 0) {
    //     g_print("📊 CAMERA: pitch=%.1f°→%.1f° | FOV=%.1f°→%.1f° | yaw=%.1f°\n",
    //             vcam->pitch, vcam->target_pitch,
    //             vcam->fov, vcam->target_fov,
    //             vcam->yaw);
    // }

    // ============================================================================
    // ПРИМЕНЕНИЕ СГЛАЖИВАНИЯ
    // ============================================================================
    // Формула: new_value = current_value + (target - current) * smooth_factor
    // Это называется "exponential smoothing" или "lerp" (linear interpolation)
    //
    // Пример с smooth_factor = 0.3:
    // - Кадр 1: current=0°, target=10° → diff=10°, new = 0 + 10*0.3 = 3°
    // - Кадр 2: current=3°, target=10° → diff=7°, new = 3 + 7*0.3 = 5.1°
    // - Кадр 3: current=5.1°, target=10° → diff=4.9°, new = 5.1 + 4.9*0.3 = 6.57°
    // - И так далее, плавно приближаясь к 10°

    // Горизонтальное движение (yaw):
    if (fabs(yaw_diff) > DEAD_ZONE) {
        vcam->yaw += yaw_diff * vcam->smooth_factor;
    }

    // Вертикальное движение (pitch):
    if (fabs(pitch_diff) > DEAD_ZONE) {
        vcam->pitch += pitch_diff * vcam->smooth_factor;
    }

    // Изменение зума (FOV):
    if (fabs(fov_diff) > FOV_DEAD_ZONE) {
        vcam->fov += fov_diff * vcam->smooth_factor;
    }

    // ============================================================================
    // ФИНАЛЬНОЕ ОГРАНИЧЕНИЕ FOV
    // ============================================================================
    // ВАЖНО: Всегда ограничиваем FOV в диапазоне [40°, 68°], даже если
    // сглаживание не сработало (diff < dead_zone).
    // Это защищает от "залипших" значений из предыдущих запусков или багов.
    gfloat fov_before_clamp = vcam->fov;
    vcam->fov = CLAMP(vcam->fov,
                     NvdsVirtualCamConfig::FOV_MIN,  // 40°
                     NvdsVirtualCamConfig::FOV_MAX); // 68°

    // Отладка: если FOV был обрезан, выводим предупреждение
    if (fov_before_clamp != vcam->fov) {
        LOG_WARNING(vcam, "FOV clamped: %.1f° → %.1f° (limits: %.1f-%.1f°)",
                    fov_before_clamp, vcam->fov,
                    NvdsVirtualCamConfig::FOV_MIN,
                    NvdsVirtualCamConfig::FOV_MAX);
    }
    
    
    // Roll обновляем только если yaw действительно изменился
    static gfloat last_yaw_for_roll = 0.0f;
    if (fabs(vcam->yaw - last_yaw_for_roll) > 0.01f) {
        const gfloat ROLL_MAX = NvdsVirtualCamConfig::ROLL_MAX;
        const gfloat YAW_MAX = NvdsVirtualCamConfig::YAW_MAX;  // 90°
        gfloat normalized_pos = vcam->yaw / 110;
        vcam->roll = normalized_pos * ROLL_MAX;
        last_yaw_for_roll = vcam->yaw;
    }

    // Ограничения применяются ТОЛЬКО в конфиге:
    // - FOV: ограничен FOV_MIN/FOV_MAX (40-68°) - применяется выше
    // - Pitch: ограничен PITCH_MIN/PITCH_MAX (-32 до +22°) - через GStreamer properties
    // - Yaw: ограничен YAW_MIN/YAW_MAX (-90 до +90°) - через GStreamer properties
}
/* ============================================================================
 * Вспомогательные функции
 * ============================================================================ */

static gboolean
allocate_cuda_resources(GstNvdsVirtualCam *vcam)
{
    cudaError_t cuda_err;
    
    LOG_INFO(vcam, "Allocating CUDA resources on GPU %d", vcam->gpu_id);
    
    cuda_err = cudaSetDevice(vcam->gpu_id);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to set CUDA device %d: %s", 
                  vcam->gpu_id, cudaGetErrorString(cuda_err));
        return FALSE;
    }
    
    cuda_err = cudaStreamCreateWithFlags(&vcam->cuda_stream, cudaStreamNonBlocking);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to create CUDA stream: %s", 
                  cudaGetErrorString(cuda_err));
        return FALSE;
    }
    
    // Выделяем память для предвычисленных лучей камеры
    size_t rays_size = vcam->output_width * vcam->output_height * 3 * sizeof(float);
    cuda_err = cudaMalloc(&vcam->rays_gpu, rays_size);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to allocate rays memory: %s", 
                  cudaGetErrorString(cuda_err));
        cudaStreamDestroy(vcam->cuda_stream);
        vcam->cuda_stream = NULL;
        return FALSE;
    }
    
    // Выделяем память для LUT карт
    size_t lut_size = vcam->output_width * vcam->output_height * sizeof(float);
    cuda_err = cudaMalloc(&vcam->remap_u_gpu, lut_size);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to allocate remap_u memory: %s", 
                  cudaGetErrorString(cuda_err));
        cudaFree(vcam->rays_gpu);
        cudaStreamDestroy(vcam->cuda_stream);
        return FALSE;
    }
    
    cuda_err = cudaMalloc(&vcam->remap_v_gpu, lut_size);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to allocate remap_v memory: %s", 
                  cudaGetErrorString(cuda_err));
        cudaFree(vcam->rays_gpu);
        cudaFree(vcam->remap_u_gpu);
        cudaStreamDestroy(vcam->cuda_stream);
        return FALSE;
    }
    
    LOG_INFO(vcam, "CUDA resources allocated successfully");
    return TRUE;
}

static void
free_cuda_resources(GstNvdsVirtualCam *vcam)
{
    LOG_DEBUG(vcam, "Freeing CUDA resources");
    
    if (vcam->cuda_stream) {
        cudaStreamSynchronize(vcam->cuda_stream);
        cudaStreamDestroy(vcam->cuda_stream);
        vcam->cuda_stream = NULL;
    }
    
    if (vcam->rays_gpu) {
        cudaFree(vcam->rays_gpu);
        vcam->rays_gpu = NULL;
    }
    
    if (vcam->remap_u_gpu) {
        cudaFree(vcam->remap_u_gpu);
        vcam->remap_u_gpu = NULL;
    }
    
    if (vcam->remap_v_gpu) {
        cudaFree(vcam->remap_v_gpu);
        vcam->remap_v_gpu = NULL;
    }
    
    LOG_INFO(vcam, "CUDA resources freed");
}

static gboolean
update_lut_if_needed(GstNvdsVirtualCam *vcam)
{
    // Получаем snapshot параметров камеры (уже защищено mutex в caller)
    g_mutex_lock(&vcam->properties_mutex);
    gfloat current_yaw = vcam->yaw;
    gfloat current_pitch = vcam->pitch;
    gfloat current_roll = vcam->roll;
    gfloat current_fov = vcam->fov;
    g_mutex_unlock(&vcam->properties_mutex);

    // Защита от race condition при многопоточном доступе к кешу
    g_mutex_lock(&vcam->lut_cache.mutex);

    // Проверяем, нужно ли обновить LUT (используем snapshot значения)
    if (vcam->lut_cache.valid &&
        std::fabs(vcam->lut_cache.last_yaw - current_yaw) < NvdsVirtualCamConfig::ANGLE_CHANGE_THRESHOLD &&
        std::fabs(vcam->lut_cache.last_pitch - current_pitch) < NvdsVirtualCamConfig::ANGLE_CHANGE_THRESHOLD &&
        std::fabs(vcam->lut_cache.last_roll - current_roll) < NvdsVirtualCamConfig::ANGLE_CHANGE_THRESHOLD) {
        g_mutex_unlock(&vcam->lut_cache.mutex);
        return TRUE;
    }

    // Предвычисляем лучи камеры если FOV изменился
    if (!vcam->rays_computed || std::fabs(vcam->last_fov - current_fov) > 0.1f) {
        cudaError_t err = precompute_camera_rays(
            vcam->rays_gpu,
            vcam->output_width, vcam->output_height,
            current_fov,
            vcam->cuda_stream
        );

        if (err != cudaSuccess) {
            LOG_ERROR(vcam, "Failed to compute camera rays: %s",
                      cudaGetErrorString(err));
            g_mutex_unlock(&vcam->lut_cache.mutex);
            return FALSE;
        }

        vcam->rays_computed = TRUE;
        vcam->last_fov = current_fov;
        LOG_DEBUG(vcam, "Camera rays updated for FOV %.1f", current_fov);
    }

    // Генерируем новые LUT карты (используем snapshot значения)
    cudaError_t err = generate_remap_lut(
        vcam->rays_gpu,
        vcam->remap_u_gpu,
        vcam->remap_v_gpu,
        current_yaw,
        current_pitch,
        current_roll,
        &vcam->kernel_config,
        vcam->cuda_stream
    );

    if (err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to generate LUT: %s", cudaGetErrorString(err));
        g_mutex_unlock(&vcam->lut_cache.mutex);
        return FALSE;
    }

    // Обновляем кеш (сохраняем snapshot значения)
    vcam->lut_cache.last_yaw = current_yaw;
    vcam->lut_cache.last_pitch = current_pitch;
    vcam->lut_cache.last_roll = current_roll;
    vcam->lut_cache.valid = TRUE;

    LOG_DEBUG(vcam, "LUT updated for yaw=%.1f, pitch=%.1f, roll=%.1f",
              current_yaw, current_pitch, current_roll);

    g_mutex_unlock(&vcam->lut_cache.mutex);
    return TRUE;
}

/* ============================================================================
 * EGL Cache Management
 * ============================================================================ */

// Структура для кеша EGL->CUDA маппингов
struct EGLCacheEntry {
    CUgraphicsResource cuda_resource;
    CUeglFrame egl_frame;
    void* cuda_ptr;
    bool is_registered;
};

// Глобальный кеш для входных буферов (так как их мало - обычно 4-6)
static std::unordered_map<void*, EGLCacheEntry> g_egl_cache;
static GMutex g_egl_cache_mutex;
static bool g_egl_cache_initialized = false;

static void init_egl_cache() {
    if (!g_egl_cache_initialized) {
        g_mutex_init(&g_egl_cache_mutex);
        g_egl_cache_initialized = true;
        GST_INFO("EGL cache initialized");
    }
}

static void cleanup_egl_cache() {
    if (!g_egl_cache_initialized) return;
    
    g_mutex_lock(&g_egl_cache_mutex);
    
    for (auto& pair : g_egl_cache) {
        if (pair.second.is_registered) {
            cuGraphicsUnregisterResource(pair.second.cuda_resource);
        }
    }
    g_egl_cache.clear();
    
    g_mutex_unlock(&g_egl_cache_mutex);
    g_mutex_clear(&g_egl_cache_mutex);
    g_egl_cache_initialized = false;
    
    GST_INFO("EGL cache cleaned up");
}

static void* 
get_cached_cuda_pointer(void* egl_image)
{
    if (!egl_image) {
        GST_ERROR("NULL EGL image");
        return nullptr;
    }
    
    init_egl_cache();
    
    g_mutex_lock(&g_egl_cache_mutex);
    
    // Ищем в кеше
    auto it = g_egl_cache.find(egl_image);
    if (it != g_egl_cache.end() && it->second.is_registered) {
        // Нашли в кеше - возвращаем сразу
        void* ptr = it->second.cuda_ptr;
        g_mutex_unlock(&g_egl_cache_mutex);
        GST_LOG("EGL cache HIT for %p -> %p", egl_image, ptr);
        return ptr;
    }
    
    // Не нашли - регистрируем
    GST_DEBUG("EGL cache MISS for %p, registering", egl_image);
    
    EGLCacheEntry entry;
    entry.is_registered = false;
    
    // Регистрируем EGL image в CUDA
    CUresult cu_result = cuGraphicsEGLRegisterImage(
        &entry.cuda_resource,
        (CUeglImage)egl_image,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE
    );
    
    if (cu_result != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(cu_result, &error_str);
        GST_ERROR("Failed to register EGL image: %s", error_str);
        g_mutex_unlock(&g_egl_cache_mutex);
        return nullptr;
    }
    
    // Получаем mapped frame
    cu_result = cuGraphicsResourceGetMappedEglFrame(
        &entry.egl_frame,
        entry.cuda_resource,
        0, 0
    );
    
    if (cu_result != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(cu_result, &error_str);
        GST_ERROR("Failed to get mapped EGL frame: %s", error_str);
        cuGraphicsUnregisterResource(entry.cuda_resource);
        g_mutex_unlock(&g_egl_cache_mutex);
        return nullptr;
    }
    
    entry.cuda_ptr = (void*)entry.egl_frame.frame.pPitch[0];
    entry.is_registered = true;
    
    // Сохраняем в кеш
    g_egl_cache[egl_image] = entry;
    
    GST_INFO("Registered EGL %p -> CUDA %p (cache size: %zu)", 
             egl_image, entry.cuda_ptr, g_egl_cache.size());
    
    void* result = entry.cuda_ptr;
    g_mutex_unlock(&g_egl_cache_mutex);
    
    return result;
}

/* ============================================================================
 * Инициализация фиксированного пула буферов
 * ============================================================================ */

static gboolean 
setup_fixed_output_pool(GstNvdsVirtualCam *vcam)
{
    LOG_INFO(vcam, "Setting up fixed output pool with %d buffers", FIXED_OUTPUT_POOL_SIZE);
    
    g_mutex_init(&vcam->output_pool_fixed.mutex);
    vcam->output_pool_fixed.current_index = 0;
    
    // Предварительно выделяем буферы
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        GstFlowReturn flow_ret = gst_buffer_pool_acquire_buffer(
            vcam->output_pool, 
            &vcam->output_pool_fixed.buffers[i], 
            NULL);
        
        if (flow_ret != GST_FLOW_OK) {
            LOG_ERROR(vcam, "Failed to acquire fixed buffer %d", i);
            // Освобождаем уже выделенные буферы
            for (int j = 0; j < i; j++) {
                gst_buffer_unref(vcam->output_pool_fixed.buffers[j]);
                vcam->output_pool_fixed.buffers[j] = NULL;
            }
            return FALSE;
        }
        
        // Получаем память для каждого буфера
        vcam->output_pool_fixed.memories[i] = 
            gst_nvdsvirtualcam_buffer_get_memory(vcam->output_pool_fixed.buffers[i]);
            
        if (!vcam->output_pool_fixed.memories[i]) {
            LOG_ERROR(vcam, "Failed to get memory for fixed buffer %d", i);
            for (int j = 0; j <= i; j++) {
                if (vcam->output_pool_fixed.buffers[j]) {
                    gst_buffer_unref(vcam->output_pool_fixed.buffers[j]);
                    vcam->output_pool_fixed.buffers[j] = NULL;
                }
            }
            return FALSE;
        }
        
        // Регистрируем EGL/CUDA ресурсы для выходных буферов
        if (vcam->output_pool_fixed.memories[i]->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
            if (!vcam->output_pool_fixed.memories[i]->egl_mapped) {
                gst_nvdsvirtualcam_memory_map_egl(vcam->output_pool_fixed.memories[i]);
            }
            if (!vcam->output_pool_fixed.memories[i]->cuda_registered) {
                gst_nvdsvirtualcam_memory_register_cuda(vcam->output_pool_fixed.memories[i]);
            }
        }
        
        LOG_DEBUG(vcam, "Fixed buffer %d allocated successfully", i);
    }
    
    vcam->output_pool_fixed.initialized = TRUE;
    LOG_INFO(vcam, "Fixed output pool ready with %d buffers", FIXED_OUTPUT_POOL_SIZE);
    
    return TRUE;
}

/* ============================================================================
 * Buffer processing
 * ============================================================================ */

static GstFlowReturn
gst_nvds_virtual_cam_submit_input_buffer(GstBaseTransform *btrans,
                                         gboolean discont, GstBuffer *inbuf)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(btrans);
    GstBuffer *outbuf = NULL;
    GstNvdsVirtualCamMemory *out_memory = NULL;
    NvBufSurface *in_surface = NULL;
    NvBufSurface *out_surface = NULL;
    GstMapInfo in_map = GST_MAP_INFO_INIT;
    GstFlowReturn flow_ret = GST_FLOW_OK;
    cudaError_t cuda_err;
    
    (void)discont;

    
    // Проверки
    if (!inbuf) {
        LOG_ERROR(vcam, "Input buffer is NULL");
        return GST_FLOW_ERROR;
    }
    
    if (!vcam->output_pool || !vcam->output_pool_fixed.initialized) {
        LOG_ERROR(vcam, "Output pool is not initialized");
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // ============================================================================
    // МЕРТВЫЙ КОД - НЕ ИСПОЛЬЗУЕТСЯ
    // Эти переменные объявлены но нигде не применяются (unused variables warning)
    // Roll теперь рассчитывается автоматически в smooth_camera_tracking() (строка 254)
    // и при ручной установке yaw через set_property() (строка 1239)
    // Оставлено закомментированным для истории, если что-то сломается - можно вернуться
    // ============================================================================
    // const gfloat ROLL_MAX = NvdsVirtualCamConfig::ROLL_MAX;
    // const gfloat YAW_MAX = NvdsVirtualCamConfig::YAW_MAX;
    // gfloat normalized_pos = vcam->yaw / YAW_MAX;

    // Создаём snapshot параметров камеры для потокобезопасности
    // Это защищает от race condition если параметры меняются через set_property
    g_mutex_lock(&vcam->properties_mutex);
    gfloat current_yaw = vcam->yaw;
    gfloat current_pitch = vcam->pitch;
    gfloat current_roll = vcam->roll;
    gfloat current_fov = vcam->fov;
    g_mutex_unlock(&vcam->properties_mutex);

    // if (vcam->auto_follow && vcam->tracking_active) {
    update_camera_from_ball(vcam);
    // }
    // if (!vcam->auto_follow) {

    // ОТЛАДКА: проверим target_pitch до и после save/restore
    // static int restore_log_counter = 0;
    // if (restore_log_counter++ % 30 == 0) {
    //     g_print("🔄 BEFORE save/restore: target_pitch=%.1f°, current_pitch=%.1f°\n",
    //             vcam->target_pitch, current_pitch);
    // }

    // УДАЛЕНО: бесполезный код save/restore который ничего не делал
    // Он был нужен только если apply_edge_safe_limits вызывался между сохранением и восстановлением
    // Но apply_edge_safe_limits удалён - все лимиты теперь только в конфиге!

    // if (restore_log_counter % 30 == 1) {
    //     g_print("🔄 AFTER update_camera_from_ball: target_pitch=%.1f°\n", vcam->target_pitch);
    // }

    // Обновляем LUT если параметры изменились (используем snapshot значения)
    if (!update_lut_if_needed(vcam)) {
        LOG_ERROR(vcam, "Failed to update LUT");
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Получаем буфер из фиксированного пула
    g_mutex_lock(&vcam->output_pool_fixed.mutex);
    gint buf_idx = vcam->output_pool_fixed.current_index;
    GstBuffer *pool_buf = vcam->output_pool_fixed.buffers[buf_idx];
    out_memory = vcam->output_pool_fixed.memories[buf_idx];
    vcam->output_pool_fixed.current_index = (buf_idx + 1) % FIXED_OUTPUT_POOL_SIZE;
    g_mutex_unlock(&vcam->output_pool_fixed.mutex);
    
    // Создаем новый буфер с ref на память из пула
    outbuf = gst_buffer_new();
    GstMemory *mem = gst_buffer_peek_memory(pool_buf, 0);
    gst_buffer_append_memory(outbuf, gst_memory_ref(mem));
    
    out_surface = out_memory->surf;
    
    if (!out_surface || !out_surface->surfaceList) {
        LOG_ERROR(vcam, "Output surface invalid");
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Устанавливаем numFilled
    out_surface->numFilled = 1;
    out_surface->batchSize = 1;
    
    // Маппинг входного буфера для получения NvBufSurface
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        LOG_ERROR(vcam, "Failed to map input buffer");
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    in_surface = (NvBufSurface *)in_map.data;
    
    if (!in_surface || !in_surface->surfaceList || in_surface->numFilled == 0) {
        LOG_ERROR(vcam, "Invalid input surface");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Detect and validate input format
    NvBufSurfaceColorFormat input_format = in_surface->surfaceList[0].colorFormat;
    if (input_format != NVBUF_COLOR_FORMAT_RGBA &&
        input_format != NVBUF_COLOR_FORMAT_NV12) {
        LOG_ERROR(vcam, "Unsupported input format: %d (expected RGBA=%d or NV12=%d)",
                  input_format, NVBUF_COLOR_FORMAT_RGBA, NVBUF_COLOR_FORMAT_NV12);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    LOG_DEBUG(vcam, "Input format: %s",
              input_format == NVBUF_COLOR_FORMAT_NV12 ? "NV12" : "RGBA");

    // Получение CUDA указателей
    unsigned char *input_ptr = nullptr;
    unsigned char *output_ptr = nullptr;
    
    // ВХОДНОЙ БУФЕР - обработка SURFACE_ARRAY через EGL с кешированием
    if (in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
        // Делаем EGL mapping если его еще нет
        if (in_surface->surfaceList[0].mappedAddr.eglImage == nullptr) {
            LOG_DEBUG(vcam, "Performing EGL mapping for input surface");
            int egl_result = NvBufSurfaceMapEglImage(in_surface, 0);
            if (egl_result != 0) {
                LOG_ERROR(vcam, "Failed to map EGL image for input: %d", egl_result);
                gst_buffer_unmap(inbuf, &in_map);
                gst_buffer_unref(outbuf);
                gst_buffer_unref(inbuf);
                return GST_FLOW_ERROR;
            }
        }
        
        // Используем кешированный маппинг или регистрируем новый
        void* egl_image = in_surface->surfaceList[0].mappedAddr.eglImage;
        input_ptr = (unsigned char*)get_cached_cuda_pointer(egl_image);
        
        if (!input_ptr) {
            LOG_ERROR(vcam, "Failed to get CUDA pointer for input EGL image");
            gst_buffer_unmap(inbuf, &in_map);
            gst_buffer_unref(outbuf);
            gst_buffer_unref(inbuf);
            return GST_FLOW_ERROR;
        }
        
    } else if (in_surface->memType == NVBUF_MEM_CUDA_DEVICE || 
               in_surface->memType == NVBUF_MEM_CUDA_UNIFIED) {
        // CUDA память - используем напрямую
        LOG_DEBUG(vcam, "Input is CUDA memory (type %d), direct access", in_surface->memType);
        input_ptr = (unsigned char*)in_surface->surfaceList[0].dataPtr;
    } else {
        LOG_ERROR(vcam, "Unsupported input memory type: %d", in_surface->memType);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // ВЫХОДНОЙ БУФЕР - используем заранее зарегистрированный указатель
    if (out_memory->cuda_registered && !out_memory->frame_memory_ptrs.empty()) {
        output_ptr = (unsigned char*)out_memory->frame_memory_ptrs[0];
    } else {
        LOG_ERROR(vcam, "Output buffer not properly registered in CUDA");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Проверка валидности указателей
    if (!input_ptr || !output_ptr) {
        LOG_ERROR(vcam, "Invalid GPU pointers: input=%p, output=%p", 
                  input_ptr, output_ptr);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }


    // ВАЛИДАЦИЯ: проверяем что размеры панорамы установлены через properties
    if (vcam->input_width == 0 || vcam->input_height == 0) {
        LOG_ERROR(vcam, "❌ ОШИБКА: panorama-width и panorama-height ОБЯЗАТЕЛЬНЫ!");
        LOG_ERROR(vcam, "   Добавьте в pipeline: panorama-width=6528 panorama-height=1800");
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Валидация размеров входного буфера
    if (in_surface->surfaceList[0].width != vcam->input_width ||
        in_surface->surfaceList[0].height != vcam->input_height) {
        LOG_ERROR(vcam, "Invalid input buffer size: %dx%d (expected %dx%d)",
                  in_surface->surfaceList[0].width,
                  in_surface->surfaceList[0].height,
                  vcam->input_width, vcam->input_height);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Валидация размеров выходного буфера (должен соответствовать настройкам)
    if (out_surface->surfaceList[0].width != vcam->output_width ||
        out_surface->surfaceList[0].height != vcam->output_height) {
        LOG_ERROR(vcam, "Invalid output buffer size: %dx%d (expected %dx%d)",
                  out_surface->surfaceList[0].width,
                  out_surface->surfaceList[0].height,
                  vcam->output_width, vcam->output_height);
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    // Конфигурация kernel
    vcam->kernel_config.input_width = in_surface->surfaceList[0].width;
    vcam->kernel_config.input_height = in_surface->surfaceList[0].height;
    vcam->kernel_config.input_pitch = in_surface->surfaceList[0].planeParams.pitch[0];
    vcam->kernel_config.output_width = out_surface->surfaceList[0].width;
    vcam->kernel_config.output_height = out_surface->surfaceList[0].height;
    vcam->kernel_config.output_pitch = out_surface->surfaceList[0].planeParams.pitch[0];
    
    LOG_DEBUG(vcam, "CUDA Kernel Config: in=%dx%d (pitch=%d), out=%dx%d (pitch=%d)",
             vcam->kernel_config.input_width,
             vcam->kernel_config.input_height,
             vcam->kernel_config.input_pitch,
             vcam->kernel_config.output_width,
             vcam->kernel_config.output_height,
             vcam->kernel_config.output_pitch);
    
    // CUDA PROCESSING
    cuda_err = cudaSetDevice(vcam->gpu_id);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "Failed to set CUDA device: %s", cudaGetErrorString(cuda_err));
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Начало измерения производительности
    auto start_time = std::chrono::high_resolution_clock::now();

    // Вызов CUDA kernel - формат зависит от входного формата
    if (input_format == NVBUF_COLOR_FORMAT_NV12) {
        // NV12 input: separate Y and UV planes
        // Use planeParams.offset[] for correct plane addressing
        unsigned char* input_y_ptr = input_ptr +
            in_surface->surfaceList[0].planeParams.offset[0];
        unsigned char* input_uv_ptr = input_ptr +
            in_surface->surfaceList[0].planeParams.offset[1];

        int pitch_y = in_surface->surfaceList[0].planeParams.pitch[0];
        int pitch_uv = in_surface->surfaceList[0].planeParams.pitch[1];

        LOG_DEBUG(vcam, "NV12 remap: Y offset=%u, UV offset=%u, pitch_y=%d, pitch_uv=%d",
                  in_surface->surfaceList[0].planeParams.offset[0],
                  in_surface->surfaceList[0].planeParams.offset[1],
                  pitch_y, pitch_uv);

        cuda_err = apply_virtual_camera_remap_nv12(
            input_y_ptr,
            input_uv_ptr,
            output_ptr,
            vcam->remap_u_gpu,
            vcam->remap_v_gpu,
            &vcam->kernel_config,
            pitch_y,
            pitch_uv,
            vcam->cuda_stream
        );
    } else {
        // RGBA input: single plane
        LOG_DEBUG(vcam, "Calling RGBA remap kernel");

        cuda_err = apply_virtual_camera_remap(
            input_ptr,
            output_ptr,
            vcam->remap_u_gpu,
            vcam->remap_v_gpu,
            &vcam->kernel_config,
            vcam->cuda_stream
        );
    }

    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "CUDA processing failed: %s", cudaGetErrorString(cuda_err));
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }
    
    // Ждем завершения CUDA операций
    cuda_err = cudaStreamSynchronize(vcam->cuda_stream);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR(vcam, "CUDA stream synchronization failed: %s",
                  cudaGetErrorString(cuda_err));
        gst_buffer_unmap(inbuf, &in_map);
        gst_buffer_unref(outbuf);
        gst_buffer_unref(inbuf);
        return GST_FLOW_ERROR;
    }

    gst_buffer_unmap(inbuf, &in_map);
    
    // Измерение производительности
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    vcam->total_processing_time += duration.count();
    
    if (vcam->min_processing_time == 0 || (guint64)duration.count() < vcam->min_processing_time) {
        vcam->min_processing_time = duration.count();
    }
    if ((guint64)duration.count() > vcam->max_processing_time) {
        vcam->max_processing_time = duration.count();
    }
    
    // Timestamps и метаданные
    GST_BUFFER_PTS(outbuf) = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DTS(outbuf) = GST_BUFFER_DTS(inbuf);
    GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
    GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
    GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
    GST_BUFFER_FLAGS(outbuf) = GST_BUFFER_FLAGS(inbuf);
    
    // Копирование метаданных
    gst_buffer_copy_into(outbuf, inbuf,
                        (GstBufferCopyFlags)(GST_BUFFER_COPY_META), 0, -1);
    
    // Push буфера
    flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(btrans), outbuf);
    
    gst_buffer_unref(inbuf);
    
    vcam->frame_count++;
    
    // Периодическое логирование статистики
    if (vcam->frame_count % 300 == 0) {
        guint64 avg_time = vcam->total_processing_time / vcam->frame_count;
        LOG_INFO(vcam, "Performance stats after %lu frames:", vcam->frame_count);
        LOG_INFO(vcam, "  Average: %lu µs (%.1f FPS)", avg_time, 1000000.0 / avg_time);
        LOG_INFO(vcam, "  Min: %lu µs, Max: %lu µs", 
                 vcam->min_processing_time, vcam->max_processing_time);
        LOG_INFO(vcam, "  Current view: yaw=%.1f°, pitch=%.1f°, roll=%.1f°, fov=%.1f°",
                 vcam->yaw, vcam->pitch, vcam->roll, vcam->fov);
    }
    
    vcam->last_flow_ret = flow_ret;
    return flow_ret;
}

static GstFlowReturn
gst_nvds_virtual_cam_generate_output(GstBaseTransform *btrans, GstBuffer **outbuf)
{
    // Критически важно: возвращаем NULL
    *outbuf = NULL;
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(btrans);
    return vcam->last_flow_ret;
}

/* ============================================================================
 * Caps negotiation
 * ============================================================================ */

static GstCaps* 
gst_nvds_virtual_cam_transform_caps(GstBaseTransform *trans,
                                    GstPadDirection direction,
                                    GstCaps *caps,
                                    GstCaps *filter)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(trans);
    GstCaps *othercaps = NULL;
    
    if (direction == GST_PAD_SINK) {
        // Sink->Src: всегда выдаем 1920x1080
        othercaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "RGBA",
            "width", G_TYPE_INT, 1920,
            "height", G_TYPE_INT, 1080,
            NULL);
        gst_caps_set_features(othercaps, 0, 
            gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_NVMM, NULL));
    } else {
        // Src->Sink: требуем размер входной панорамы
        othercaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "RGBA",
            "width", G_TYPE_INT, vcam->input_width,
            "height", G_TYPE_INT, vcam->input_height,
            NULL);
        gst_caps_set_features(othercaps, 0, 
            gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_NVMM, NULL));
    }
    
    if (filter) {
        GstCaps *intersect = gst_caps_intersect(othercaps, filter);
        gst_caps_unref(othercaps);
        othercaps = intersect;
    }
    
    return othercaps;
}

static GstCaps* 
gst_nvds_virtual_cam_fixate_caps(GstBaseTransform *trans,
                                 GstPadDirection direction,
                                 GstCaps *caps,
                                 GstCaps *othercaps)
{
    // Уже всё фиксировано, просто передаем дальше
    return gst_caps_fixate(othercaps);
}


static gboolean 
gst_nvds_virtual_cam_set_caps(GstBaseTransform *btrans,
                              GstCaps *incaps, GstCaps *outcaps)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(btrans);
    
    LOG_INFO(vcam, "set_caps called");
    LOG_INFO(vcam, "incaps: %" GST_PTR_FORMAT, incaps);
    LOG_INFO(vcam, "outcaps: %" GST_PTR_FORMAT, outcaps);
    
    GstStructure *in_struct = gst_caps_get_structure(incaps, 0);
    GstStructure *out_struct = gst_caps_get_structure(outcaps, 0);
    
    gint in_width, in_height;
    gst_structure_get_int(in_struct, "width", &in_width);
    gst_structure_get_int(in_struct, "height", &in_height);
    
    gint out_width, out_height;
    gst_structure_get_int(out_struct, "width", &out_width);
    gst_structure_get_int(out_struct, "height", &out_height);
    
    LOG_INFO(vcam, "Negotiated: Input %dx%d -> Output %dx%d", 
             in_width, in_height, out_width, out_height);
    
    // Обновляем конфигурацию
    vcam->kernel_config.input_width = in_width;
    vcam->kernel_config.input_height = in_height;
    vcam->kernel_config.output_width = out_width;
    vcam->kernel_config.output_height = out_height;
    
    // Сбрасываем кеши при изменении размеров
    vcam->rays_computed = FALSE;
    vcam->lut_cache.valid = FALSE;
    
    return TRUE;
}

/* ============================================================================
 * START/STOP
 * ============================================================================ */

static gboolean 
gst_nvds_virtual_cam_start(GstBaseTransform *trans)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(trans);
    
    LOG_INFO(vcam, "Starting nvdsvirtualcam");
    LOG_INFO(vcam, "Output: %dx%d, FOV: %.1f°, GPU: %d", 
             vcam->output_width, vcam->output_height, vcam->fov, vcam->gpu_id);
    
    // Allocate CUDA resources
    if (!allocate_cuda_resources(vcam)) {
        LOG_ERROR(vcam, "Failed to allocate CUDA resources");
        return FALSE;
    }
    
    // Create output buffer pool с новым allocator
    vcam->output_pool = gst_buffer_pool_new();
    if (!vcam->output_pool) {
        LOG_ERROR(vcam, "Failed to create output buffer pool");
        free_cuda_resources(vcam);
        return FALSE;
    }
    
    GstStructure *pool_config = gst_buffer_pool_get_config(vcam->output_pool);
    
    // Используем gstnvdsstitch_allocator
    GstAllocator *allocator = gst_nvdsvirtualcam_allocator_new(
        vcam->output_width,
        vcam->output_height,
        vcam->gpu_id
    );
    
    if (!allocator) {
        LOG_ERROR(vcam, "Failed to create nvdsstitch allocator");
        gst_structure_free(pool_config);
        gst_object_unref(vcam->output_pool);
        vcam->output_pool = NULL;
        free_cuda_resources(vcam);
        return FALSE;
    }
    
    // Настройка caps для пула
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",
        "width", G_TYPE_INT, vcam->output_width,
        "height", G_TYPE_INT, vcam->output_height,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    gst_caps_set_features(caps, 0, gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_NVMM, NULL));
    
    // Configure pool
    gst_buffer_pool_config_set_params(pool_config, caps,
                                     sizeof(NvBufSurface), 
                                     FIXED_OUTPUT_POOL_SIZE + 2,
                                     FIXED_OUTPUT_POOL_SIZE + 4);
    
    GstAllocationParams allocation_params;
    memset(&allocation_params, 0, sizeof(allocation_params));
    
    gst_buffer_pool_config_set_allocator(pool_config, allocator, &allocation_params);
    
    gst_caps_unref(caps);
    gst_object_unref(allocator);
    
    if (!gst_buffer_pool_set_config(vcam->output_pool, pool_config)) {
        LOG_ERROR(vcam, "Failed to set config on output pool");
        gst_object_unref(vcam->output_pool);
        vcam->output_pool = NULL;
        free_cuda_resources(vcam);
        return FALSE;
    }
    
    if (!gst_buffer_pool_set_active(vcam->output_pool, TRUE)) {
        LOG_ERROR(vcam, "Failed to activate output pool");
        gst_object_unref(vcam->output_pool);
        vcam->output_pool = NULL;
        free_cuda_resources(vcam);
        return FALSE;
    }
    
    LOG_INFO(vcam, "Output buffer pool created and activated");
    
    // Инициализация фиксированного пула
    if (!setup_fixed_output_pool(vcam)) {
        LOG_ERROR(vcam, "Failed to setup fixed output pool");
        gst_buffer_pool_set_active(vcam->output_pool, FALSE);
        gst_object_unref(vcam->output_pool);
        vcam->output_pool = NULL;
        free_cuda_resources(vcam);
        return FALSE;
    }
    
    // Initialize kernel config
    vcam->kernel_config.lon_min = NvdsVirtualCamConfig::LON_MIN;
    vcam->kernel_config.lon_max = NvdsVirtualCamConfig::LON_MAX;
    vcam->kernel_config.lat_min = NvdsVirtualCamConfig::LAT_MIN;
    vcam->kernel_config.lat_max = NvdsVirtualCamConfig::LAT_MAX;
    
    // Reset state
    vcam->frame_count = 0;
    vcam->lut_cache.valid = FALSE;
    vcam->rays_computed = FALSE;
    vcam->last_flow_ret = GST_FLOW_OK;
    
    // Performance tracking
    vcam->total_processing_time = 0;
    vcam->max_processing_time = 0;
    vcam->min_processing_time = 0;
    vcam->last_perf_log_frame = 0;
    
    // Настройка оптимизаций CUDA
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    LOG_INFO(vcam, "nvdsvirtualcam started successfully");
    
    return TRUE;
}

static gboolean 
gst_nvds_virtual_cam_stop(GstBaseTransform *trans)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(trans);
    
    LOG_INFO(vcam, "Stopping nvdsvirtualcam");
    LOG_INFO(vcam, "Total frames processed: %lu", vcam->frame_count);
    
    if (vcam->frame_count > 0) {
        guint64 avg_time = vcam->total_processing_time / vcam->frame_count;
        LOG_INFO(vcam, "Performance: avg=%luµs, min=%luµs, max=%luµs",
                 avg_time, vcam->min_processing_time, vcam->max_processing_time);
    }
    
    // Очистка фиксированного пула
    if (vcam->output_pool_fixed.initialized) {
        g_mutex_lock(&vcam->output_pool_fixed.mutex);
        
        // Освобождаем все буферы
        for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
            if (vcam->output_pool_fixed.buffers[i]) {
                gst_buffer_unref(vcam->output_pool_fixed.buffers[i]);
                vcam->output_pool_fixed.buffers[i] = NULL;
            }
            vcam->output_pool_fixed.memories[i] = NULL;
        }
        
        vcam->output_pool_fixed.initialized = FALSE;
        vcam->output_pool_fixed.current_index = 0;
        
        g_mutex_unlock(&vcam->output_pool_fixed.mutex);
        g_mutex_clear(&vcam->output_pool_fixed.mutex);
        
        LOG_DEBUG(vcam, "Fixed output pool cleaned up");
    }
    
    // Deactivate and free output buffer pool
    if (vcam->output_pool) {
        if (gst_buffer_pool_is_active(vcam->output_pool)) {
            gst_buffer_pool_set_active(vcam->output_pool, FALSE);
        }
        gst_object_unref(vcam->output_pool);
        vcam->output_pool = NULL;
    }
    
    // Free CUDA resources
    free_cuda_resources(vcam);
    
    // Очищаем глобальный EGL кеш
    cleanup_egl_cache();
    
    // Сброс счетчиков
    vcam->frame_count = 0;
    vcam->total_processing_time = 0;
    vcam->max_processing_time = 0;
    vcam->min_processing_time = 0;
    
    LOG_INFO(vcam, "nvdsvirtualcam stopped successfully");
    
    return TRUE;
}

/* ============================================================================
 * Properties
 * ============================================================================ */

static void 
gst_nvds_virtual_cam_set_property(GObject *object, guint prop_id,
                                  const GValue *value, GParamSpec *pspec)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(object);
    
    switch (prop_id) {
        case PROP_YAW:
            vcam->yaw = g_value_get_float(value);
            // ДОБАВЛЕНО: автоматически обновляем roll при изменении yaw
            {
                const gfloat ROLL_MAX = NvdsVirtualCamConfig::ROLL_MAX;
                const gfloat YAW_MAX = NvdsVirtualCamConfig::YAW_MAX;
                gfloat normalized_pos = vcam->yaw / YAW_MAX;
                vcam->roll = normalized_pos * ROLL_MAX;
            }
            // Инвалидируем кеш с защитой mutex
            g_mutex_lock(&vcam->lut_cache.mutex);
            vcam->lut_cache.valid = FALSE;
            g_mutex_unlock(&vcam->lut_cache.mutex);
            break;
        case PROP_PITCH:
            vcam->pitch = g_value_get_float(value);
            g_mutex_lock(&vcam->lut_cache.mutex);
            vcam->lut_cache.valid = FALSE;
            g_mutex_unlock(&vcam->lut_cache.mutex);
            break;
        case PROP_ROLL:
            vcam->roll = g_value_get_float(value);
            g_mutex_lock(&vcam->lut_cache.mutex);
            vcam->lut_cache.valid = FALSE;
            g_mutex_unlock(&vcam->lut_cache.mutex);
            break;
        case PROP_FOV:
            vcam->fov = g_value_get_float(value);
            vcam->rays_computed = FALSE;
            break;
        case PROP_GPU_ID:
            vcam->gpu_id = g_value_get_uint(value);
            break;
        case PROP_OUTPUT_WIDTH:
            vcam->output_width = g_value_get_uint(value);
            break;
        case PROP_OUTPUT_HEIGHT:
            vcam->output_height = g_value_get_uint(value);
            break;
        case PROP_PANORAMA_WIDTH:
            vcam->input_width = g_value_get_uint(value);
            vcam->kernel_config.input_width = vcam->input_width;
            break;
        case PROP_PANORAMA_HEIGHT:
            vcam->input_height = g_value_get_uint(value);
            vcam->kernel_config.input_height = vcam->input_height;
            break;
        case PROP_AUTO_FOLLOW:
            vcam->auto_follow = g_value_get_boolean(value);
            break;
        case PROP_SMOOTH_FACTOR:
            vcam->smooth_factor = g_value_get_float(value);
            break;
        case PROP_S_TARGET:
            vcam->s_target = g_value_get_float(value);
            break;
        case PROP_BALL_X:
            vcam->ball_x = g_value_get_float(value);
            vcam->tracking_active = TRUE;  // Активируем трекинг при обновлении
            break;
        case PROP_BALL_Y:
            vcam->ball_y = g_value_get_float(value);
            vcam->tracking_active = TRUE;
            break;
        case PROP_BALL_ACTUAL_RADIUS:
            vcam->ball_actual_radius = g_value_get_float(value);
            break;
        case PROP_TARGET_BALL_SIZE:
            vcam->target_ball_size = g_value_get_float(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

// ============================================================================
// ФУНКЦИЯ: update_camera_from_ball
// ============================================================================
// Эта функция управляет автоматическим слежением камеры за мячом.
// Она преобразует координаты мяча в панораме в целевые углы камеры.
//
// ЛОГИКА РАБОТЫ:
// 1. Преобразует пиксельные координаты мяча (ball_x, ball_y) в углы (yaw, pitch)
// 2. Добавляет смещение, если мяч близко к краю кадра (предсказание движения)
// 3. Границы применяются ПОСЛЕ этой функции в update_virtual_camera()
//
// ПАРАМЕТРЫ СМЕЩЕНИЯ:
// - EDGE_DISTANCE = 300px: зона у края, где добавляется смещение
// - offset_yaw = ±8°: горизонтальное смещение
// - offset_pitch = ±4°: вертикальное смещение
//
// Смещение нужно чтобы мяч не прилипал к краю кадра - камера заранее
// смещается в сторону движения мяча.
static void update_camera_from_ball(GstNvdsVirtualCam *vcam)
{
    // ============================================================================
    // ШАГ 1: ПРЕОБРАЗОВАНИЕ КООРДИНАТ МЯЧА В УГЛЫ КАМЕРЫ
    // ============================================================================
    // Функция pano_xy_to_yaw_pitch использует ту же формулу, что и в CUDA ядре,
    // но инвертированную (обратное преобразование):
    // - ball_x → yaw (горизонтальный угол, -90° до +90°)
    // - ball_y → pitch (вертикальный угол, -27° до +27°)
    pano_xy_to_yaw_pitch(vcam->ball_x, vcam->ball_y,
                        vcam->kernel_config.input_width,
                        vcam->kernel_config.input_height,
                        &vcam->target_yaw, &vcam->target_pitch);

    // ============================================================================
    // ШАГ 2: ДОБАВЛЕНИЕ СМЕЩЕНИЯ ДЛЯ МЯЧА У КРАЯ
    // ============================================================================
    // Если мяч близко к краю кадра (в пределах EDGE_DISTANCE пикселей),
    // камера смещается в противоположную сторону, чтобы:
    // 1. Мяч не прилипал к краю кадра
    // 2. Было видно куда мяч движется (предсказание)
    //
    // УПРОЩЁННАЯ ЛОГИКА - без плавных зон и гистерезиса:
    // - Мяч либо у края (добавляем фиксированное смещение)
    // - Либо не у края (смещение = 0)
    const gfloat EDGE_DISTANCE = 300.0f;  // Пиксели от края панорамы
    gfloat offset_yaw = 0.0f;
    gfloat offset_pitch = 0.0f;

    // Горизонтальное смещение (лево/право):
    if (vcam->ball_x < EDGE_DISTANCE) {
        // Мяч слева → смещаем камеру вправо (показываем больше правой стороны)
        offset_yaw = 8.0f;
    } else if (vcam->ball_x > vcam->input_width - EDGE_DISTANCE) {
        // Мяч справа → смещаем камеру влево (показываем больше левой стороны)
        offset_yaw = -8.0f;
    }

    // Вертикальное смещение (верх/низ):
    if (vcam->ball_y < EDGE_DISTANCE) {
        // Мяч сверху → смещаем камеру вниз (показываем больше нижней части)
        offset_pitch = -4.0f;
    } else if (vcam->ball_y > vcam->input_height - EDGE_DISTANCE) {
        // Мяч снизу → смещаем камеру вверх (показываем больше верхней части)
        offset_pitch = 4.0f;
    }

    // Применяем смещение к целевым углам
    vcam->target_yaw += offset_yaw;
    vcam->target_pitch += offset_pitch;

    // ОТЛАДКА: выводим target_pitch после добавления offset
    // static int offset_log_counter = 0;
    // if (offset_log_counter++ % 30 == 0) {
    //     g_print("📐 TARGET_PITCH после offset: %.1f° (offset_pitch=%.1f°)\n",
    //             vcam->target_pitch, offset_pitch);
    // }

    // 3. Автоматический зум по размеру мяча
    // ИНВЕРТИРОВАННАЯ интерполяция: маленький мяч→зум (FOV_MIN), большой мяч→широко (FOV_MAX)
    using namespace NvdsVirtualCamConfig;

    gfloat radius = CLAMP(vcam->ball_actual_radius, BALL_RADIUS_MIN, BALL_RADIUS_MAX);

    // Линейная интерполяция FOV от радиуса: FOV = FOV_MIN + (radius - R_MIN) * slope
    gfloat fov_range = FOV_MAX - FOV_MIN;  // Диапазон FOV (например, 68-45=23°)
    gfloat radius_range = BALL_RADIUS_MAX - BALL_RADIUS_MIN;  // Диапазон радиуса (50-5=45px)
    gfloat slope = fov_range / radius_range;  // Наклон (например, 23/45≈0.511)

    vcam->target_fov = FOV_MIN + (radius - BALL_RADIUS_MIN) * slope;
    vcam->target_fov = CLAMP(vcam->target_fov, FOV_MIN, FOV_MAX);

    // ОТЛАДКА: выводим зум
    // static int zoom_log_counter = 0;
    // if (zoom_log_counter++ % 30 == 0) {
    //     g_print("🔎 ZOOM: radius=%.1fpx → target_fov=%.1f° (range: %.0f°-%.0f°, slope=%.3f)\n",
    //             radius, vcam->target_fov, FOV_MIN, FOV_MAX, slope);
    // }

    // 4. Проверка границ панорамы с учётом FOV
    // ТОЧНЫЕ границы панорамы - минимизируем черные полосы
    // При FOV>54° камера может выйти за границу с ОДНОЙ стороны (это нормально!)
    // ============================================================================
    // РАСЧЁТ ГРАНИЦ КАМЕРЫ С УЧЁТОМ СФЕРИЧЕСКОЙ ГЕОМЕТРИИ
    // ============================================================================
    // Эта секция решает ключевую проблему: как ограничить движение виртуальной
    // камеры так, чтобы она никогда не выходила за пределы панорамы и не создавала
    // чёрные полосы, при этом давая максимальную свободу движения.
    //
    // ПРОБЛЕМЫ, КОТОРЫЕ МЫ РЕШАЕМ:
    // 1. При FOV=68° и повороте к краю панорамы появляются чёрные полосы
    // 2. Когда камера повёрнута горизонтально (yaw ≠ 0), вертикальное движение
    //    создаёт более длинную дугу на сфере (сферическая геометрия)
    // 3. Нужен баланс между свободой движения и отсутствием чёрных полос
    //
    // НАСТРОЙКИ ГРАНИЦ ПАНОРАМЫ:
    // Вертикаль: -27° до +27° (54° покрытия, отцентрировано лучше чем -32/+22)
    // Горизонталь: -90° до +90° (180° покрытия)
    const gfloat EFFECTIVE_LAT_MIN = NvdsVirtualCamConfig::LAT_MIN;  // -27° (вертикальная граница низа)
    const gfloat EFFECTIVE_LAT_MAX = NvdsVirtualCamConfig::LAT_MAX;  // +27° (вертикальная граница верха)
    const gfloat EFFECTIVE_LON_MIN = NvdsVirtualCamConfig::LON_MIN;  // -90° (горизонтальная граница слева)
    const gfloat EFFECTIVE_LON_MAX = NvdsVirtualCamConfig::LON_MAX;  // +90° (горизонтальная граница справа)

    // ============================================================================
    // ШАГ 1: БАЗОВЫЙ РАСЧЁТ FOV
    // ============================================================================
    // Вертикальный FOV камеры и соответствующий горизонтальный FOV
    gfloat half_fov = vcam->target_fov / 2.0f;                      // Половина вертикального FOV
    gfloat aspect_ratio = 16.0f / 9.0f;                              // Соотношение сторон экрана
    gfloat horizontal_fov = vcam->target_fov * aspect_ratio;         // Горизонтальный FOV (шире из-за aspect ratio)
    gfloat half_fov_h = horizontal_fov / 2.0f;                       // Половина горизонтального FOV

    // ============================================================================
    // ШАГ 2: КОЭФФИЦИЕНТЫ СФЕРИЧЕСКОЙ ПРОЕКЦИИ
    // ============================================================================
    // На equirectangular (сферической) проекции FOV камеры не совпадает 1:1 с
    // угловым покрытием панорамы. Нужны коэффициенты коррекции.
    //
    // ВЕРТИКАЛЬНЫЙ КОЭФФИЦИЕНТ:
    // - Панорама: 54° высоты (от -27° до +27°)
    // - Максимальный FOV камеры: 68°
    // - Базовое соотношение: 54/68 ≈ 0.794
    // - Умножаем на 0.8 для баланса → SPHERICAL_FACTOR_V ≈ 0.635
    // - Это означает: FOV=68° реально покрывает ~43° панорамы (68 × 0.635)
    //
    // ГОРИЗОНТАЛЬНЫЙ КОЭФФИЦИЕНТ:
    // - На экваторе (центр по вертикали) искажения меньше
    // - Используем 0.63 эмпирически (баланс свободы и безопасности)
    const gfloat PANORAMA_HEIGHT = EFFECTIVE_LAT_MAX - EFFECTIVE_LAT_MIN;  // 54°
    const gfloat MAX_FOV = NvdsVirtualCamConfig::FOV_MAX;  // Из конфига (68°)

    const gfloat SPHERICAL_FACTOR_V = PANORAMA_HEIGHT / MAX_FOV * 0.8;  // ≈ 0.635 (динамически)
    const gfloat SPHERICAL_FACTOR_H = 0.63f;  // Оптимизировано вручную

    // Реальное покрытие FOV на панораме (после применения коэффициентов)
    gfloat effective_half_fov_v = half_fov * SPHERICAL_FACTOR_V;  // Реальное вертикальное покрытие
    gfloat effective_half_fov_h = half_fov_h * SPHERICAL_FACTOR_H; // Реальное горизонтальное покрытие

    // ============================================================================
    // ШАГ 3: КОРРЕКЦИЯ ДЛЯ СФЕРИЧЕСКОЙ ГЕОМЕТРИИ (YAW FACTOR)
    // ============================================================================
    // КЛЮЧЕВАЯ ПРОБЛЕМА:
    // Когда камера повёрнута по горизонтали (yaw ≠ 0), вертикальное движение
    // (изменение pitch) создаёт более ДЛИННУЮ дугу на сфере.
    //
    // Представь глобус:
    // - В центре (yaw=0°): движение вверх-вниз идёт по меридиану (кратчайший путь)
    // - На краю (yaw=±90°): движение вверх-вниз создаёт диагональную дугу (длиннее!)
    //
    // РЕШЕНИЕ:
    // Используем cos(yaw) для динамической коррекции вертикальных границ:
    // - yaw=0° (центр): cos(0) = 1.0 → границы не меняются
    // - yaw=±90° (края): cos(±90°) ≈ 0.0 → границы сужаются
    //
    // Нормализуем yaw в диапазон [-1, 1]:
    gfloat yaw_normalized = (vcam->target_yaw - EFFECTIVE_LON_MIN) / (EFFECTIVE_LON_MAX - EFFECTIVE_LON_MIN);
    yaw_normalized = (yaw_normalized - 0.5f) * 2.0f;  // [0,1] → [-1,1]

    // Вычисляем коэффициент коррекции через косинус:
    gfloat yaw_factor = cosf(yaw_normalized * M_PI * 0.5f);  // cos(0)=1.0, cos(±π/2)=0.0

    // ВАЖНО: Используем ДЕЛЕНИЕ, а не умножение!
    // При попытке использовать умножение пользователь сказал "стало хуже, верни"
    // Деление работает правильно: чем меньше yaw_factor, тем меньше corrected_half_fov_v
    // Минимум 0.6 предотвращает слишком сильное сужение на краях
    gfloat corrected_half_fov_v = effective_half_fov_v / fmaxf(yaw_factor, 0.6f);

    // ============================================================================
    // ШАГ 4: КРИТИЧНОЕ ОГРАНИЧЕНИЕ (CLAMPING) - ПРЕДОТВРАЩЕНИЕ ЧЁРНЫХ ПОЛОС
    // ============================================================================
    // ПРОБЛЕМА БЕЗ CLAMPING:
    // При большом FOV (например 68°) и повороте к краю панорамы,
    // corrected_half_fov_v может стать больше половины высоты панорамы (27°).
    // Это приводит к:
    //   pitch_min_safe = -27 + 30 = +3°
    //   pitch_max_safe = +27 - 30 = -3°
    //   → pitch_min >= pitch_max (НЕВАЛИДНО!)
    //
    // Когда границы невалидны, fallback механизм их фиксирует, но пользователь видит:
    // - При максимальном зуме у верхнего края - чёрные полосы сверху
    // - При максимальном зуме у нижнего края - чёрные полосы снизу
    // - При максимальном зуме у боковых краёв - чёрные полосы по бокам
    //
    // РЕШЕНИЕ:
    // Ограничиваем corrected_half_fov_v и effective_half_fov_h, чтобы они НИКОГДА
    // не превышали половину размера панорамы минус небольшой запас.
    //
    // ЗАПАСЫ (подобраны пользователем через итеративное тестирование):
    // - Вертикальный: -0.2° (баланс между свободой и отсутствием полос)
    // - Горизонтальный: -0.1° (максимум свободы, минимальный запас)
    //
    // ВЕРТИКАЛЬНОЕ ОГРАНИЧЕНИЕ:
    gfloat max_half_panorama_v = fminf(EFFECTIVE_LAT_MAX, -EFFECTIVE_LAT_MIN);  // 27°
    corrected_half_fov_v = fminf(corrected_half_fov_v, max_half_panorama_v - 0.2f);  // Запас -0.2° (итеративно подобрано)

    // ГОРИЗОНТАЛЬНОЕ ОГРАНИЧЕНИЕ:
    gfloat max_half_panorama_h = (EFFECTIVE_LON_MAX - EFFECTIVE_LON_MIN) / 2.0f;  // 90°
    gfloat clamped_half_fov_h = fminf(effective_half_fov_h, max_half_panorama_h - 0.1f);  // Запас -0.1° (итеративно подобрано)

    // ============================================================================
    // ШАГ 5: ВЫЧИСЛЕНИЕ БЕЗОПАСНЫХ ГРАНИЦ
    // ============================================================================
    // Теперь вычисляем "безопасную зону" - диапазон, в котором может находиться
    // ЦЕНТР камеры, чтобы её края не выходили за панораму.
    //
    // Логика:
    // - Если камера видит ±26.8° от центра (corrected_half_fov_v = 26.8°)
    // - То центр камеры может быть от -0.2° до +0.2° (очень узкая зона)
    // - Чем меньше FOV, тем шире безопасная зона
    gfloat pitch_min_safe = EFFECTIVE_LAT_MIN + corrected_half_fov_v;  // Минимум для центра камеры
    gfloat pitch_max_safe = EFFECTIVE_LAT_MAX - corrected_half_fov_v;  // Максимум для центра камеры
    gfloat yaw_min_safe = EFFECTIVE_LON_MIN + clamped_half_fov_h;
    gfloat yaw_max_safe = EFFECTIVE_LON_MAX - clamped_half_fov_h;

    // ============================================================================
    // ШАГ 6: FALLBACK ДЛЯ НЕВАЛИДНЫХ ГРАНИЦ
    // ============================================================================
    // Если границы пересеклись (min >= max), это означает что FOV слишком большой
    // для текущей позиции камеры. В этом случае мы НЕ отключаем ограничения,
    // а наоборот делаем их СТРОЖЕ - фиксируем камеру в центре с узким диапазоном ±2°.
    //
    // Это предотвращает ситуацию, когда камера застревает или выходит за границы.
    if (pitch_min_safe >= pitch_max_safe) {
        // Границы пересеклись по вертикали
        gfloat center_pitch = (EFFECTIVE_LAT_MIN + EFFECTIVE_LAT_MAX) / 2.0f;  // Центр панорамы
        pitch_min_safe = center_pitch - 2.0f;  // Узкий диапазон ±2°
        pitch_max_safe = center_pitch + 2.0f;
    }

    if (yaw_min_safe >= yaw_max_safe) {
        // Границы пересеклись по горизонтали
        gfloat center_yaw = (EFFECTIVE_LON_MIN + EFFECTIVE_LON_MAX) / 2.0f;
        yaw_min_safe = center_yaw - 2.0f;
        yaw_max_safe = center_yaw + 2.0f;
    }

    // ============================================================================
    // ШАГ 7: ДВУХУРОВНЕВОЕ ПРИМЕНЕНИЕ ГРАНИЦ
    // ============================================================================
    // Применяем ограничения в ДВА уровня для максимальной надёжности:
    //
    // УРОВЕНЬ 1: Абсолютные границы панорамы
    // - Центр камеры НИКОГДА не выходит за -27°/+27° (вертикаль) и -90°/+90° (горизонталь)
    // - Это гарантирует что центр камеры всегда в панораме
    //
    // УРОВЕНЬ 2: Безопасная зона (с учётом FOV)
    // - Центр камеры дополнительно ограничен так, чтобы края камеры не выходили
    // - Это предотвращает чёрные полосы
    //
    // ВЕРТИКАЛЬ (pitch):
    vcam->target_pitch = fmaxf(EFFECTIVE_LAT_MIN, fminf(EFFECTIVE_LAT_MAX, vcam->target_pitch));  // Уровень 1
    vcam->target_pitch = fmaxf(pitch_min_safe, fminf(pitch_max_safe, vcam->target_pitch));        // Уровень 2

    // ГОРИЗОНТАЛЬ (yaw):
    vcam->target_yaw = fmaxf(EFFECTIVE_LON_MIN, fminf(EFFECTIVE_LON_MAX, vcam->target_yaw));      // Уровень 1
    vcam->target_yaw = fmaxf(yaw_min_safe, fminf(yaw_max_safe, vcam->target_yaw));                // Уровень 2

    // ОТЛАДКА: вывод позиции камеры (каждые 30 кадров)
    // static int boundary_log_counter = 0;
    // if (boundary_log_counter++ % 30 == 0) {
    //     g_print("🔧 CAMERA: FOV=%.1f° pitch=%.1f° (%.1f°..%.1f°) yaw=%.1f° (%.1f°..%.1f°)\n",
    //             vcam->target_fov, vcam->target_pitch, pitch_min_safe, pitch_max_safe,
    //             vcam->target_yaw, yaw_min_safe, yaw_max_safe);
    // }

    // 5. Вызываем сглаживание
    smooth_camera_tracking(vcam);
}





static void 
gst_nvds_virtual_cam_get_property(GObject *object, guint prop_id,
                                  GValue *value, GParamSpec *pspec)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(object);
    
    switch (prop_id) {
        case PROP_YAW:
            g_value_set_float(value, vcam->yaw);
            break;
        case PROP_PITCH:
            g_value_set_float(value, vcam->pitch);
            break;
        case PROP_ROLL:
            g_value_set_float(value, vcam->roll);
            break;
        case PROP_FOV:
            g_value_set_float(value, vcam->fov);
            break;
        case PROP_GPU_ID:
            g_value_set_uint(value, vcam->gpu_id);
            break;
        case PROP_OUTPUT_WIDTH:
            g_value_set_uint(value, vcam->output_width);
            break;
        case PROP_OUTPUT_HEIGHT:
            g_value_set_uint(value, vcam->output_height);
            break;
        case PROP_PANORAMA_WIDTH:
            g_value_set_uint(value, vcam->input_width);
            break;
        case PROP_PANORAMA_HEIGHT:
            g_value_set_uint(value, vcam->input_height);
            break;
        case PROP_AUTO_FOLLOW:
            g_value_set_boolean(value, vcam->auto_follow);
            break;
        case PROP_SMOOTH_FACTOR:
            g_value_set_float(value, vcam->smooth_factor);
            break;
        case PROP_S_TARGET:
            g_value_set_float(value, vcam->s_target);
            break;
        case PROP_BALL_X:
            g_value_set_float(value, vcam->ball_x);
            break;
        case PROP_BALL_Y:
            g_value_set_float(value, vcam->ball_y);
            break;
        case PROP_BALL_ACTUAL_RADIUS:
            g_value_set_float(value, vcam->ball_actual_radius);
            break;
        case PROP_TARGET_BALL_SIZE:
            g_value_set_float(value, vcam->target_ball_size);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvds_virtual_cam_finalize(GObject *object)
{
    GstNvdsVirtualCam *vcam = GST_NVDS_VIRTUAL_CAM(object);

    LOG_DEBUG(vcam, "Finalizing nvdsvirtualcam");

    // Очистка mutex
    g_mutex_clear(&vcam->lut_cache.mutex);
    g_mutex_clear(&vcam->properties_mutex);

    // Вызываем родительский финализатор
    G_OBJECT_CLASS(gst_nvds_virtual_cam_parent_class)->finalize(object);
}

/* ============================================================================
 * Class Init
 * ============================================================================ */

static void 
gst_nvds_virtual_cam_class_init(GstNvdsVirtualCamClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);
    
    gst_element_class_set_static_metadata(gstelement_class,
        "NvDsVirtualCam", "Video/Filter",
        "NVIDIA DeepStream Virtual Camera Plugin", "NVIDIA");
    
    gst_element_class_add_static_pad_template(gstelement_class, &sink_template);
    gst_element_class_add_static_pad_template(gstelement_class, &src_template);
    
    gobject_class->set_property = gst_nvds_virtual_cam_set_property;
    gobject_class->get_property = gst_nvds_virtual_cam_get_property;
    gobject_class->finalize = gst_nvds_virtual_cam_finalize;
    
    // Properties
    g_object_class_install_property(gobject_class, PROP_S_TARGET,
    g_param_spec_float("s-target", "S Target",
                      "Target screen fraction for tracked object (0.01-0.1)",
                      0.01f, 0.1f, 0.035f,
                      (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_YAW,
        g_param_spec_float("yaw", "Yaw", "Camera yaw angle",
                          NvdsVirtualCamConfig::YAW_MIN, 
                          NvdsVirtualCamConfig::YAW_MAX, 
                          NvdsVirtualCamConfig::DEFAULT_YAW,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_PITCH,
        g_param_spec_float("pitch", "Pitch", "Camera pitch angle",
                          NvdsVirtualCamConfig::PITCH_MIN,
                          NvdsVirtualCamConfig::PITCH_MAX,
                          NvdsVirtualCamConfig::DEFAULT_PITCH,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_ROLL,
        g_param_spec_float("roll", "Roll", "Camera roll angle",
                          NvdsVirtualCamConfig::ROLL_MIN,
                          NvdsVirtualCamConfig::ROLL_MAX,
                          NvdsVirtualCamConfig::DEFAULT_ROLL,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_FOV,
        g_param_spec_float("fov", "FOV", "Field of view",
                          NvdsVirtualCamConfig::FOV_MIN,
                          NvdsVirtualCamConfig::FOV_MAX,
                          NvdsVirtualCamConfig::DEFAULT_FOV,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "GPU Device ID",
                         0, 7, NvdsVirtualCamConfig::GPU_ID,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_OUTPUT_WIDTH,
        g_param_spec_uint("output-width", "Output Width", "Virtual view width",
                         320, 3840, NvdsVirtualCamConfig::DEFAULT_OUTPUT_WIDTH,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_OUTPUT_HEIGHT,
        g_param_spec_uint("output-height", "Output Height", "Virtual view height",
                         240, 2160, NvdsVirtualCamConfig::DEFAULT_OUTPUT_HEIGHT,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_WIDTH,
        g_param_spec_uint("panorama-width", "Panorama Width", "Input panorama width (REQUIRED!)",
                         0, 10000, 0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО передавать через properties!
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PANORAMA_HEIGHT,
        g_param_spec_uint("panorama-height", "Panorama Height", "Input panorama height (REQUIRED!)",
                         0, 10000, 0,  // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_AUTO_FOLLOW,
        g_param_spec_boolean("auto-follow", "Auto Follow",
                            "Enable automatic object tracking",
                            FALSE,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_SMOOTH_FACTOR,
        g_param_spec_float("smooth-factor", "Smooth Factor",
                          "Smoothing factor for camera movement (0.0-1.0)",
                          0.0f, 1.0f, 0.15f,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_BALL_X,
        g_param_spec_float("ball-x", "Ball X", "Ball X position on panorama",
                        0.0f, 10000.0f, 3264.0f,  // Дефолт: центр панорамы (6528/2)
                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_BALL_Y,
        g_param_spec_float("ball-y", "Ball Y", "Ball Y position on panorama",
                        0.0f, 10000.0f, 900.0f,  // Дефолт: центр панорамы (1800/2)
                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_BALL_ACTUAL_RADIUS,
        g_param_spec_float("ball-radius", "Ball Radius", "Ball radius in pixels",
                        1.0f, 100.0f, 20.0f,
                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_TARGET_BALL_SIZE,
        g_param_spec_float("target-ball-size", "Target Ball Size",
                        "Target ball size on screen (0.01-0.1)",
                        0.01f, 0.15f, 0.035f,
                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    // Callbacks - используем submit_input_buffer/generate_output pattern
    gstbasetransform_class->submit_input_buffer = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_submit_input_buffer);
    gstbasetransform_class->generate_output = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_generate_output);
    gstbasetransform_class->transform_caps = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_transform_caps);
    gstbasetransform_class->fixate_caps = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_fixate_caps);
    gstbasetransform_class->set_caps = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_set_caps);
    gstbasetransform_class->start = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_start);
    gstbasetransform_class->stop = 
        GST_DEBUG_FUNCPTR(gst_nvds_virtual_cam_stop);
}

/* ============================================================================
 * INIT
 * ============================================================================ */

static void 
gst_nvds_virtual_cam_init(GstNvdsVirtualCam *vcam)
{
    // Инициализация параметров по умолчанию
    vcam->yaw = NvdsVirtualCamConfig::DEFAULT_YAW;
    vcam->pitch = NvdsVirtualCamConfig::DEFAULT_PITCH;
    vcam->roll = NvdsVirtualCamConfig::DEFAULT_ROLL;
    vcam->fov = NvdsVirtualCamConfig::DEFAULT_FOV;
    g_mutex_init(&vcam->properties_mutex);  // Инициализация mutex для properties
    vcam->gpu_id = NvdsVirtualCamConfig::GPU_ID;
    vcam->output_width = NvdsVirtualCamConfig::DEFAULT_OUTPUT_WIDTH;
    vcam->output_height = NvdsVirtualCamConfig::DEFAULT_OUTPUT_HEIGHT;
    vcam->input_width = 0;   // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!
    vcam->input_height = 0;  // НЕТ дефолта - ОБЯЗАТЕЛЬНО через properties!

    vcam->ball_x = 0.0f;  // Будет установлен после получения input размеров
    vcam->ball_y = 816.0f;
    vcam->ball_actual_radius = 20.0f;
    vcam->target_ball_size = 0.035f;
    vcam->safe_fov_limit = NvdsVirtualCamConfig::FOV_MAX;
    
    // Режимы работы
    vcam->auto_follow = TRUE;
    vcam->smooth_factor = 0.15f;
    vcam->tracking_active = TRUE;
    vcam->tracked_object_id = 0;
    vcam->target_yaw = NvdsVirtualCamConfig::DEFAULT_YAW;     // 0.0° (центр)
    vcam->target_pitch = NvdsVirtualCamConfig::DEFAULT_PITCH;  // 0.0° (центр, было 15.0° захардкожено)
    vcam->target_fov = NvdsVirtualCamConfig::DEFAULT_FOV;      // 68° (максимальный обзор)

    // Параметры автозума
    vcam->s_target = NvdsVirtualCamConfig::S_TARGET_DEFAULT;  // 0.035f
    vcam->ball_angular_size = 0.0f;
    
    // Счетчики и состояние
    vcam->frame_count = 0;
    vcam->last_flow_ret = GST_FLOW_OK;
    
    // ВАЖНО: НЕ in-place - создаем новый буфер!
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(vcam), FALSE);
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(vcam), FALSE);
    
    // Инициализация указателей
    vcam->output_pool = NULL;
    vcam->cuda_stream = NULL;
    vcam->rays_gpu = NULL;
    vcam->remap_u_gpu = NULL;
    vcam->remap_v_gpu = NULL;
    vcam->rays_computed = FALSE;
    vcam->last_fov = 0.0f;
    
    // Инициализация кеша LUT
    vcam->lut_cache.valid = FALSE;
    vcam->lut_cache.last_yaw = 0.0f;
    vcam->lut_cache.last_pitch = 0.0f;
    vcam->lut_cache.last_roll = 0.0f;
    g_mutex_init(&vcam->lut_cache.mutex);  // Инициализация mutex
    
    // Инициализация фиксированного пула
    vcam->output_pool_fixed.initialized = FALSE;
    vcam->output_pool_fixed.current_index = 0;
    for (int i = 0; i < FIXED_OUTPUT_POOL_SIZE; i++) {
        vcam->output_pool_fixed.buffers[i] = NULL;
        vcam->output_pool_fixed.memories[i] = NULL;
    }
    
    // Инициализация конфигурации kernel
    memset(&vcam->kernel_config, 0, sizeof(vcam->kernel_config));
    vcam->kernel_config.input_width = vcam->input_width;
    vcam->kernel_config.input_height = vcam->input_height;
    vcam->kernel_config.output_width = vcam->output_width;
    vcam->kernel_config.output_height = vcam->output_height;
    
    // Performance metrics
    vcam->total_processing_time = 0;
    vcam->max_processing_time = 0;
    vcam->min_processing_time = 0;
    vcam->last_perf_log_frame = 0;
    
    // Метаданные
    vcam->add_virtual_cam_meta = FALSE;
    
    LOG_DEBUG(vcam, "Virtual camera initialized with defaults: yaw=%.1f, pitch=%.1f, roll=%.1f, fov=%.1f",
             vcam->yaw, vcam->pitch, vcam->roll, vcam->fov);
}

/* ============================================================================
 * Plugin Init
 * ============================================================================ */

static gboolean 
nvds_virtual_cam_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvds_virtual_cam_debug, "nvdsvirtualcam", 0,
                           "NVIDIA DeepStream Virtual Camera Plugin Debug");
    
    return gst_element_register(plugin, "nvdsvirtualcam", GST_RANK_PRIMARY,
                               GST_TYPE_NVDS_VIRTUAL_CAM);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsvirtualcam,
    "NVIDIA DeepStream Virtual Camera Plugin",
    nvds_virtual_cam_plugin_init,
    "1.0",
    "Proprietary",
    "nvdsvirtualcam",
    "https://developer.nvidia.com/"
)