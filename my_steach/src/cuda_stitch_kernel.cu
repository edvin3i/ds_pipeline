// cuda_stitch_kernel.cu - Исправленная версия CUDA kernel для панорамной склейки
#include "cuda_stitch_kernel.h"
#include "nvdsstitch_config.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <cfloat>  // Для FLT_MAX

// ============================================================================
// КОНСТАНТНАЯ ПАМЯТЬ ДЛЯ ЦВЕТОКОРРЕКЦИИ
// ============================================================================
__constant__ float g_color_gains[6];

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
extern "C" cudaError_t launch_panorama_kernel_fixed(
    const unsigned char* input_left,
    const unsigned char* input_right,
    unsigned char* output,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    const StitchKernelConfig* config,
    cudaStream_t stream,
    bool enable_edge_boost);

// ============================================================================
// СТРУКТУРА ДЛЯ КОНТЕКСТА ЦВЕТОКОРРЕКЦИИ
// ============================================================================
struct ColorCorrectionContext {
    float* d_sum_left;      // RGB суммы левой камеры
    float* d_sum_right;     // RGB суммы правой камеры  
    int* d_count_left;      // Счётчик пикселей левой
    int* d_count_right;     // Счётчик пикселей правой
    float prev_gains[6];    // Предыдущие значения для сглаживания
    bool initialized;
    // Pinned memory для асинхронного копирования
    float* h_sum_left;
    float* h_sum_right;
    int* h_count_left;
    int* h_count_right;
};

// ============================================================================
// БИЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ
// ============================================================================
__device__ inline uchar4 bilinear_sample(
    const unsigned char* image,
    float u, float v,
    int width, int height,
    int pitch)
{
    int x0 = __float2int_rd(u);  // floor
    int y0 = __float2int_rd(v);  // floor
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    x0 = max(0, x0);
    y0 = max(0, y0);
    
    float fx = u - x0;
    float fy = v - y0;
    
    uchar4 p00 = *((const uchar4*)(image + y0 * pitch + x0 * 4));
    uchar4 p10 = *((const uchar4*)(image + y0 * pitch + x1 * 4));
    uchar4 p01 = *((const uchar4*)(image + y1 * pitch + x0 * 4));
    uchar4 p11 = *((const uchar4*)(image + y1 * pitch + x1 * 4));
    
    float inv_fx = 1.0f - fx;
    float inv_fy = 1.0f - fy;
    
    float4 result;
    result.x = inv_fx * inv_fy * p00.x + fx * inv_fy * p10.x + 
               inv_fx * fy * p01.x + fx * fy * p11.x;
    result.y = inv_fx * inv_fy * p00.y + fx * inv_fy * p10.y + 
               inv_fx * fy * p01.y + fx * fy * p11.y;
    result.z = inv_fx * inv_fy * p00.z + fx * inv_fy * p10.z + 
               inv_fx * fy * p01.z + fx * fy * p11.z;
    
    return make_uchar4(
        __float2uint_rn(result.x),
        __float2uint_rn(result.y),
        __float2uint_rn(result.z),
        255
    );
}

// ============================================================================
// ОПТИМИЗИРОВАННОЕ ЯДРО АНАЛИЗА ЗОНЫ ПЕРЕКРЫТИЯ С SHARED MEMORY
// ============================================================================
__global__ void analyze_overlap_zone_kernel(
    const unsigned char* input_left,
    const unsigned char* input_right,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    float* rgb_sum_left,
    float* rgb_sum_right,
    int* pixel_count_left,
    int* pixel_count_right,
    int input_width,
    int input_height,
    int input_pitch,
    int output_width,
    int output_height)
{
    // Shared memory для редукции внутри блока
    extern __shared__ float shared_data[];
    float* block_sum_left = shared_data;  // 3 floats
    float* block_sum_right = &shared_data[3];  // 3 floats
    int* block_count_left = (int*)&shared_data[6];  // 1 int
    int* block_count_right = (int*)&shared_data[7];  // 1 int
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Инициализация shared memory
    if (tid == 0) {
        block_sum_left[0] = block_sum_left[1] = block_sum_left[2] = 0.0f;
        block_sum_right[0] = block_sum_right[1] = block_sum_right[2] = 0.0f;
        *block_count_left = 0;
        *block_count_right = 0;
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < output_width && y < output_height) {
        int lut_idx = y * output_width + x;
        
        float w_l = weight_left[lut_idx];
        float w_r = weight_right[lut_idx];
        
        // Анализируем только зону перекрытия (где оба веса значимы)
        const float overlap_threshold = 0.1f;
        
        if (w_l > overlap_threshold && w_r > overlap_threshold) {
            // Получаем координаты из LUT
            float left_u = lut_left_x[lut_idx];
            float left_v = lut_left_y[lut_idx];
            float right_u = lut_right_x[lut_idx];
            float right_v = lut_right_y[lut_idx];
            
            // Проверяем валидность координат
            if (left_u >= 0 && left_u < input_width && 
                left_v >= 0 && left_v < input_height &&
                right_u >= 0 && right_u < input_width && 
                right_v >= 0 && right_v < input_height) {
                
                // Сэмплируем пиксели
                uchar4 pixel_l = bilinear_sample(input_left, left_u, left_v,
                                                input_width, input_height, input_pitch);
                uchar4 pixel_r = bilinear_sample(input_right, right_u, right_v,
                                                input_width, input_height, input_pitch);
                
                // Атомарное добавление в shared memory
                atomicAdd(&block_sum_left[0], (float)pixel_l.x);
                atomicAdd(&block_sum_left[1], (float)pixel_l.y);
                atomicAdd(&block_sum_left[2], (float)pixel_l.z);
                atomicAdd(block_count_left, 1);
                
                atomicAdd(&block_sum_right[0], (float)pixel_r.x);
                atomicAdd(&block_sum_right[1], (float)pixel_r.y);
                atomicAdd(&block_sum_right[2], (float)pixel_r.z);
                atomicAdd(block_count_right, 1);
            }
        }
    }
    
    __syncthreads();
    
    // Финальная редукция: только первый поток блока пишет в глобальную память
    if (tid == 0) {
        if (*block_count_left > 0) {
            atomicAdd(&rgb_sum_left[0], block_sum_left[0]);
            atomicAdd(&rgb_sum_left[1], block_sum_left[1]);
            atomicAdd(&rgb_sum_left[2], block_sum_left[2]);
            atomicAdd(pixel_count_left, *block_count_left);
        }
        
        if (*block_count_right > 0) {
            atomicAdd(&rgb_sum_right[0], block_sum_right[0]);
            atomicAdd(&rgb_sum_right[1], block_sum_right[1]);
            atomicAdd(&rgb_sum_right[2], block_sum_right[2]);
            atomicAdd(pixel_count_right, *block_count_right);
        }
    }
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ УЛУЧШЕННОЙ ЦВЕТОКОРРЕКЦИИ
// ============================================================================
extern "C" cudaError_t init_color_correction_advanced(ColorCorrectionContext** ctx_out) {
    // Выделяем контекст
    ColorCorrectionContext* ctx = new ColorCorrectionContext();
    
    // Инициализируем поля
    ctx->d_sum_left = nullptr;
    ctx->d_sum_right = nullptr;
    ctx->d_count_left = nullptr;
    ctx->d_count_right = nullptr;
    ctx->h_sum_left = nullptr;
    ctx->h_sum_right = nullptr;
    ctx->h_count_left = nullptr;
    ctx->h_count_right = nullptr;
    
    // Выделяем постоянные буферы на GPU (один раз!)
    cudaError_t err;
    err = cudaMalloc(&ctx->d_sum_left, 3 * sizeof(float));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_sum_right, 3 * sizeof(float));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_count_left, sizeof(int));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_count_right, sizeof(int));
    if (err != cudaSuccess) goto error;
    
    // Выделяем pinned memory для быстрого асинхронного копирования
    err = cudaHostAlloc(&ctx->h_sum_left, 3 * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_sum_right, 3 * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_count_left, sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_count_right, sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    // Инициализация предыдущих gains
    for (int i = 0; i < 6; i++) {
        ctx->prev_gains[i] = 1.0f;
    }
    ctx->initialized = false;
    
    // Инициализация constant memory - объявляем ДО goto
    {
        float initial_gains[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        err = cudaMemcpyToSymbol(g_color_gains, initial_gains, 6 * sizeof(float));
        if (err != cudaSuccess) goto error;
    }
    
    *ctx_out = ctx;
    printf("✓ Advanced color correction initialized with persistent buffers\n");
    return cudaSuccess;

error:
    // Очистка при ошибке
    if (ctx) {
        if (ctx->d_sum_left) cudaFree(ctx->d_sum_left);
        if (ctx->d_sum_right) cudaFree(ctx->d_sum_right);
        if (ctx->d_count_left) cudaFree(ctx->d_count_left);
        if (ctx->d_count_right) cudaFree(ctx->d_count_right);
        if (ctx->h_sum_left) cudaFreeHost(ctx->h_sum_left);
        if (ctx->h_sum_right) cudaFreeHost(ctx->h_sum_right);
        if (ctx->h_count_left) cudaFreeHost(ctx->h_count_left);
        if (ctx->h_count_right) cudaFreeHost(ctx->h_count_right);
        delete ctx;
    }
    printf("ERROR: Failed to initialize color correction: %s\n", cudaGetErrorString(err));
    return err;
}


// ============================================================================
// ОБНОВЛЕНИЕ ЦВЕТОКОРРЕКЦИИ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)
// ============================================================================
extern "C" cudaError_t update_color_correction_advanced(
    const unsigned char* left_frame,
    const unsigned char* right_frame,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    int input_width,
    int input_height,
    int input_pitch,
    int output_width,
    int output_height,
    cudaStream_t stream,
    ColorCorrectionContext* ctx,
    float smoothing_factor)
{
    if (!ctx) return cudaErrorInvalidValue;
    
    // Очистка буферов
    cudaMemsetAsync(ctx->d_sum_left, 0, 3 * sizeof(float), stream);
    cudaMemsetAsync(ctx->d_sum_right, 0, 3 * sizeof(float), stream);
    cudaMemsetAsync(ctx->d_count_left, 0, sizeof(int), stream);
    cudaMemsetAsync(ctx->d_count_right, 0, sizeof(int), stream);
    
    // Запуск анализа
    dim3 block(16, 16);
    dim3 grid((output_width + block.x - 1) / block.x,
              (output_height + block.y - 1) / block.y);
    
    size_t shared_size = 6 * sizeof(float) + 2 * sizeof(int);
    
    analyze_overlap_zone_kernel<<<grid, block, shared_size, stream>>>(
        left_frame, right_frame,
        lut_left_x, lut_left_y,
        lut_right_x, lut_right_y,
        weight_left, weight_right,
        ctx->d_sum_left, ctx->d_sum_right,
        ctx->d_count_left, ctx->d_count_right,
        input_width, input_height, input_pitch,
        output_width, output_height);
    
    // Проверка ошибки
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: analyze_overlap_zone_kernel failed: %s\n", 
               cudaGetErrorString(err));
        return err;
    }
    
    // Асинхронное копирование
    cudaMemcpyAsync(ctx->h_sum_left, ctx->d_sum_left, 3 * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_sum_right, ctx->d_sum_right, 3 * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_count_left, ctx->d_count_left, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_count_right, ctx->d_count_right, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    
    // ВАЖНО: НЕ делаем cudaStreamSynchronize здесь!
    // Просто возвращаем успех
    
    return cudaSuccess;
}


// ============================================================================
// СОВМЕСТИМОСТЬ: СТАРАЯ ПРОСТАЯ ВЕРСИЯ (deprecated)
// ============================================================================
extern "C" cudaError_t init_color_correction() {
    float initial_gains[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    cudaError_t err = cudaMemcpyToSymbol(g_color_gains, initial_gains, 
                                         6 * sizeof(float));
    if (err != cudaSuccess) {
        printf("ERROR: Failed to init g_color_gains: %s\n", 
               cudaGetErrorString(err));
    } else {
        printf("Simple color correction initialized\n");
    }
    return err;
}

// Заглушка для старой функции
extern "C" cudaError_t update_color_correction_simple(
    const unsigned char* left_frame,
    const unsigned char* right_frame,
    const float* weight_left,
    const float* weight_right,
    int width,
    int height,
    int pitch,
    cudaStream_t stream)
{
    // Просто возвращаем успех для совместимости
    return cudaSuccess;
}

// ============================================================================
// ОСНОВНОЕ ЯДРО ПАНОРАМНОЙ СКЛЕЙКИ (С ИСПРАВЛЕНИЯМИ)
// ============================================================================
__global__ void panorama_lut_kernel(
    const unsigned char* __restrict__ input_left,
    const unsigned char* __restrict__ input_right,
    unsigned char* __restrict__ output,
    const float* __restrict__ lut_left_x,
    const float* __restrict__ lut_left_y,
    const float* __restrict__ lut_right_x,
    const float* __restrict__ lut_right_y,
    const float* __restrict__ weight_left,
    const float* __restrict__ weight_right,
    int input_width,
    int input_height,
    int input_pitch,
    int output_width,
    int output_height,
    int output_pitch,
    bool enable_edge_boost)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    int lut_idx = y * output_width + x;
    
    // Читаем координаты из LUT
    float left_u = lut_left_x[lut_idx];
    float left_v = lut_left_y[lut_idx];
    float right_u = lut_right_x[lut_idx];
    float right_v = lut_right_y[lut_idx];
    
    // Читаем веса
    float w_left = weight_left[lut_idx];
    float w_right = weight_right[lut_idx];
    
    uchar4 pixel_left = make_uchar4(0,0,0,0);
    uchar4 pixel_right = make_uchar4(0,0,0,0);
    
    // ВАЖНОЕ ИСПРАВЛЕНИЕ: Эффективные веса
    float wL_eff = 0.0f;  // Эффективный вес левой камеры
    float wR_eff = 0.0f;  // Эффективный вес правой камеры
    
    // Сэмплируем левую камеру ТОЛЬКО если координаты валидны
    if (w_left > 0.001f && left_u >= 0 && left_u < input_width && 
        left_v >= 0 && left_v < input_height) {
        
        pixel_left = bilinear_sample(input_left, left_u, left_v, 
                                     input_width, input_height, input_pitch);
        // Применяем цветокоррекцию
        pixel_left = make_uchar4(
            min(255, (int)((float)pixel_left.x * g_color_gains[0])),
            min(255, (int)((float)pixel_left.y * g_color_gains[1])),
            min(255, (int)((float)pixel_left.z * g_color_gains[2])),
            pixel_left.w
        );
        
        // Устанавливаем эффективный вес ТОЛЬКО если пиксель был получен
        wL_eff = w_left;
    }
    
    // Сэмплируем правую камеру ТОЛЬКО если координаты валидны
    if (w_right > 0.001f && right_u >= 0 && right_u < input_width && 
        right_v >= 0 && right_v < input_height) {
        
        pixel_right = bilinear_sample(input_right, right_u, right_v,
                                      input_width, input_height, input_pitch);
        // Применяем цветокоррекцию
        pixel_right = make_uchar4(
            min(255, (int)((float)pixel_right.x * g_color_gains[3])),
            min(255, (int)((float)pixel_right.y * g_color_gains[4])),
            min(255, (int)((float)pixel_right.z * g_color_gains[5])),
            pixel_right.w
        );
        
        // Устанавливаем эффективный вес ТОЛЬКО если пиксель был получен
        wR_eff = w_right;
    }
    
    // ИСПРАВЛЕНО: Используем ЭФФЕКТИВНЫЕ веса для смешивания
    float4 result;
    const float eps = 1e-6f;
    float total_weight = wL_eff + wR_eff + eps;  // Теперь используем wL_eff и wR_eff!
    
    // Смешиваем с эффективными весами
    result.x = ((float)pixel_left.x * wL_eff + (float)pixel_right.x * wR_eff) / total_weight;
    result.y = ((float)pixel_left.y * wL_eff + (float)pixel_right.y * wR_eff) / total_weight;
    result.z = ((float)pixel_left.z * wL_eff + (float)pixel_right.z * wR_eff) / total_weight;
    
    // Опциональное усиление яркости по краям (ОТКЛЮЧЕНО по умолчанию)
    if (enable_edge_boost) {
        float dist_x = fabsf(x - output_width * 0.5f) / (output_width * 0.5f);
        if (dist_x > 0.8f) {
            float brightness_boost = 1.0f + (dist_x - 0.8f) * 1.75f;
            brightness_boost = fminf(1.35f, brightness_boost);  // Ограничиваем до 1.35x
            result.x = fminf(255.0f, result.x * brightness_boost);
            result.y = fminf(255.0f, result.y * brightness_boost);
            result.z = fminf(255.0f, result.z * brightness_boost);
        }
    }
    
    // Записываем результат (с переворотом как у вас было)
    // Cast to size_t before arithmetic to prevent integer overflow
    size_t out_idx = ((size_t)output_height - 1 - (size_t)y) * output_pitch +
                     ((size_t)output_width - 1 - (size_t)x) * 4;
    
    *((uchar4*)(output + out_idx)) = make_uchar4(
        __float2uint_rn(result.x),
        __float2uint_rn(result.y),
        __float2uint_rn(result.z),
        255
    );
}

// ============================================================================
// БЕЗОПАСНАЯ ЗАГРУЗКА LUT С ВАЛИДАЦИЕЙ
// ============================================================================
extern "C" cudaError_t load_panorama_luts(
    const char* left_x_path,
    const char* left_y_path,
    const char* right_x_path,
    const char* right_y_path,
    const char* weight_left_path,
    const char* weight_right_path,
    float** lut_left_x_gpu,
    float** lut_left_y_gpu,
    float** lut_right_x_gpu,
    float** lut_right_y_gpu,
    float** weight_left_gpu,
    float** weight_right_gpu,
    int lut_width,
    int lut_height)
{
    size_t expected_size = lut_width * lut_height * sizeof(float);
    printf("Loading panorama LUT maps: %dx%d (%.2f MB each)\n", 
           lut_width, lut_height, expected_size / (1024.0f * 1024.0f));
    
    // Структура для отслеживания выделенной памяти
    struct GPUBuffer {
        float** ptr;
        bool allocated;
    };
    
    GPUBuffer buffers[] = {
        {lut_left_x_gpu, false},
        {lut_left_y_gpu, false},
        {lut_right_x_gpu, false},
        {lut_right_y_gpu, false},
        {weight_left_gpu, false},
        {weight_right_gpu, false}
    };
    
    const char* paths[] = {
        left_x_path, left_y_path,
        right_x_path, right_y_path,
        weight_left_path, weight_right_path
    };
    
    const char* names[] = {
        "left_x", "left_y", "right_x", "right_y", 
        "weight_left", "weight_right"
    };
    
    cudaError_t err = cudaSuccess;
    
    // Функция очистки при ошибке
    auto cleanup = [&buffers]() {
        for (int i = 0; i < 6; i++) {
            if (buffers[i].allocated && *(buffers[i].ptr)) {
                cudaFree(*(buffers[i].ptr));
                *(buffers[i].ptr) = nullptr;
            }
        }
    };
    
    // Выделяем память на GPU с проверкой
    for (int i = 0; i < 6; i++) {
        err = cudaMalloc(buffers[i].ptr, expected_size);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to allocate GPU memory for %s: %s\n", 
                   names[i], cudaGetErrorString(err));
            cleanup();
            return err;
        }
        buffers[i].allocated = true;
    }
    
    // Временный буфер для загрузки и валидации
    std::vector<float> temp_buffer(lut_width * lut_height);
    
    // Загружаем и проверяем каждый файл
    for (int file_idx = 0; file_idx < 6; file_idx++) {
        std::ifstream file(paths[file_idx], std::ios::binary | std::ios::ate);
        
        if (!file.is_open()) {
            printf("ERROR: Cannot open LUT file: %s\n", paths[file_idx]);
            cleanup();
            return cudaErrorInvalidValue;
        }
        
        // Проверяем размер файла
        size_t file_size = file.tellg();
        if (file_size != expected_size) {
            printf("ERROR: Invalid file size for %s: expected %zu, got %zu\n", 
                   names[file_idx], expected_size, file_size);
            file.close();
            cleanup();
            return cudaErrorInvalidValue;
        }
        
        // Читаем данные
        file.seekg(0);
        file.read(reinterpret_cast<char*>(temp_buffer.data()), expected_size);
        
        if (!file.good()) {
            printf("ERROR: Failed to read file %s\n", paths[file_idx]);
            file.close();
            cleanup();
            return cudaErrorInvalidValue;
        }
        file.close();
        
        // Валидация данных
        bool is_coordinate = (file_idx < 4);  // x,y координаты
        bool is_weight = (file_idx >= 4);     // веса
        
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        int invalid_count = 0;
        int nan_count = 0;
        
        for (size_t i = 0; i < temp_buffer.size(); i++) {
            float val = temp_buffer[i];
            
            // Проверка на NaN и Inf
            if (!std::isfinite(val)) {
                nan_count++;
                temp_buffer[i] = 0.0f;  // Заменяем на безопасное значение
                continue;
            }
            
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
            
            // Валидация диапазонов
            if (is_coordinate) {
                // Координаты должны быть в разумных пределах
                if (val < -1000.0f || val > 10000.0f) {
                    invalid_count++;
                    temp_buffer[i] = fmaxf(-1000.0f, fminf(10000.0f, val));
                }
            } else if (is_weight) {
                // Веса должны быть в [0, 1]
                if (val < 0.0f || val > 1.0f) {
                    invalid_count++;
                    temp_buffer[i] = fmaxf(0.0f, fminf(1.0f, val));
                }
            }
        }
        
        if (nan_count > 0) {
            printf("WARNING: Fixed %d NaN/Inf values in %s\n", nan_count, names[file_idx]);
        }
        if (invalid_count > 0) {
            printf("WARNING: Clamped %d out-of-range values in %s\n", 
                   invalid_count, names[file_idx]);
        }
        
        printf("  ✓ Loaded %s: range [%.3f, %.3f]\n", 
               names[file_idx], min_val, max_val);
        
        // Копируем на GPU
        err = cudaMemcpy(*(buffers[file_idx].ptr), temp_buffer.data(), 
                        expected_size, cudaMemcpyHostToDevice);
        
        if (err != cudaSuccess) {
            printf("ERROR: Failed to copy %s to GPU: %s\n", 
                   names[file_idx], cudaGetErrorString(err));
            cleanup();
            return err;
        }
    }
    
    printf("✅ All panorama LUT maps loaded and validated successfully\n");
    return cudaSuccess;
}

// ============================================================================
// ЗАПУСК ОСНОВНОГО KERNEL
// ============================================================================
// Новая версия с параметром edge_boost
extern "C" cudaError_t launch_panorama_kernel_fixed(
    const unsigned char* input_left,
    const unsigned char* input_right,
    unsigned char* output,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    const StitchKernelConfig* config,
    cudaStream_t stream,
    bool enable_edge_boost)
{
    if (!input_left || !input_right || !output || !config) {
        return cudaErrorInvalidValue;
    }
    
    dim3 block(NvdsStitchConfig::BLOCK_SIZE_X, NvdsStitchConfig::BLOCK_SIZE_Y);
    dim3 grid(
        (config->output_width + block.x - 1) / block.x,
        (config->output_height + block.y - 1) / block.y
    );
    
    panorama_lut_kernel<<<grid, block, 0, stream>>>(
        input_left, input_right, output,
        lut_left_x, lut_left_y,
        lut_right_x, lut_right_y,
        weight_left, weight_right,
        config->input_width,
        config->input_height,
        config->input_pitch,
        config->output_width,
        config->output_height,
        config->output_pitch,
        enable_edge_boost
    );
    
    return cudaGetLastError();
}

// Старая версия для совместимости
extern "C" cudaError_t launch_panorama_kernel(
    const unsigned char* input_left,
    const unsigned char* input_right,
    unsigned char* output,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    const StitchKernelConfig* config,
    cudaStream_t stream)
{
    // Вызываем с отключенным edge boost по умолчанию
    return launch_panorama_kernel_fixed(
        input_left, input_right, output,
        lut_left_x, lut_left_y,
        lut_right_x, lut_right_y,
        weight_left, weight_right,
        config, stream, false);
}

// ============================================================================
// ОСВОБОЖДЕНИЕ ПАМЯТИ
// ============================================================================
extern "C" void free_panorama_luts(
    float* lut_left_x,
    float* lut_left_y,
    float* lut_right_x,
    float* lut_right_y,
    float* weight_left,
    float* weight_right)
{
    if (lut_left_x) cudaFree(lut_left_x);
    if (lut_left_y) cudaFree(lut_left_y);
    if (lut_right_x) cudaFree(lut_right_x);
    if (lut_right_y) cudaFree(lut_right_y);
    if (weight_left) cudaFree(weight_left);
    if (weight_right) cudaFree(weight_right);
}

extern "C" void free_color_correction(ColorCorrectionContext* ctx) {
    if (ctx) {
        if (ctx->d_sum_left) cudaFree(ctx->d_sum_left);
        if (ctx->d_sum_right) cudaFree(ctx->d_sum_right);
        if (ctx->d_count_left) cudaFree(ctx->d_count_left);
        if (ctx->d_count_right) cudaFree(ctx->d_count_right);
        if (ctx->h_sum_left) cudaFreeHost(ctx->h_sum_left);
        if (ctx->h_sum_right) cudaFreeHost(ctx->h_sum_right);
        if (ctx->h_count_left) cudaFreeHost(ctx->h_count_left);
        if (ctx->h_count_right) cudaFreeHost(ctx->h_count_right);
        delete ctx;
    }
}
