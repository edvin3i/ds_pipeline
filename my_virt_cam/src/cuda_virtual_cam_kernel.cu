// cuda_virtual_cam_kernel.cu - ИСПРАВЛЕННАЯ версия
#include "cuda_virtual_cam_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Предвычисление лучей камеры
 * ============================================================================ */

__global__ void precompute_rays_kernel(
    float* rays,
    int width,
    int height,
    float fov_rad)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float f = 0.5f * width / tanf(fov_rad / 2.0f);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    
    // Нормализованные координаты БЕЗ инверсии
    float nx = (x - cx) / f;
    float ny = (y - cy) / f;  // БЕЗ минуса!
    
    float len = sqrtf(nx*nx + ny*ny + 1.0f);
    
    int idx = y * width + x;
    rays[idx * 3 + 0] = nx / len;
    rays[idx * 3 + 1] = ny / len;
    rays[idx * 3 + 2] = 1.0f / len;
}

extern "C" cudaError_t precompute_camera_rays(
    float* rays_gpu,
    int width, int height,
    float fov_deg,
    cudaStream_t stream)
{
    if (!rays_gpu) {
        printf("ERROR: rays_gpu is NULL\n");
        return cudaErrorInvalidValue;
    }
    
    float fov_rad = fov_deg * M_PI / 180.0f;

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    
    
    precompute_rays_kernel<<<grid, block, 0, stream>>>(
        rays_gpu, width, height, fov_rad
    );
    
    return cudaGetLastError();
}

/* ============================================================================
 * Генерация LUT для remap
 * ============================================================================ */

__global__ void generate_remap_lut_kernel(
    const float* rays_cam,
    float* remap_u,
    float* remap_v,
    float yaw_rad,
    float pitch_rad,
    float roll_rad,
    int width,
    int height,
    float lon_min, float lon_max,
    float lat_min, float lat_max,
    int pano_width,
    int pano_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Получаем луч камеры
    float rx = rays_cam[idx * 3 + 0];
    float ry = rays_cam[idx * 3 + 1];
    float rz = rays_cam[idx * 3 + 2];
    
    // ========== ИСПРАВЛЕННЫЙ ПОРЯДОК ВРАЩЕНИЙ ==========
    // Правильный порядок: ROLL -> PITCH -> YAW
    // (как в Python версии через make_view_matrix)
    
    // 1. ROLL (вращение вокруг оси Z - взгляда камеры)
    // Это вращение в плоскости изображения
    float cos_roll = cosf(roll_rad);
    float sin_roll = sinf(roll_rad);
    
    float rx_roll = rx * cos_roll - ry * sin_roll;
    float ry_roll = rx * sin_roll + ry * cos_roll;
    float rz_roll = rz;
    
    // 2. PITCH (вращение вокруг оси X - горизонтальной)
    float cos_pitch = cosf(pitch_rad);
    float sin_pitch = sinf(pitch_rad);
    
    float rx_pitch = rx_roll;
    float ry_pitch = ry_roll * cos_pitch - rz_roll * sin_pitch;
    float rz_pitch = ry_roll * sin_pitch + rz_roll * cos_pitch;
    
    // 3. YAW (вращение вокруг оси Y - вертикальной)
    float cos_yaw = cosf(yaw_rad);
    float sin_yaw = sinf(yaw_rad);
    
    float final_x = rx_pitch * cos_yaw + rz_pitch * sin_yaw;
    float final_y = ry_pitch;
    float final_z = -rx_pitch * sin_yaw + rz_pitch * cos_yaw;
    
    // ========== ПРЕОБРАЗУЕМ В СФЕРИЧЕСКИЕ КООРДИНАТЫ ==========
    float lambda = atan2f(final_x, final_z);
    float y_clamped = fmaxf(-1.0f, fminf(1.0f, final_y));
    float phi = asinf(y_clamped);
    
    // ========== ПРЕОБРАЗУЕМ В КООРДИНАТЫ ПАНОРАМЫ ==========
    float u_norm = (lambda - lon_min) / (lon_max - lon_min);
    float v_norm = (phi - lat_min) / (lat_max - lat_min);
    
    // Преобразуем в пиксельные координаты
    float u = u_norm * (pano_width - 1);
    float v = v_norm * (pano_height - 1);
    
    // Сохраняем результат
    remap_u[idx] = u;
    remap_v[idx] = v;
}

extern "C" cudaError_t generate_remap_lut(
    const float* rays_gpu,
    float* remap_u_gpu,
    float* remap_v_gpu,
    float yaw_deg,
    float pitch_deg,
    float roll_deg,
    const VirtualCamConfig* config,
    cudaStream_t stream)
{
    if (!rays_gpu || !remap_u_gpu || !remap_v_gpu) {
        printf("ERROR: NULL pointers in generate_remap_lut\n");
        return cudaErrorInvalidValue;
    }
    
    float yaw_rad = yaw_deg * M_PI / 180.0f;
    float pitch_rad = pitch_deg * M_PI / 180.0f;
    float roll_rad = roll_deg * M_PI / 180.0f;
    
    float lon_min = config->lon_min * M_PI / 180.0f;
    float lon_max = config->lon_max * M_PI / 180.0f;
    float lat_min = config->lat_min * M_PI / 180.0f;
    float lat_max = config->lat_max * M_PI / 180.0f;

    dim3 block(16, 16);
    dim3 grid(
        (config->output_width + block.x - 1) / block.x,
        (config->output_height + block.y - 1) / block.y
    );
    

    
    generate_remap_lut_kernel<<<grid, block, 0, stream>>>(
        rays_gpu,
        remap_u_gpu,
        remap_v_gpu,
        yaw_rad, pitch_rad, roll_rad,
        config->output_width,
        config->output_height,
        lon_min, lon_max,
        lat_min, lat_max,
        config->input_width,
        config->input_height
    );
    
    return cudaGetLastError();
}

/* ============================================================================
 * БЕЗОПАСНАЯ версия remap - сначала проверим что все работает
 * ============================================================================ */

// Полный remap kernel с исправлениями
__global__ void apply_remap_nearest_kernel(
    const unsigned char* input,
    unsigned char* output,
    const float* remap_u,
    const float* remap_v,
    int out_width,
    int out_height,
    int in_width,
    int in_height,
    int in_pitch,
    int out_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_width || y >= out_height) return;
    
    // Читаем из LUT
    int lut_idx = y * out_width + x;
    float u = remap_u[lut_idx];
    float v = remap_v[lut_idx];
    
    // Выходной индекс
    int out_idx = y * out_pitch + x * 4;
    
    // Проверка границ
    if (u < 0 || u >= in_width - 1 || v < 0 || v >= in_height - 1) {
        // Черный для областей вне панорамы
        output[out_idx + 0] = 0;
        output[out_idx + 1] = 0;
        output[out_idx + 2] = 0;
        output[out_idx + 3] = 255;
        return;
    }
    
    // Nearest neighbor (без интерполяции для начала)
    int src_x = (int)(u + 0.5f);
    int src_y = (int)(v + 0.5f);
    
    // Clamp к границам
    src_x = min(max(src_x, 0), in_width - 1);
    src_y = min(max(src_y, 0), in_height - 1);
    
    // Входной индекс
    int in_idx = src_y * in_pitch + src_x * 4;
    
    // Копируем пиксель
    output[out_idx + 0] = input[in_idx + 0];
    output[out_idx + 1] = input[in_idx + 1];
    output[out_idx + 2] = input[in_idx + 2];
    output[out_idx + 3] = input[in_idx + 3];

    // Отладка для нескольких точек
    if ((x == out_width/2 && y == out_height/2) ||
        (x == 0 && y == 0) ||
        (x == out_width-1 && y == out_height-1)) {
        // printf("Remap[%d,%d]: u=%.1f,v=%.1f -> src=(%d,%d) -> color=(%d,%d,%d,%d)\n",
        //        x, y, u, v, src_x, src_y,
        //        output[out_idx + 0], output[out_idx + 1],
        //        output[out_idx + 2], output[out_idx + 3]);
    }
}

/* ============================================================================
 * Тестовый kernel - просто заполнение цветом
 * ============================================================================ */

__global__ void fill_test_color_kernel(
    unsigned char* output,
    int width,
    int height,
    int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // ВАЖНО: используем pitch, а не width*4!
    int pixel_idx = y * pitch + x * 4;
    
    // Проверка границ
    if (pixel_idx + 3 >= height * pitch) return;
    
    // Градиент
    output[pixel_idx + 0] = (x * 255) / width;   // R
    output[pixel_idx + 1] = (y * 255) / height;  // G
    output[pixel_idx + 2] = 128;                 // B
    output[pixel_idx + 3] = 255;                 // A

    if (x == 0 && y == 0) {
        printf("Test fill: %dx%d, pitch=%d\n", width, height, pitch);
    }
}

// Обновим основную функцию
extern "C" cudaError_t apply_virtual_camera_remap(
    const unsigned char* input_pano,
    unsigned char* output_view,
    const float* remap_u,
    const float* remap_v,
    const VirtualCamConfig* config,
    cudaStream_t stream)
{
    // printf("apply_virtual_camera_remap: starting full remap\n");

    dim3 block(16, 16);
    dim3 grid(
        (config->output_width + block.x - 1) / block.x,
        (config->output_height + block.y - 1) / block.y
    );
    
    // Используем полный remap
    apply_remap_nearest_kernel<<<grid, block, 0, stream>>>(
        input_pano,
        output_view,
        remap_u,
        remap_v,
        config->output_width,
        config->output_height,
        config->input_width,
        config->input_height,
        config->input_pitch,
        config->output_pitch
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel - virtual: %s\n", cudaGetErrorString(err));
    }

    // Синхронизация для отладки
    cudaStreamSynchronize(stream);

    return err;
}
