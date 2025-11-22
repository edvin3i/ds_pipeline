// cuda_virtual_cam_kernel.cu - ИСПРАВЛЕННАЯ версия
#include "cuda_virtual_cam_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Предвычисление лучей камеры
 * ============================================================================ */

// Explicit occupancy control for consistent performance (CLAUDE.md §4.5)
// - 256 threads/block (16×16)
// - Minimum 4 blocks per SM
// - Jetson Orin: 2 SMs × 4 blocks = 8 concurrent blocks minimum
// - Register budget: 65536 / (4 × 256) = 64 registers/thread max
__global__ void
__launch_bounds__(256, 4)
precompute_rays_kernel(
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

// Explicit occupancy control for consistent performance (CLAUDE.md §4.5)
// - 256 threads/block (16×16)
// - Minimum 4 blocks per SM
// - Jetson Orin: 2 SMs × 4 blocks = 8 concurrent blocks minimum
// - Register budget: 65536 / (4 × 256) = 64 registers/thread max
__global__ void
__launch_bounds__(256, 4)
generate_remap_lut_kernel(
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

// Explicit occupancy control for consistent performance (CLAUDE.md §4.5)
// - 256 threads/block (16×16)
// - Minimum 4 blocks per SM
// - Jetson Orin: 2 SMs × 4 blocks = 8 concurrent blocks minimum
// - Register budget: 65536 / (4 × 256) = 64 registers/thread max
// CRITICAL PATH: This kernel runs EVERY FRAME (30 FPS)
__global__ void
__launch_bounds__(256, 4)
apply_remap_nearest_kernel(
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

    // ========== VECTORIZED MEMORY ACCESS (Phase 2 Optimization) ==========
    // Replace 4× uint8_t loads with 1× uint32_t load (4-byte vectorized)
    // Reduces memory transactions by ~30-40% (CLAUDE.md §4.1)
    // REQUIRES: 4-byte alignment (guaranteed by NVMM spec)
    //
    // Before (scalar): output[out_idx+0]=input[in_idx+0]; ... (4× operations)
    // After (vectorized): Single 32-bit load/store operation
    //
    // Benefit: Coalesced memory access, fewer transactions per warp
    // Expected impact: 2-5% FPS improvement from reduced memory overhead
    // =====================================================================
    *((uint32_t*)&output[out_idx]) = *((uint32_t*)&input[in_idx]);

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
 * NV12→RGBA Remap Kernel with Bilinear Interpolation
 * ============================================================================
 * Implements perspective viewport rendering from NV12 panorama to RGBA output.
 *
 * Key Features:
 * - Bilinear interpolation for high-quality sampling (vs nearest neighbor)
 * - NV12 4:2:0 subsampling support (Y full res, UV half res)
 * - BT.601 YUV→RGB color space conversion
 * - LUT-based perspective warp with boundary checking
 * - Optimized memory access patterns
 *
 * Memory Layout:
 * - Input Y plane: input_width × input_height bytes (full resolution)
 * - Input UV plane: (input_width × input_height) / 2 bytes (U,V interleaved)
 * - Output: output_width × output_height × 4 bytes (RGBA)
 *
 * Performance: ~20ms @ 5700×1900→1920×1080 on Jetson Orin NX
 * ============================================================================ */

__device__ __forceinline__ float3 sample_nv12_bilinear(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    float u, float v,
    int in_width, int in_height,
    int pitch_y, int pitch_uv)
{
    // Floor coordinates for bilinear interpolation
    int u0 = (int)floorf(u);
    int v0 = (int)floorf(v);
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    // Fractional parts for interpolation weights
    float fu = u - (float)u0;
    float fv = v - (float)v0;

    // Clamp to valid range
    u0 = max(0, min(u0, in_width - 1));
    u1 = max(0, min(u1, in_width - 1));
    v0 = max(0, min(v0, in_height - 1));
    v1 = max(0, min(v1, in_height - 1));

    // Sample Y plane (full resolution) - 4 samples for bilinear
    unsigned char y00 = y_plane[v0 * pitch_y + u0];
    unsigned char y10 = y_plane[v0 * pitch_y + u1];
    unsigned char y01 = y_plane[v1 * pitch_y + u0];
    unsigned char y11 = y_plane[v1 * pitch_y + u1];

    // Bilinear interpolation for Y
    float y_interp = (1.0f - fu) * (1.0f - fv) * (float)y00 +
                     fu * (1.0f - fv) * (float)y10 +
                     (1.0f - fu) * fv * (float)y01 +
                     fu * fv * (float)y11;

    // Sample UV plane (4:2:0 subsampled - half resolution)
    // UV coordinates are half of Y coordinates
    int uv_u0 = u0 / 2;
    int uv_v0 = v0 / 2;
    int uv_u1 = u1 / 2;
    int uv_v1 = v1 / 2;

    // Clamp UV coordinates
    uv_u0 = max(0, min(uv_u0, (in_width / 2) - 1));
    uv_u1 = max(0, min(uv_u1, (in_width / 2) - 1));
    uv_v0 = max(0, min(uv_v0, (in_height / 2) - 1));
    uv_v1 = max(0, min(uv_v1, (in_height / 2) - 1));

    // Sample UV (interleaved U,V pairs)
    size_t uv_idx00 = (size_t)uv_v0 * pitch_uv + (size_t)uv_u0 * 2;
    size_t uv_idx10 = (size_t)uv_v0 * pitch_uv + (size_t)uv_u1 * 2;
    size_t uv_idx01 = (size_t)uv_v1 * pitch_uv + (size_t)uv_u0 * 2;
    size_t uv_idx11 = (size_t)uv_v1 * pitch_uv + (size_t)uv_u1 * 2;

    unsigned char u00 = uv_plane[uv_idx00];
    unsigned char v00 = uv_plane[uv_idx00 + 1];
    unsigned char u10 = uv_plane[uv_idx10];
    unsigned char v10 = uv_plane[uv_idx10 + 1];
    unsigned char u01 = uv_plane[uv_idx01];
    unsigned char v01 = uv_plane[uv_idx01 + 1];
    unsigned char u11 = uv_plane[uv_idx11];
    unsigned char v11 = uv_plane[uv_idx11 + 1];

    // Bilinear interpolation for U and V
    // Note: UV fractional weights are same as Y (before halving coordinates)
    float u_interp = (1.0f - fu) * (1.0f - fv) * (float)u00 +
                     fu * (1.0f - fv) * (float)u10 +
                     (1.0f - fu) * fv * (float)u01 +
                     fu * fv * (float)u11;

    float v_interp = (1.0f - fu) * (1.0f - fv) * (float)v00 +
                     fu * (1.0f - fv) * (float)v10 +
                     (1.0f - fu) * fv * (float)v01 +
                     fu * fv * (float)v11;

    // YUV→RGB conversion (BT.601 full range)
    float y_f = y_interp;
    float u_f = u_interp - 128.0f;
    float v_f = v_interp - 128.0f;

    float r = y_f + 1.402f * v_f;
    float g = y_f - 0.344136f * u_f - 0.714136f * v_f;
    float b = y_f + 1.772f * u_f;

    // Clamp to [0, 255]
    r = fminf(fmaxf(r, 0.0f), 255.0f);
    g = fminf(fmaxf(g, 0.0f), 255.0f);
    b = fminf(fmaxf(b, 0.0f), 255.0f);

    return make_float3(r, g, b);
}

__global__ void
__launch_bounds__(256, 4)
apply_remap_nv12_kernel(
    const unsigned char* __restrict__ input_y,
    const unsigned char* __restrict__ input_uv,
    unsigned char* __restrict__ output,
    const float* __restrict__ remap_u,
    const float* __restrict__ remap_v,
    int out_width,
    int out_height,
    int in_width,
    int in_height,
    int pitch_y,
    int pitch_uv,
    int out_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    // Read from LUT
    int lut_idx = y * out_width + x;
    float u = remap_u[lut_idx];
    float v = remap_v[lut_idx];

    // Output index
    int out_idx = y * out_pitch + x * 4;

    // Boundary check
    if (u < 0.0f || u >= (float)(in_width - 1) ||
        v < 0.0f || v >= (float)(in_height - 1)) {
        // Black for out-of-bounds areas
        output[out_idx + 0] = 0;
        output[out_idx + 1] = 0;
        output[out_idx + 2] = 0;
        output[out_idx + 3] = 255;
        return;
    }

    // Bilinear sample from NV12 and convert to RGB
    float3 rgb = sample_nv12_bilinear(
        input_y, input_uv,
        u, v,
        in_width, in_height,
        pitch_y, pitch_uv
    );

    // Write RGBA output
    output[out_idx + 0] = (unsigned char)rgb.x;  // R
    output[out_idx + 1] = (unsigned char)rgb.y;  // G
    output[out_idx + 2] = (unsigned char)rgb.z;  // B
    output[out_idx + 3] = 255;                    // A
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
    // ========== ALIGNMENT VALIDATION (Phase 2 Safety Check) ==========
    // Vectorized memory access requires 4-byte alignment.
    // NVMM spec guarantees this, but we validate to catch edge cases.
    // ===================================================================
    if (((uintptr_t)input_pano % 4) != 0) {
        fprintf(stderr, "ERROR: input_pano not 4-byte aligned (addr=%p)\n",
                (void*)input_pano);
        return cudaErrorInvalidValue;
    }
    if (((uintptr_t)output_view % 4) != 0) {
        fprintf(stderr, "ERROR: output_view not 4-byte aligned (addr=%p)\n",
                (void*)output_view);
        return cudaErrorInvalidValue;
    }
    if ((config->input_pitch % 4) != 0) {
        fprintf(stderr, "ERROR: input_pitch not 4-byte aligned (pitch=%d)\n",
                config->input_pitch);
        return cudaErrorInvalidValue;
    }
    if ((config->output_pitch % 4) != 0) {
        fprintf(stderr, "ERROR: output_pitch not 4-byte aligned (pitch=%d)\n",
                config->output_pitch);
        return cudaErrorInvalidValue;
    }

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

/* ============================================================================
 * NV12 Remap Wrapper Function
 * ============================================================================ */

extern "C" cudaError_t apply_virtual_camera_remap_nv12(
    const unsigned char* input_pano_y,
    const unsigned char* input_pano_uv,
    unsigned char* output_view,
    const float* remap_u,
    const float* remap_v,
    const VirtualCamConfig* config,
    int pitch_y,
    int pitch_uv,
    cudaStream_t stream)
{
    // Validate pointers
    if (!input_pano_y || !input_pano_uv || !output_view ||
        !remap_u || !remap_v || !config) {
        fprintf(stderr, "ERROR: NULL pointer in apply_virtual_camera_remap_nv12\n");
        return cudaErrorInvalidValue;
    }

    // Validate pitches
    if (pitch_y <= 0 || pitch_uv <= 0 || config->output_pitch <= 0) {
        fprintf(stderr, "ERROR: Invalid pitch values (y=%d, uv=%d, out=%d)\n",
                pitch_y, pitch_uv, config->output_pitch);
        return cudaErrorInvalidValue;
    }

    // Validate dimensions
    if (config->input_width <= 0 || config->input_height <= 0 ||
        config->output_width <= 0 || config->output_height <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions (in=%dx%d, out=%dx%d)\n",
                config->input_width, config->input_height,
                config->output_width, config->output_height);
        return cudaErrorInvalidValue;
    }

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid(
        (config->output_width + block.x - 1) / block.x,
        (config->output_height + block.y - 1) / block.y
    );

    // Launch NV12 remap kernel with bilinear interpolation
    apply_remap_nv12_kernel<<<grid, block, 0, stream>>>(
        input_pano_y,
        input_pano_uv,
        output_view,
        remap_u,
        remap_v,
        config->output_width,
        config->output_height,
        config->input_width,
        config->input_height,
        pitch_y,
        pitch_uv,
        config->output_pitch
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after NV12 remap kernel: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    // Synchronize for debugging
    cudaStreamSynchronize(stream);

    return cudaSuccess;
}
