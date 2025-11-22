// cuda_tile_extractor.cu - ФИНАЛЬНАЯ ВЕРСИЯ
#include <cuda_runtime.h>
#include <stdio.h>

// Конфигурация плагина
#define TILES_PER_BATCH 6
#define TILE_WIDTH 1024
#define TILE_HEIGHT 1024
// PANORAMA_WIDTH, PANORAMA_HEIGHT - НЕ нужны! Размеры передаются через параметры функции

// Constant memory
__constant__ struct {
    int x;
    int y;
} d_tile_positions[TILES_PER_BATCH];

__constant__ void* d_tile_output_ptrs[TILES_PER_BATCH];

// ОПТИМИЗИРОВАННЫЙ KERNEL
__global__ void extract_tiles_kernel_multi(
    const unsigned char* __restrict__ src_panorama,
    int src_width,
    int src_height,
    int src_pitch,
    int tile_size,
    int tile_pitch)
{
    const int tile_id = blockIdx.z;
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tile_x >= tile_size || tile_y >= tile_size || tile_id >= TILES_PER_BATCH) {
        return;
    }
    
    const int src_x = d_tile_positions[tile_id].x + tile_x;
    const int src_y = d_tile_positions[tile_id].y + tile_y;
    
    unsigned char* dst_base = (unsigned char*)d_tile_output_ptrs[tile_id];
    const int dst_idx = tile_y * tile_pitch + tile_x * 4;
    
    // Проверка границ
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        // Черный цвет для out-of-bounds
        *((unsigned int*)(dst_base + dst_idx)) = 0xFF000000U;  // RGBA black
        return;
    }
    
    // Копируем пиксель
    const size_t src_idx = (size_t)src_y * src_pitch + (size_t)src_x * 4;
    *((unsigned int*)(dst_base + dst_idx)) = 
        *((const unsigned int*)(src_panorama + src_idx));
}

// HOST ФУНКЦИИ
extern "C" {

int cuda_init_tile_positions(int positions[][2])
{
    cudaError_t err = cudaMemcpyToSymbol(
        d_tile_positions,
        positions,
        sizeof(int) * 2 * TILES_PER_BATCH
    );
    
    if (err != cudaSuccess) {
        printf("ERROR: Failed to init tile positions: %s\n",
               cudaGetErrorString(err));
        return -1;
    }
    
    printf("✓ Tile positions initialized in GPU constant memory\n");
    return 0;
}

int cuda_set_tile_pointers(void** tile_ptrs)
{
    cudaError_t err = cudaMemcpyToSymbol(
        d_tile_output_ptrs,
        tile_ptrs,
        sizeof(void*) * TILES_PER_BATCH
    );
    
    if (err != cudaSuccess) {
        printf("ERROR: Failed to set tile pointers: %s\n",
               cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

extern "C" int cuda_extract_tiles(
    void* src_gpu,
    int src_width,    // ← 1-й параметр
    int src_height,   // ← 2-й параметр  
    int src_pitch,    // ← 3-й параметр
    int tile_pitch,   // ← 4-й параметр
    cudaStream_t stream
)
{
    // Валидация параметров (размеры передаются динамически через properties!)
    if (src_width <= 0 || src_height <= 0) {
        printf("ERROR: Invalid input dimensions: %dx%d\n", src_width, src_height);
        return -1;
    }

    if (tile_pitch <= 0 || src_pitch <= 0) {
        printf("ERROR: Invalid pitch values: src=%d, tile=%d\n",
               src_pitch, tile_pitch);
        return -1;
    }
    
    // Launch configuration
    dim3 block(32, 32, 1);
    dim3 grid(
        (TILE_WIDTH + block.x - 1) / block.x,
        (TILE_HEIGHT + block.y - 1) / block.y,
        TILES_PER_BATCH
    );
    
    // Запуск kernel
    extract_tiles_kernel_multi<<<grid, block, 0, stream>>>(
        (const unsigned char*)src_gpu,
        src_width,
        src_height,
        src_pitch,
        TILE_WIDTH,
        tile_pitch
    );
    
    // Проверка ошибок
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// NV12 Input Support - YUV→RGB Conversion
// ============================================================================

/**
 * @brief Extract tiles from NV12 panorama with YUV→RGB conversion
 *
 * Samples NV12 panorama (Y plane + UV plane), converts to RGB using BT.601,
 * and writes RGBA tiles for YOLO inference.
 *
 * NV12 Layout:
 * - Y plane: width × height bytes (full resolution luma)
 * - UV plane: (width × height) / 2 bytes (half resolution chroma, interleaved U,V)
 *
 * YUV→RGB Conversion (BT.601):
 * R = Y + 1.402 * (V - 128)
 * G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
 * B = Y + 1.772 * (U - 128)
 *
 * Performance: Expected ~3-4ms for 6 tiles @ 1024×1024 (Jetson Orin NX)
 */
__global__ void __launch_bounds__(1024, 1)
extract_tiles_kernel_nv12(
    const unsigned char* __restrict__ src_y,      // Y plane pointer
    const unsigned char* __restrict__ src_uv,     // UV plane pointer
    int src_width,
    int src_height,
    int src_pitch_y,    // Y plane pitch
    int src_pitch_uv,   // UV plane pitch
    int tile_size,
    int tile_pitch)
{
    const int tile_id = blockIdx.z;
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tile_size || tile_y >= tile_size || tile_id >= TILES_PER_BATCH) {
        return;
    }

    const int src_x = d_tile_positions[tile_id].x + tile_x;
    const int src_y_coord = d_tile_positions[tile_id].y + tile_y;

    unsigned char* dst_base = (unsigned char*)d_tile_output_ptrs[tile_id];
    const int dst_idx = tile_y * tile_pitch + tile_x * 4;

    // Check bounds
    if (src_x < 0 || src_x >= src_width || src_y_coord < 0 || src_y_coord >= src_height) {
        // Black for out-of-bounds
        dst_base[dst_idx + 0] = 0;    // R
        dst_base[dst_idx + 1] = 0;    // G
        dst_base[dst_idx + 2] = 0;    // B
        dst_base[dst_idx + 3] = 255;  // A
        return;
    }

    // Sample Y plane (full resolution)
    size_t y_idx = (size_t)src_y_coord * src_pitch_y + (size_t)src_x;
    unsigned char y_val = src_y[y_idx];

    // Sample UV plane (half resolution, 4:2:0 subsampling)
    size_t uv_idx = ((size_t)src_y_coord / 2) * src_pitch_uv + ((size_t)src_x / 2) * 2;
    unsigned char u_val = src_uv[uv_idx];
    unsigned char v_val = src_uv[uv_idx + 1];

    // YUV→RGB conversion (BT.601)
    // Normalize to [0, 255] range
    float y_f = (float)y_val;
    float u_f = (float)u_val - 128.0f;
    float v_f = (float)v_val - 128.0f;

    // BT.601 conversion coefficients
    float r = y_f + 1.402f * v_f;
    float g = y_f - 0.344136f * u_f - 0.714136f * v_f;
    float b = y_f + 1.772f * u_f;

    // Clamp to [0, 255]
    r = fminf(fmaxf(r, 0.0f), 255.0f);
    g = fminf(fmaxf(g, 0.0f), 255.0f);
    b = fminf(fmaxf(b, 0.0f), 255.0f);

    // Write RGBA (alpha=255)
    dst_base[dst_idx + 0] = (unsigned char)r;
    dst_base[dst_idx + 1] = (unsigned char)g;
    dst_base[dst_idx + 2] = (unsigned char)b;
    dst_base[dst_idx + 3] = 255;  // Alpha
}

/**
 * @brief Launch NV12 tile extraction kernel with YUV→RGB conversion
 *
 * Extracts 6 tiles from NV12 panorama, converting YUV to RGBA for inference.
 *
 * @param src_y_gpu Y plane GPU pointer
 * @param src_uv_gpu UV plane GPU pointer
 * @param src_width Panorama width
 * @param src_height Panorama height
 * @param src_pitch_y Y plane pitch
 * @param src_pitch_uv UV plane pitch
 * @param tile_pitch Output tile pitch
 * @param stream CUDA stream
 * @return 0 on success, -1 on error
 */
int cuda_extract_tiles_nv12(
    void* src_y_gpu,       // Y plane pointer
    void* src_uv_gpu,      // UV plane pointer
    int src_width,
    int src_height,
    int src_pitch_y,       // Y plane pitch
    int src_pitch_uv,      // UV plane pitch
    int tile_pitch,
    cudaStream_t stream)
{
    // Validate parameters
    if (!src_y_gpu || !src_uv_gpu) {
        printf("ERROR: NULL NV12 plane pointer (Y=%p UV=%p)\n", src_y_gpu, src_uv_gpu);
        return -1;
    }

    if (src_width <= 0 || src_height <= 0) {
        printf("ERROR: Invalid input dimensions: %dx%d\n", src_width, src_height);
        return -1;
    }

    if (src_pitch_y <= 0 || src_pitch_uv <= 0 || tile_pitch <= 0) {
        printf("ERROR: Invalid pitch values: Y=%d UV=%d tile=%d\n",
               src_pitch_y, src_pitch_uv, tile_pitch);
        return -1;
    }

    // Launch configuration
    dim3 block(32, 32, 1);
    dim3 grid(
        (TILE_WIDTH + block.x - 1) / block.x,
        (TILE_HEIGHT + block.y - 1) / block.y,
        TILES_PER_BATCH
    );

    // Launch NV12 kernel
    extract_tiles_kernel_nv12<<<grid, block, 0, stream>>>(
        (const unsigned char*)src_y_gpu,
        (const unsigned char*)src_uv_gpu,
        src_width,
        src_height,
        src_pitch_y,
        src_pitch_uv,
        TILE_WIDTH,
        tile_pitch
    );

    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: NV12 kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

} // extern "C"