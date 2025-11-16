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

} // extern "C"