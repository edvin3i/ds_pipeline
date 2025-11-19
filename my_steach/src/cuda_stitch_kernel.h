// cuda_stitch_kernel.h - Заголовочный файл только для панорамного режима
#ifndef __CUDA_STITCH_KERNEL_H__
#define __CUDA_STITCH_KERNEL_H__

#include <cuda_runtime.h>
#include "nvdsstitch_config.h"

typedef struct {
    int input_width;
    int input_height;
    int input_pitch;
    int output_width;
    int output_height;
    int output_pitch;
    int warp_width;   // Размер LUT
    int warp_height;  // Размер LUT
    int overlap;      // Не используется в панораме
    int crop_top;     // Не используется в панораме
    int crop_bottom;  // Не используется в панораме
    int crop_sides;   // Не используется в панораме
    int full_height;  // Не используется в панораме
    int full_width;   // Не используется в панораме
} StitchKernelConfig;

#ifdef __cplusplus
extern "C" {
#endif

// Загрузка LUT карт и весов для панорамы
cudaError_t load_panorama_luts(
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
    int lut_height
);

// Запуск панорамного kernel
cudaError_t launch_panorama_kernel(
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
    cudaStream_t stream
);

// Освобождение памяти
void free_panorama_luts(
    float* lut_left_x,
    float* lut_left_y,
    float* lut_right_x,
    float* lut_right_y,
    float* weight_left,
    float* weight_right
);

cudaError_t update_color_correction_simple(
    const unsigned char* left_frame,
    const unsigned char* right_frame,
    const float* weight_left,
    const float* weight_right,
    int width,
    int height,
    int pitch,
    cudaStream_t stream
);
cudaError_t init_color_correction(void);

// ========== ASYNC COLOR CORRECTION (Hardware-Sync-Aware) ==========

// Analyze color differences in overlap region (async, non-blocking)
cudaError_t analyze_color_correction_async(
    const unsigned char* left_ptr,
    const unsigned char* right_ptr,
    int left_pitch,
    int right_pitch,
    int pano_width,
    int pano_height,
    const float* lut_left_x,
    const float* lut_left_y,
    const float* lut_right_x,
    const float* lut_right_y,
    const float* weight_left,
    const float* weight_right,
    float overlap_center_x,
    float overlap_width,
    float spatial_falloff,
    float* output_buffer,      // Device buffer for 9 floats (reduction results)
    cudaStream_t stream
);

// Finalize color correction factors (CPU-side post-processing)
void finalize_color_correction_factors(
    const float* accumulated_sums,  // Input: 9 values from GPU
    ColorCorrectionFactors* output, // Output: 8 correction factors
    bool enable_gamma
);

// Update color correction factors in device constant memory
cudaError_t update_color_correction_factors(
    const ColorCorrectionFactors* factors
);

#ifdef __cplusplus
}
#endif

#endif // __CUDA_STITCH_KERNEL_H__