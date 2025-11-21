// cuda_stitch_kernel.h - Header file for panorama stitching CUDA kernels
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
    int warp_width;   // LUT dimensions
    int warp_height;  // LUT dimensions
    int overlap;      // Not used in panorama mode
    int crop_top;     // Not used in panorama mode
    int crop_bottom;  // Not used in panorama mode
    int crop_sides;   // Not used in panorama mode
    int full_height;  // Not used in panorama mode
    int full_width;   // Not used in panorama mode
} StitchKernelConfig;

#ifdef __cplusplus
extern "C" {
#endif

// Load LUT maps and blending weights for panorama stitching
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

// Launch panorama stitching kernel
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

// Free GPU memory for LUT maps
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
// Returns: 0 on success, -1 on insufficient samples, -2 on invalid data (NaN/Inf)
int finalize_color_correction_factors(
    const float* accumulated_sums,  // Input: 9 values from GPU
    ColorCorrectionFactors* output, // Output: 8 correction factors
    bool enable_gamma
);

// Update color correction factors in device constant memory
cudaError_t update_color_correction_factors(
    const ColorCorrectionFactors* factors
);

// ========== ERROR HANDLING ==========
// Comprehensive CUDA error checking with custom error actions
// Usage: CUDA_CHECK_RETURN(cudaMalloc(...), return GST_FLOW_ERROR);
#define CUDA_CHECK_RETURN(call, error_action)                                \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%s)\n",                     \
                __FILE__, __LINE__,                                           \
                cudaGetErrorString(err), cudaGetErrorName(err));              \
        error_action;                                                         \
    }                                                                         \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // __CUDA_STITCH_KERNEL_H__