// cuda_virtual_cam_kernel.h - Заголовок для CUDA kernels
#ifndef __CUDA_VIRTUAL_CAM_KERNEL_H__
#define __CUDA_VIRTUAL_CAM_KERNEL_H__

#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif

// Структура параметров для kernel
typedef struct {
    int input_width;
    int input_height;
    int input_pitch;
    int output_width;
    int output_height;
    int output_pitch;
    float lon_min, lon_max;
    float lat_min, lat_max;
} VirtualCamConfig;

// Предвычисление лучей камеры
cudaError_t precompute_camera_rays(
    float* rays_gpu,
    int width, int height,
    float fov_deg,
    cudaStream_t stream
);

// Генерация LUT для remap
cudaError_t generate_remap_lut(
    const float* rays_gpu,
    float* remap_u_gpu,
    float* remap_v_gpu,
    float yaw_deg,
    float pitch_deg,
    float roll_deg,
    const VirtualCamConfig* config,
    cudaStream_t stream
);

// Применение remap с билинейной интерполяцией
cudaError_t apply_virtual_camera_remap(
    const unsigned char* input_pano,
    unsigned char* output_view,
    const float* remap_u,
    const float* remap_v,
    const VirtualCamConfig* config,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // __CUDA_VIRTUAL_CAM_KERNEL_H__