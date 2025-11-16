// nvdsstitch_config.h - Конфигурация только для ПАНОРАМНОГО режима
#ifndef __NVDSSTITCH_CONFIG_H__
#define __NVDSSTITCH_CONFIG_H__

#ifdef __cplusplus
#include <string>
#endif

namespace NvdsStitchConfig {
    
    // ========== ПАРАМЕТРЫ ВХОДА ==========
    constexpr int INPUT_WIDTH = 3840;
    constexpr int INPUT_HEIGHT = 2160;
    constexpr int INPUT_FORMAT = 19; // NVBUF_COLOR_FORMAT_RGBA
    constexpr int LEFT_SOURCE_ID = 0;
    constexpr int RIGHT_SOURCE_ID = 1;
    
    // ========== ПАРАМЕТРЫ ПАНОРАМЫ ==========
    // УДАЛЕНО: OUTPUT_WIDTH, OUTPUT_HEIGHT - теперь передаются через properties!
    constexpr int OUTPUT_FORMAT = 19;    // NVBUF_COLOR_FORMAT_RGBA

    // ========== LUT КАРТЫ ==========
    // УДАЛЕНО: WARP_WIDTH, WARP_HEIGHT - теперь берутся из stitch->output_width/height!
    constexpr const char* WARP_MAPS_DIR = "warp_maps";

    // Файлы LUT и весов (6528x1800 - TRUE SPHERICAL from pano_cuda_debug_stages.py!)
    constexpr const char* WARP_LEFT_X_FILE = "lut_left_x.bin";
    constexpr const char* WARP_LEFT_Y_FILE = "lut_left_y.bin";
    constexpr const char* WARP_RIGHT_X_FILE = "lut_right_x.bin";
    constexpr const char* WARP_RIGHT_Y_FILE = "lut_right_y.bin";
    constexpr const char* WEIGHT_LEFT_FILE = "weight_left.bin";
    constexpr const char* WEIGHT_RIGHT_FILE = "weight_right.bin";
    
    // Обрезка не нужна в панорамном режиме
    constexpr int DEFAULT_CROP_TOP = 0;
    constexpr int DEFAULT_CROP_BOTTOM = 0;
    constexpr int DEFAULT_CROP_SIDES = 0;
    constexpr int OVERLAP = 0;  // В панораме overlap через веса
    
    // ========== CUDA ПАРАМЕТРЫ ==========
    constexpr int GPU_ID = 0;
    constexpr int BLOCK_SIZE_X = 32;
    constexpr int BLOCK_SIZE_Y = 8;
    
    // ========== BUFFER POOL ==========
    constexpr int POOL_MIN_BUFFERS = 8;
    constexpr int POOL_MAX_BUFFERS = 10;
    constexpr int POOL_MEMTYPE = 4; // NVBUF_MEM_SURFACE_ARRAY
    
#ifdef __cplusplus
    inline std::string getWarpPath(const char* filename) {
        return std::string(WARP_MAPS_DIR) + "/" + filename;
    }
    
    inline std::string getWarpLeftXPath() {
        return getWarpPath(WARP_LEFT_X_FILE);
    }
    
    inline std::string getWarpLeftYPath() {
        return getWarpPath(WARP_LEFT_Y_FILE);
    }
    
    inline std::string getWarpRightXPath() {
        return getWarpPath(WARP_RIGHT_X_FILE);
    }
    
    inline std::string getWarpRightYPath() {
        return getWarpPath(WARP_RIGHT_Y_FILE);
    }
    
    inline std::string getWeightLeftPath() {
        return getWarpPath(WEIGHT_LEFT_FILE);
    }
    
    inline std::string getWeightRightPath() {
        return getWarpPath(WEIGHT_RIGHT_FILE);
    }
#endif
    
    inline int calculatePitch(int width, int bytes_per_pixel = 4) {
        int min_pitch = width * bytes_per_pixel;
        return ((min_pitch + 31) / 32) * 32;
    }
    
    inline int getInputPitch() {
        return calculatePitch(INPUT_WIDTH);
    }

    // УДАЛЕНО: getOutputPitch() - теперь pitch вычисляется динамически!
}

#endif // __NVDSSTITCH_CONFIG_H__
