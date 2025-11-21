// nvdsstitch_config.h - Configuration for PANORAMA mode stitching
#ifndef __NVDSSTITCH_CONFIG_H__
#define __NVDSSTITCH_CONFIG_H__

#ifdef __cplusplus
#include <string>
#endif

namespace NvdsStitchConfig {

    // ========== INPUT PARAMETERS ==========
    constexpr int INPUT_WIDTH = 3840;
    constexpr int INPUT_HEIGHT = 2160;
    constexpr int INPUT_FORMAT = 19; // NVBUF_COLOR_FORMAT_RGBA
    constexpr int LEFT_SOURCE_ID = 0;
    constexpr int RIGHT_SOURCE_ID = 1;

    // ========== PANORAMA PARAMETERS ==========
    // Output dimensions (width, height) configured via properties
    constexpr int OUTPUT_FORMAT = 19;    // NVBUF_COLOR_FORMAT_RGBA

    // ========== LUT MAPS ==========
    // LUT dimensions match output dimensions (from properties)
    constexpr const char* WARP_MAPS_DIR = "warp_maps";

    // LUT and weight files (e.g. 5700x1900 - TRUE SPHERICAL from pano_cuda_debug_stages.py!)
    constexpr const char* WARP_LEFT_X_FILE = "lut_left_x.bin";
    constexpr const char* WARP_LEFT_Y_FILE = "lut_left_y.bin";
    constexpr const char* WARP_RIGHT_X_FILE = "lut_right_x.bin";
    constexpr const char* WARP_RIGHT_Y_FILE = "lut_right_y.bin";
    constexpr const char* WEIGHT_LEFT_FILE = "weight_left.bin";
    constexpr const char* WEIGHT_RIGHT_FILE = "weight_right.bin";

    // Cropping parameters (zero for full panorama output)
    constexpr int DEFAULT_CROP_TOP = 0;
    constexpr int DEFAULT_CROP_BOTTOM = 0;
    constexpr int DEFAULT_CROP_SIDES = 0;
    constexpr int OVERLAP = 0;  // Overlap blending handled via weight maps

    // ========== COLOR CORRECTION (ASYNC) ==========
    // Hardware-sync-aware color correction for overlap region
    // Based on ISP config: Gamma 2.4, AE target 120, AWB soft clamp
    namespace ColorCorrectionConfig {
        // Analysis parameters (configurable via properties)
        constexpr float DEFAULT_OVERLAP_SIZE = 10.0f;        // Degrees (5-15 range)
        constexpr unsigned int DEFAULT_ANALYZE_INTERVAL = 30;  // Frames (0-120 range, 0=disable)
        constexpr float DEFAULT_SMOOTHING_FACTOR = 0.15f;    // 15% update per frame
        constexpr float DEFAULT_SPATIAL_FALLOFF = 2.0f;      // Vignetting compensation exponent
        constexpr bool DEFAULT_ENABLE_GAMMA = true;          // Enable gamma correction

        // Overlap analysis region (calculated from panorama width)
        constexpr float OVERLAP_CENTER_X = 0.50f;            // Center of panorama (50%)
        constexpr float OVERLAP_MIN_SAMPLES = 10000.0f;      // Minimum valid pixels for analysis

        // Gamma correction limits (based on ISP gamma 2.4, conservative range)
        constexpr float GAMMA_MIN = 0.8f;   // Darken by max 20% (underexposed camera)
        constexpr float GAMMA_MAX = 1.2f;   // Brighten by max 20% (overexposed camera)

        // RGB gain limits (AWB/CCM correction)
        constexpr float GAIN_MIN = 0.5f;    // Minimum color gain
        constexpr float GAIN_MAX = 2.0f;    // Maximum color gain
    }

    // ========== CUDA PARAMETERS ==========
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
}

// ========== COLOR CORRECTION FACTORS STRUCTURE ==========
// Used by both C++ (host) and CUDA (device) code
// 8 correction factors: RGB gains + gamma for left and right cameras
typedef struct {
    float left_r;      // Red gain for left camera (1.0 = no correction)
    float left_g;      // Green gain for left camera
    float left_b;      // Blue gain for left camera
    float left_gamma;  // Gamma correction for left camera (1.0 = no correction)
    float right_r;     // Red gain for right camera
    float right_g;     // Green gain for right camera
    float right_b;     // Blue gain for right camera
    float right_gamma; // Gamma correction for right camera
} ColorCorrectionFactors;

#endif // __NVDSSTITCH_CONFIG_H__
