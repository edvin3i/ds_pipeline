// cuda_stitch_kernel.cu - CUDA Kernels for Panorama Stitching with Color Correction
#include "cuda_stitch_kernel.h"
#include "nvdsstitch_config.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <cfloat>  // For FLT_MAX

// ============================================================================
// CONSTANT MEMORY FOR COLOR CORRECTION (Phase 1.5)
// ============================================================================
// Async color correction with gamma (8 factors)
__constant__ ColorCorrectionFactors g_color_factors;

// Legacy color correction (6 gains, no gamma)
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
// COLOR CORRECTION CONTEXT STRUCTURE
// ============================================================================
struct ColorCorrectionContext {
    float* d_sum_left;      // RGB sums for left camera
    float* d_sum_right;     // RGB sums for right camera
    int* d_count_left;      // Pixel counter for left camera
    int* d_count_right;     // Pixel counter for right camera
    float prev_gains[6];    // Previous values for temporal smoothing
    bool initialized;
    // Pinned memory for async copy operations
    float* h_sum_left;
    float* h_sum_right;
    int* h_count_left;
    int* h_count_right;
};

// ============================================================================
// BILINEAR INTERPOLATION
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
// OPTIMIZED OVERLAP ZONE ANALYSIS KERNEL WITH SHARED MEMORY
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
    // Shared memory for block-level reduction
    extern __shared__ float shared_data[];
    float* block_sum_left = shared_data;  // 3 floats
    float* block_sum_right = &shared_data[3];  // 3 floats
    int* block_count_left = (int*)&shared_data[6];  // 1 int
    int* block_count_right = (int*)&shared_data[7];  // 1 int

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared memory
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

        // Analyze only overlap zone (where both weights are significant)
        const float overlap_threshold = 0.1f;

        if (w_l > overlap_threshold && w_r > overlap_threshold) {
            // Get coordinates from LUT
            float left_u = lut_left_x[lut_idx];
            float left_v = lut_left_y[lut_idx];
            float right_u = lut_right_x[lut_idx];
            float right_v = lut_right_y[lut_idx];

            // Validate coordinate bounds
            if (left_u >= 0 && left_u < input_width &&
                left_v >= 0 && left_v < input_height &&
                right_u >= 0 && right_u < input_width &&
                right_v >= 0 && right_v < input_height) {

                // Sample pixels via bilinear interpolation
                uchar4 pixel_l = bilinear_sample(input_left, left_u, left_v,
                                                input_width, input_height, input_pitch);
                uchar4 pixel_r = bilinear_sample(input_right, right_u, right_v,
                                                input_width, input_height, input_pitch);

                // Atomic add to shared memory
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

    // Final reduction: only first thread in block writes to global memory
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
// ADVANCED COLOR CORRECTION INITIALIZATION
// ============================================================================

/**
 * @brief Initialize advanced color correction context (persistent buffers)
 *
 * Allocates persistent GPU and pinned host memory for color correction analysis.
 * This is an advanced version with optimized memory management for continuous use.
 *
 * Allocated resources:
 * - GPU buffers: d_sum_left/right (3 floats), d_count_left/right (1 int)
 * - Pinned host memory: h_sum_left/right (3 floats), h_count_left/right (1 int)
 * - Total: ~64 bytes GPU + ~64 bytes pinned host
 *
 * @param[out] ctx_out Pointer to receive allocated context (caller must free with free_color_correction)
 *
 * @return cudaSuccess on success, CUDA error code on failure
 * @retval cudaSuccess Context allocated and initialized
 * @retval cudaErrorMemoryAllocation Failed to allocate GPU or pinned memory
 *
 * @note This is an internal function, use async color correction API instead
 * @note Caller must call free_color_correction() to release resources
 *
 * @see free_color_correction
 * @see update_color_correction_advanced
 */
extern "C" cudaError_t init_color_correction_advanced(ColorCorrectionContext** ctx_out) {
    // Allocate context
    ColorCorrectionContext* ctx = new ColorCorrectionContext();

    // Initialize fields
    ctx->d_sum_left = nullptr;
    ctx->d_sum_right = nullptr;
    ctx->d_count_left = nullptr;
    ctx->d_count_right = nullptr;
    ctx->h_sum_left = nullptr;
    ctx->h_sum_right = nullptr;
    ctx->h_count_left = nullptr;
    ctx->h_count_right = nullptr;

    // Allocate persistent GPU buffers (once!)
    cudaError_t err;
    err = cudaMalloc(&ctx->d_sum_left, 3 * sizeof(float));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_sum_right, 3 * sizeof(float));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_count_left, sizeof(int));
    if (err != cudaSuccess) goto error;
    
    err = cudaMalloc(&ctx->d_count_right, sizeof(int));
    if (err != cudaSuccess) goto error;

    // Allocate pinned memory for fast async copying
    err = cudaHostAlloc(&ctx->h_sum_left, 3 * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_sum_right, 3 * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_count_left, sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;
    
    err = cudaHostAlloc(&ctx->h_count_right, sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) goto error;

    // Initialize previous gains for temporal smoothing
    for (int i = 0; i < 6; i++) {
        ctx->prev_gains[i] = 1.0f;
    }
    ctx->initialized = false;

    // Initialize constant memory - declared before goto
    {
        float initial_gains[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        err = cudaMemcpyToSymbol(g_color_gains, initial_gains, 6 * sizeof(float));
        if (err != cudaSuccess) goto error;
    }
    
    *ctx_out = ctx;
    printf("✓ Advanced color correction initialized with persistent buffers\n");
    return cudaSuccess;

error:
    // Cleanup on error
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
// COLOR CORRECTION UPDATE (OPTIMIZED VERSION)
// ============================================================================

/**
 * @brief Update color correction using advanced analysis (internal function)
 *
 * Analyzes overlap zone between cameras and updates correction gains with
 * temporal smoothing. This is an optimized version with persistent buffers
 * to minimize memory allocation overhead.
 *
 * @param[in] left_frame Left camera frame (GPU memory, RGBA)
 * @param[in] right_frame Right camera frame (GPU memory, RGBA)
 * @param[in] lut_left_x Left camera X coordinate LUT
 * @param[in] lut_left_y Left camera Y coordinate LUT
 * @param[in] lut_right_x Right camera X coordinate LUT
 * @param[in] lut_right_y Right camera Y coordinate LUT
 * @param[in] weight_left Left camera blending weights
 * @param[in] weight_right Right camera blending weights
 * @param[in] input_width Input frame width
 * @param[in] input_height Input frame height
 * @param[in] input_pitch Input frame pitch/stride
 * @param[in] output_width Panorama width
 * @param[in] output_height Panorama height
 * @param[in] stream CUDA stream for execution
 * @param[in,out] ctx Color correction context (persistent buffers)
 * @param[in] smoothing_factor Temporal smoothing (0.0-1.0, e.g., 0.15 for 15% update)
 *
 * @return cudaSuccess on success, CUDA error code on failure
 * @retval cudaErrorInvalidValue ctx is NULL
 *
 * @note Internal function - use async color correction API instead
 * @see init_color_correction_advanced
 * @see analyze_color_correction_async
 */
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

    // Clear buffers asynchronously
    cudaMemsetAsync(ctx->d_sum_left, 0, 3 * sizeof(float), stream);
    cudaMemsetAsync(ctx->d_sum_right, 0, 3 * sizeof(float), stream);
    cudaMemsetAsync(ctx->d_count_left, 0, sizeof(int), stream);
    cudaMemsetAsync(ctx->d_count_right, 0, sizeof(int), stream);

    // Launch analysis kernel
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

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: analyze_overlap_zone_kernel failed: %s\n", 
               cudaGetErrorString(err));
        return err;
    }

    // Async copy results to pinned host memory
    cudaMemcpyAsync(ctx->h_sum_left, ctx->d_sum_left, 3 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_sum_right, ctx->d_sum_right, 3 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_count_left, ctx->d_count_left, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx->h_count_right, ctx->d_count_right, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    // IMPORTANT: Do NOT call cudaStreamSynchronize here!
    // Return immediately to allow async execution on low-priority stream

    return cudaSuccess;
}


// ============================================================================
// BACKWARD COMPATIBILITY: LEGACY SIMPLE VERSION (deprecated)
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

// Stub for legacy function (backward compatibility)
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
    // Return success for backward compatibility (no-op)
    return cudaSuccess;
}

// ============================================================================
// ASYNC COLOR CORRECTION (HARDWARE-SYNC-AWARE) - Phase 1.4
// ============================================================================

/**
 * Analyze color differences in overlap region for hardware-synchronized cameras.
 *
 * HARDWARE SYNC INSIGHT:
 * Cameras are frame-locked via XVS/XHS signals with ±1 pixel precision.
 * Overlap region contains IDENTICAL scene content at same moment.
 * Color differences are PURELY sensor response curves + lens characteristics.
 * No temporal alignment needed - focus on RGB gains + gamma correction.
 *
 * ALGORITHM:
 * 1. Each thread processes one pixel in overlap region
 * 2. Apply spatial weight to compensate vignetting: w = (1 - |x - center|/width)^falloff
 * 3. Extract RGB from both cameras at same panorama coordinate (using LUTs)
 * 4. Accumulate weighted sums in shared memory (tree reduction)
 * 5. Write 9 values to global buffer for CPU post-processing
 *
 * OUTPUT BUFFER (9 floats):
 * [0-2]: sum_L_R, sum_L_G, sum_L_B  - Left camera weighted RGB sums
 * [3]:   sum_L_luma                  - Left camera weighted luma sum
 * [4-6]: sum_R_R, sum_R_G, sum_R_B  - Right camera weighted RGB sums
 * [7]:   sum_R_luma                  - Right camera weighted luma sum
 * [8]:   total_weight                - Sum of spatial weights (for normalization)
 *
 * GAMMA CORRECTION FORMULA (ISP-aware):
 * Input is already gamma-encoded (ISP applies gamma 2.4).
 * Use simple power function in gamma space:
 *   L_corrected = L_original^(gamma_factor)
 * Conservative range: [0.8, 1.2] (±20% brightness adjustment)
 *
 * LAUNCH CONFIG: <<<(32, 16), (32, 32)>>> = 524,288 threads
 * Shared memory: 9 floats * 1024 threads = 36 KB per block
 */
__global__ void analyze_color_correction_kernel(
    const unsigned char* __restrict__ left_ptr,
    const unsigned char* __restrict__ right_ptr,
    int left_pitch,
    int right_pitch,
    int pano_width,
    int pano_height,
    const float* __restrict__ lut_left_x,
    const float* __restrict__ lut_left_y,
    const float* __restrict__ lut_right_x,
    const float* __restrict__ lut_right_y,
    const float* __restrict__ weight_left,
    const float* __restrict__ weight_right,
    float overlap_center_x,      // Overlap center (normalized, typically 0.5)
    float overlap_width,         // Overlap width (normalized, e.g., 10/360 for 10 degrees)
    float spatial_falloff,       // Vignetting compensation exponent (default 2.0)
    float* __restrict__ output_buffer  // Device buffer for 9 floats
)
{
    // Shared memory for block-level reduction (9 values per thread)
    __shared__ float shared_sums[9][32][32];  // [value_idx][y][x] - 36 KB

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Initialize shared memory to zero
    for (int i = 0; i < 9; i++) {
        shared_sums[i][ty][tx] = 0.0f;
    }
    __syncthreads();

    // Process pixel if in bounds
    if (x < pano_width && y < pano_height) {
        int lut_idx = y * pano_width + x;

        // Read weights to determine if pixel is in overlap
        float w_left = weight_left[lut_idx];
        float w_right = weight_right[lut_idx];

        // Overlap threshold: both weights must be significant
        const float overlap_threshold = 0.1f;

        if (w_left > overlap_threshold && w_right > overlap_threshold) {
            // Compute spatial weight for vignetting compensation
            // Pixels near center of overlap get higher weight
            float norm_x = (float)x / pano_width;
            float dist_from_center = fabsf(norm_x - overlap_center_x);

            // Only analyze pixels within overlap_width
            if (dist_from_center < overlap_width * 0.5f) {
                // Spatial weight: higher near center, falls off towards edges
                float spatial_weight = 1.0f - (dist_from_center / (overlap_width * 0.5f));
                spatial_weight = powf(spatial_weight, spatial_falloff);

                // Read LUT coordinates
                float left_u = lut_left_x[lut_idx];
                float left_v = lut_left_y[lut_idx];
                float right_u = lut_right_x[lut_idx];
                float right_v = lut_right_y[lut_idx];

                // Validate coordinates
                if (left_u >= 0 && left_u < 3840 && left_v >= 0 && left_v < 2160 &&
                    right_u >= 0 && right_u < 3840 && right_v >= 0 && right_v < 2160) {

                    // Sample both cameras
                    uchar4 pixel_left = bilinear_sample(left_ptr, left_u, left_v,
                                                        3840, 2160, left_pitch);
                    uchar4 pixel_right = bilinear_sample(right_ptr, right_u, right_v,
                                                         3840, 2160, right_pitch);

                    // Compute luma (Rec. 709 coefficients)
                    float luma_left = 0.2126f * pixel_left.x +
                                      0.7152f * pixel_left.y +
                                      0.0722f * pixel_left.z;
                    float luma_right = 0.2126f * pixel_right.x +
                                       0.7152f * pixel_right.y +
                                       0.0722f * pixel_right.z;

                    // Accumulate weighted sums in shared memory
                    shared_sums[0][ty][tx] += pixel_left.x * spatial_weight;    // L_R
                    shared_sums[1][ty][tx] += pixel_left.y * spatial_weight;    // L_G
                    shared_sums[2][ty][tx] += pixel_left.z * spatial_weight;    // L_B
                    shared_sums[3][ty][tx] += luma_left * spatial_weight;       // L_luma

                    shared_sums[4][ty][tx] += pixel_right.x * spatial_weight;   // R_R
                    shared_sums[5][ty][tx] += pixel_right.y * spatial_weight;   // R_G
                    shared_sums[6][ty][tx] += pixel_right.z * spatial_weight;   // R_B
                    shared_sums[7][ty][tx] += luma_right * spatial_weight;      // R_luma

                    shared_sums[8][ty][tx] += spatial_weight;                   // total_weight
                }
            }
        }
    }
    __syncthreads();

    // Tree reduction in shared memory (32x32 = 1024 threads → 1 value per block)
    // Reduce within block to minimize global atomics

    // Step 1: Reduce across x dimension (32 → 1)
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tx < stride) {
            for (int i = 0; i < 9; i++) {
                shared_sums[i][ty][tx] += shared_sums[i][ty][tx + stride];
            }
        }
        __syncthreads();
    }

    // Step 2: Reduce across y dimension (32 → 1)
    if (tx == 0) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (ty < stride) {
                for (int i = 0; i < 9; i++) {
                    shared_sums[i][ty][0] += shared_sums[i][ty + stride][0];
                }
            }
            __syncthreads();
        }

        // Final thread (0,0) writes block result to global memory
        if (ty == 0) {
            for (int i = 0; i < 9; i++) {
                atomicAdd(&output_buffer[i], shared_sums[i][0][0]);
            }
        }
    }
}

/**
 * Launch wrapper for async color analysis kernel.
 *
 * CRITICAL: This function is NON-BLOCKING. It launches kernel on provided stream
 * and returns immediately. Caller must check completion with cudaEventQuery().
 */
extern "C" cudaError_t analyze_color_correction_async(
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
    float* output_buffer,
    cudaStream_t stream
)
{
    // Validate inputs
    if (!left_ptr || !right_ptr || !output_buffer) {
        return cudaErrorInvalidValue;
    }

    // Zero output buffer before analysis
    cudaMemsetAsync(output_buffer, 0, 9 * sizeof(float), stream);

    // Launch configuration: 32x32 threads per block
    dim3 block(32, 32);  // 1024 threads/block
    dim3 grid((pano_width + 31) / 32, (pano_height + 31) / 32);

    // Shared memory: 9 values × 32×32 threads = 9216 floats = 36 KB
    // Jetson Orin has 128 KB shared memory per SM, so this is safe

    analyze_color_correction_kernel<<<grid, block, 0, stream>>>(
        left_ptr, right_ptr,
        left_pitch, right_pitch,
        pano_width, pano_height,
        lut_left_x, lut_left_y,
        lut_right_x, lut_right_y,
        weight_left, weight_right,
        overlap_center_x, overlap_width, spatial_falloff,
        output_buffer
    );

    // Return launch error (does NOT synchronize)
    return cudaGetLastError();
}

/**
 * Finalize color correction factors on CPU after GPU analysis completes.
 *
 * INPUT: accumulated_sums[9] from GPU:
 *   [0-2]: sum_L_R, sum_L_G, sum_L_B
 *   [3]:   sum_L_luma
 *   [4-6]: sum_R_R, sum_R_G, sum_R_B
 *   [7]:   sum_R_luma
 *   [8]:   total_weight
 *
 * OUTPUT: ColorCorrectionFactors (8 values):
 *   left_r, left_g, left_b, left_gamma
 *   right_r, right_g, right_b, right_gamma
 *
 * ALGORITHM:
 * 1. Validate sufficient samples (total_weight > MIN_SAMPLES)
 * 2. Compute weighted means: mean_L = sum_L / total_weight
 * 3. Compute RGB gain ratios to balance colors
 * 4. Compute gamma correction from luma ratios (if enabled)
 * 5. Clamp to safe ranges defined in config
 */
extern "C" int finalize_color_correction_factors(
    const float* accumulated_sums,
    ColorCorrectionFactors* output,
    bool enable_gamma
)
{
    // Extract values from GPU results
    float sum_L_R = accumulated_sums[0];
    float sum_L_G = accumulated_sums[1];
    float sum_L_B = accumulated_sums[2];
    float sum_L_luma = accumulated_sums[3];

    float sum_R_R = accumulated_sums[4];
    float sum_R_G = accumulated_sums[5];
    float sum_R_B = accumulated_sums[6];
    float sum_R_luma = accumulated_sums[7];

    float total_weight = accumulated_sums[8];

    // Validate sufficient samples
    const float min_samples = NvdsStitchConfig::ColorCorrectionConfig::OVERLAP_MIN_SAMPLES;
    if (total_weight < min_samples) {
        // Insufficient data - return identity correction (no change)
        output->left_r = 1.0f;
        output->left_g = 1.0f;
        output->left_b = 1.0f;
        output->left_gamma = 1.0f;
        output->right_r = 1.0f;
        output->right_g = 1.0f;
        output->right_b = 1.0f;
        output->right_gamma = 1.0f;

        printf("WARNING: Insufficient samples for color correction (%.0f < %.0f), using identity\n",
               total_weight, min_samples);
        return -1;  // Error code: insufficient samples
    }

    // Compute weighted means
    float mean_L_R = sum_L_R / total_weight;
    float mean_L_G = sum_L_G / total_weight;
    float mean_L_B = sum_L_B / total_weight;
    float mean_L_luma = sum_L_luma / total_weight;

    float mean_R_R = sum_R_R / total_weight;
    float mean_R_G = sum_R_G / total_weight;
    float mean_R_B = sum_R_B / total_weight;
    float mean_R_luma = sum_R_luma / total_weight;

    // Compute target (average of both cameras)
    float target_R = (mean_L_R + mean_R_R) * 0.5f;
    float target_G = (mean_L_G + mean_R_G) * 0.5f;
    float target_B = (mean_L_B + mean_R_B) * 0.5f;

    // Compute RGB gain factors to reach target
    // gain_left = target / mean_left (multiply left values by this to reach target)
    const float eps = 1e-6f;  // Prevent division by zero

    float gain_L_R = target_R / (mean_L_R + eps);
    float gain_L_G = target_G / (mean_L_G + eps);
    float gain_L_B = target_B / (mean_L_B + eps);

    float gain_R_R = target_R / (mean_R_R + eps);
    float gain_R_G = target_G / (mean_R_G + eps);
    float gain_R_B = target_B / (mean_R_B + eps);

    // Validate for NaN/Inf before clamping (indicates invalid sensor data or math errors)
    if (!isfinite(gain_L_R) || !isfinite(gain_L_G) || !isfinite(gain_L_B) ||
        !isfinite(gain_R_R) || !isfinite(gain_R_G) || !isfinite(gain_R_B)) {
        fprintf(stderr, "ERROR: Color correction factors contain NaN/Inf - invalid data\n");
        fprintf(stderr, "  Left gains:  R=%.3f G=%.3f B=%.3f\n", gain_L_R, gain_L_G, gain_L_B);
        fprintf(stderr, "  Right gains: R=%.3f G=%.3f B=%.3f\n", gain_R_R, gain_R_G, gain_R_B);
        fprintf(stderr, "  Accumulated sums: L[%.1f,%.1f,%.1f] R[%.1f,%.1f,%.1f] weight=%.0f\n",
                sum_L_R, sum_L_G, sum_L_B, sum_R_R, sum_R_G, sum_R_B, total_weight);

        // Return identity factors
        output->left_r = 1.0f;
        output->left_g = 1.0f;
        output->left_b = 1.0f;
        output->left_gamma = 1.0f;
        output->right_r = 1.0f;
        output->right_g = 1.0f;
        output->right_b = 1.0f;
        output->right_gamma = 1.0f;
        return -2;  // Error code: invalid data (NaN/Inf)
    }

    // Clamp RGB gains to safe ranges (prevent extreme corrections)
    const float gain_min = NvdsStitchConfig::ColorCorrectionConfig::GAIN_MIN;
    const float gain_max = NvdsStitchConfig::ColorCorrectionConfig::GAIN_MAX;

    output->left_r = fmaxf(gain_min, fminf(gain_max, gain_L_R));
    output->left_g = fmaxf(gain_min, fminf(gain_max, gain_L_G));
    output->left_b = fmaxf(gain_min, fminf(gain_max, gain_L_B));

    output->right_r = fmaxf(gain_min, fminf(gain_max, gain_R_R));
    output->right_g = fmaxf(gain_min, fminf(gain_max, gain_R_G));
    output->right_b = fmaxf(gain_min, fminf(gain_max, gain_R_B));

    // Compute gamma correction from luma ratios (if enabled)
    if (enable_gamma) {
        // Gamma factor: gamma_left = log(target_luma / mean_left_luma) / log(mean_left_luma / 255)
        // Simplified: Use ratio of lumas to estimate gamma adjustment
        // If left is darker than right, gamma_left > 1.0 (brighten)

        float target_luma = (mean_L_luma + mean_R_luma) * 0.5f;

        // Simple gamma estimation: ratio of target to current luma
        // This is approximate but works well for small adjustments
        float gamma_L = target_luma / (mean_L_luma + eps);
        float gamma_R = target_luma / (mean_R_luma + eps);

        // Clamp gamma to conservative range (ISP already applies gamma 2.4)
        const float gamma_min = NvdsStitchConfig::ColorCorrectionConfig::GAMMA_MIN;
        const float gamma_max = NvdsStitchConfig::ColorCorrectionConfig::GAMMA_MAX;

        output->left_gamma = fmaxf(gamma_min, fminf(gamma_max, gamma_L));
        output->right_gamma = fmaxf(gamma_min, fminf(gamma_max, gamma_R));
    } else {
        // Gamma correction disabled - use identity (1.0 = no change)
        output->left_gamma = 1.0f;
        output->right_gamma = 1.0f;
    }

    // Debug output
    printf("Color correction factors computed (%.0f samples):\n", total_weight);
    printf("  Left:  R=%.3f G=%.3f B=%.3f γ=%.3f\n",
           output->left_r, output->left_g, output->left_b, output->left_gamma);
    printf("  Right: R=%.3f G=%.3f B=%.3f γ=%.3f\n",
           output->right_r, output->right_g, output->right_b, output->right_gamma);

    return 0;  // Success
}

/**
 * Update color correction factors in device constant memory.
 *
 * This function uploads the computed correction factors to GPU constant memory,
 * making them available to all subsequent kernel launches without per-kernel args.
 *
 * PERFORMANCE: Constant memory is cached and broadcast to all threads in a warp,
 * making it ideal for read-only data accessed by all threads.
 */
extern "C" cudaError_t update_color_correction_factors(
    const ColorCorrectionFactors* factors
)
{
    if (!factors) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaMemcpyToSymbol(
        g_color_factors,
        factors,
        sizeof(ColorCorrectionFactors)
    );

    if (err != cudaSuccess) {
        printf("ERROR: Failed to update color correction factors: %s\n",
               cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

/**
 * Apply color correction with gamma adjustment to a pixel.
 *
 * ALGORITHM:
 * 1. Apply RGB gains to balance color channels
 * 2. Apply gamma correction for brightness (simple power function)
 * 3. Clamp to valid range [0, 255]
 *
 * GAMMA FORMULA (ISP-aware):
 * Input is already gamma-encoded (ISP gamma 2.4).
 * Correction: pixel_out = pixel_in^gamma_factor
 * - gamma < 1.0: Darken (compress highlights)
 * - gamma > 1.0: Brighten (lift shadows)
 * - gamma = 1.0: No change
 *
 * @param pixel Input pixel (RGBA, but we only modify RGB)
 * @param is_left True for left camera factors, false for right
 * @return Corrected pixel
 */
__device__ inline uchar4 apply_color_correction_gamma(
    uchar4 pixel,
    bool is_left
)
{
    // Select factors based on camera
    float gain_r = is_left ? g_color_factors.left_r : g_color_factors.right_r;
    float gain_g = is_left ? g_color_factors.left_g : g_color_factors.right_g;
    float gain_b = is_left ? g_color_factors.left_b : g_color_factors.right_b;
    float gamma = is_left ? g_color_factors.left_gamma : g_color_factors.right_gamma;

    // Step 1: Apply RGB gains
    float r = (float)pixel.x * gain_r;
    float g = (float)pixel.y * gain_g;
    float b = (float)pixel.z * gain_b;

    // Step 2: Apply gamma correction (if not identity)
    if (gamma != 1.0f) {
        // Normalize to [0, 1] for gamma operation
        r = r / 255.0f;
        g = g / 255.0f;
        b = b / 255.0f;

        // Apply gamma: out = in^gamma
        r = powf(r, gamma);
        g = powf(g, gamma);
        b = powf(b, gamma);

        // Denormalize back to [0, 255]
        r = r * 255.0f;
        g = g * 255.0f;
        b = b * 255.0f;
    }

    // Step 3: Clamp to valid range and convert back to uchar4
    return make_uchar4(
        (unsigned char)fminf(255.0f, fmaxf(0.0f, r)),
        (unsigned char)fminf(255.0f, fmaxf(0.0f, g)),
        (unsigned char)fminf(255.0f, fmaxf(0.0f, b)),
        pixel.w  // Alpha unchanged
    );
}

// ============================================================================
// MAIN PANORAMA STITCHING KERNEL (WITH FIXES)
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

    // Read coordinates from LUT
    float left_u = lut_left_x[lut_idx];
    float left_v = lut_left_y[lut_idx];
    float right_u = lut_right_x[lut_idx];
    float right_v = lut_right_y[lut_idx];

    // Read blending weights
    float w_left = weight_left[lut_idx];
    float w_right = weight_right[lut_idx];

    uchar4 pixel_left = make_uchar4(0,0,0,0);
    uchar4 pixel_right = make_uchar4(0,0,0,0);

    // IMPORTANT FIX: Effective weights (only set if pixel successfully sampled)
    float wL_eff = 0.0f;  // Effective weight for left camera
    float wR_eff = 0.0f;  // Effective weight for right camera

    // Sample left camera ONLY if coordinates are valid
    if (w_left > 0.001f && left_u >= 0 && left_u < input_width &&
        left_v >= 0 && left_v < input_height) {

        pixel_left = bilinear_sample(input_left, left_u, left_v,
                                     input_width, input_height, input_pitch);

        // Apply NEW color correction with gamma (Phase 1.5)
        pixel_left = apply_color_correction_gamma(pixel_left, true);

        // Set effective weight ONLY if pixel was successfully sampled
        wL_eff = w_left;
    }

    // Sample right camera ONLY if coordinates are valid
    if (w_right > 0.001f && right_u >= 0 && right_u < input_width &&
        right_v >= 0 && right_v < input_height) {

        pixel_right = bilinear_sample(input_right, right_u, right_v,
                                      input_width, input_height, input_pitch);

        // Apply NEW color correction with gamma (Phase 1.5)
        pixel_right = apply_color_correction_gamma(pixel_right, false);

        // Set effective weight ONLY if pixel was successfully sampled
        wR_eff = w_right;
    }

    // FIXED: Use EFFECTIVE weights for blending
    float4 result;
    const float eps = 1e-6f;
    float total_weight = wL_eff + wR_eff + eps;  // Effective weights prevent invalid pixel contribution

    // Blend using effective weights
    result.x = ((float)pixel_left.x * wL_eff + (float)pixel_right.x * wR_eff) / total_weight;
    result.y = ((float)pixel_left.y * wL_eff + (float)pixel_right.y * wR_eff) / total_weight;
    result.z = ((float)pixel_left.z * wL_eff + (float)pixel_right.z * wR_eff) / total_weight;

    // Optional edge brightness boost (DISABLED by default)
    if (enable_edge_boost) {
        float dist_x = fabsf(x - output_width * 0.5f) / (output_width * 0.5f);
        if (dist_x > 0.8f) {
            float brightness_boost = 1.0f + (dist_x - 0.8f) * 1.75f;
            brightness_boost = fminf(1.35f, brightness_boost);  // Cap at 1.35x
            result.x = fminf(255.0f, result.x * brightness_boost);
            result.y = fminf(255.0f, result.y * brightness_boost);
            result.z = fminf(255.0f, result.z * brightness_boost);
        }
    }

    // Write result (with flip as originally implemented)
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
// SAFE LUT LOADING WITH VALIDATION
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

    // Structure for tracking allocated GPU memory
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

    // Cleanup function for error handling
    auto cleanup = [&buffers]() {
        for (int i = 0; i < 6; i++) {
            if (buffers[i].allocated && *(buffers[i].ptr)) {
                cudaFree(*(buffers[i].ptr));
                *(buffers[i].ptr) = nullptr;
            }
        }
    };

    // Allocate GPU memory with validation
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

    // Temporary buffer for loading and validation
    std::vector<float> temp_buffer(lut_width * lut_height);

    // Load and validate each file
    for (int file_idx = 0; file_idx < 6; file_idx++) {
        std::ifstream file(paths[file_idx], std::ios::binary | std::ios::ate);
        
        if (!file.is_open()) {
            printf("ERROR: Cannot open LUT file: %s\n", paths[file_idx]);
            cleanup();
            return cudaErrorInvalidValue;
        }

        // Validate file size
        size_t file_size = file.tellg();
        if (file_size != expected_size) {
            printf("ERROR: Invalid file size for %s: expected %zu, got %zu\n", 
                   names[file_idx], expected_size, file_size);
            file.close();
            cleanup();
            return cudaErrorInvalidValue;
        }

        // Read data
        file.seekg(0);
        file.read(reinterpret_cast<char*>(temp_buffer.data()), expected_size);
        
        if (!file.good()) {
            printf("ERROR: Failed to read file %s\n", paths[file_idx]);
            file.close();
            cleanup();
            return cudaErrorInvalidValue;
        }
        file.close();

        // Data validation
        bool is_coordinate = (file_idx < 4);  // x,y coordinates
        bool is_weight = (file_idx >= 4);     // blending weights

        float min_val = FLT_MAX, max_val = -FLT_MAX;
        int invalid_count = 0;
        int nan_count = 0;

        for (size_t i = 0; i < temp_buffer.size(); i++) {
            float val = temp_buffer[i];

            // Check for NaN and Inf
            if (!std::isfinite(val)) {
                nan_count++;
                temp_buffer[i] = 0.0f;  // Replace with safe value
                continue;
            }

            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);

            // Validate ranges
            if (is_coordinate) {
                // Coordinates should be within reasonable bounds
                if (val < -1000.0f || val > 10000.0f) {
                    invalid_count++;
                    temp_buffer[i] = fmaxf(-1000.0f, fminf(10000.0f, val));
                }
            } else if (is_weight) {
                // Weights must be in [0, 1]
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

        // Copy to GPU
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
// KERNEL LAUNCH FUNCTIONS
// ============================================================================

/**
 * @brief Launch panorama stitching kernel with edge boost option (internal)
 *
 * Extended version of launch_panorama_kernel with optional edge brightness boost
 * for vignetting compensation. This is the internal implementation called by both
 * the public API and legacy wrapper.
 *
 * @param[in] input_left Left camera frame (GPU memory, RGBA)
 * @param[in] input_right Right camera frame (GPU memory, RGBA)
 * @param[out] output Stitched panorama output (GPU memory, RGBA)
 * @param[in] lut_left_x Left camera X coordinate LUT
 * @param[in] lut_left_y Left camera Y coordinate LUT
 * @param[in] lut_right_x Right camera X coordinate LUT
 * @param[in] lut_right_y Right camera Y coordinate LUT
 * @param[in] weight_left Left camera blending weights
 * @param[in] weight_right Right camera blending weights
 * @param[in] config Kernel configuration (dimensions, pitch)
 * @param[in] stream CUDA stream for async execution
 * @param[in] enable_edge_boost Enable edge brightness boost (typically false)
 *
 * @return cudaSuccess on kernel launch success, error code on failure
 * @retval cudaSuccess Kernel launched successfully
 * @retval cudaErrorInvalidValue NULL pointer in required parameters
 * @retval cudaErrorLaunchFailure Kernel launch failed
 *
 * @note Internal function - use launch_panorama_kernel() for public API
 * @note ASYNC function - does not wait for kernel completion
 *
 * @see launch_panorama_kernel
 */
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

// Legacy version for backward compatibility
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
    // Call with edge boost disabled by default
    return launch_panorama_kernel_fixed(
        input_left, input_right, output,
        lut_left_x, lut_left_y,
        lut_right_x, lut_right_y,
        weight_left, weight_right,
        config, stream, false);
}

// ============================================================================
// MEMORY CLEANUP
// ============================================================================

/**
 * @brief Free GPU memory allocated for panorama LUT maps
 *
 * Releases all 6 LUT arrays allocated by load_panorama_luts().
 * Safe to call with NULL pointers (no-op for unallocated maps).
 *
 * @param[in] lut_left_x Left camera X coordinate LUT (may be NULL)
 * @param[in] lut_left_y Left camera Y coordinate LUT (may be NULL)
 * @param[in] lut_right_x Right camera X coordinate LUT (may be NULL)
 * @param[in] lut_right_y Right camera Y coordinate LUT (may be NULL)
 * @param[in] weight_left Left camera blending weights (may be NULL)
 * @param[in] weight_right Right camera blending weights (may be NULL)
 *
 * @note Always call before plugin destruction to prevent memory leaks
 * @note Function is synchronous (waits for GPU operations)
 *
 * @see load_panorama_luts
 */
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

/**
 * @brief Free color correction context and all associated resources
 *
 * Releases GPU buffers, pinned host memory, and context structure allocated by
 * init_color_correction_advanced(). Safe to call with NULL context.
 *
 * @param[in] ctx Color correction context to free (may be NULL)
 *
 * @note Always call to prevent memory leaks when using advanced color correction
 * @note Function is synchronous (waits for GPU operations)
 *
 * @see init_color_correction_advanced
 */
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
