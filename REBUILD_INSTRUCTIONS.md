# Plugin Rebuild Instructions for Jetson

## Issue Summary

**TWO CRITICAL BUGS FIXED:**

### Bug 1: Plugin Not Rebuilt (FIXED - Nov 21)
The nvtilebatcher plugin was compiled on Nov 21, but Phase 3 NV12 changes were committed on Nov 22 (commit c891fbf). The old binary only accepted RGBA.

### Bug 2: transform_caps Hardcoded to RGBA (FIXED - Nov 23, commit b7fce6d)
Even after rebuilding, caps negotiation failed with:
```
link between frame-filter:src and tilebatcher:sink failed: no common format
```

**Root Cause**: The `gst_nvtilebatcher_transform_caps()` function was hardcoded to return RGBA-only caps, overriding the static pad template that advertised both formats.

**Code Before (WRONG)**:
```cpp
if (direction == GST_PAD_SINK) {
    result = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGBA",  // ❌ Hardcoded!
        ...);
}
```

**Code After (FIXED)**:
```cpp
if (direction == GST_PAD_SINK) {
    result = gst_caps_from_string(
        "video/x-raw(memory:NVMM), format=(string){ RGBA, NV12 }, "  // ✅ Both formats
        "width=(int)[ 1, 2147483647 ], height=(int)[ 1, 2147483647 ]");
}
```

**Impact**: Zero buffers reached nvtilebatcher → no inference → no ball detection → virtualcam didn't follow ball.

---

## Rebuild Steps (Run on Jetson)

### 1. Navigate to Plugin Directory
```bash
cd ~/ds_pipeline/my_tile_batcher/src
```

### 2. Clean Old Build
```bash
make clean
```

### 3. Rebuild Plugin
```bash
make -j$(nproc)
```

Expected output:
```
🔨 Compiling gstnvtilebatcher.cpp...
🔨 Compiling gstnvtilebatcher_allocator.cpp...
🚀 Compiling CUDA cuda_tile_extractor.cu...
🔗 Linking libnvtilebatcher.so...
✓ Build complete!
```

### 4. Install Plugin
```bash
make install
```

Expected output:
```
📦 Installing plugin...
✓ Plugin installed to ~/.local/share/gstreamer-1.0/plugins/
```

### 5. Clear GStreamer Cache
```bash
rm -rf ~/.cache/gstreamer-1.0/
```

### 6. Verify Plugin Loads with NV12 Support
```bash
gst-inspect-1.0 nvtilebatcher | grep -A 5 "Pad Templates"
```

Expected output should show:
```
  SRC template: 'src'
    ...
  SINK template: 'sink'
    ...
    format: { RGBA, NV12 }
```

### 7. Test NV12 Pipeline
```bash
cd ~/ds_pipeline/new_week

python3 version_masr_multiclass_REFACTORED.py \
    --skip-interval 15 \
    --panorama-format NV12 \
    --source files \
    --mode virtualcam \
    --video1 ~/Experiments/deep_cv_football/soft_record_video/camera_left_20251109_174533.mp4 \
    --video2 ~/Experiments/deep_cv_football/soft_record_video/camera_right_20251109_174533.mp4
```

**Expected**: Ball tracking should work (virtualcam follows ball).

### 8. Verify Buffer Flow
Add debug logging to confirm nvtilebatcher is processing buffers:
```bash
GST_DEBUG=nvtilebatcher:5 python3 version_masr_multiclass_REFACTORED.py \
    --panorama-format NV12 \
    --source files \
    --mode virtualcam \
    --video1 ~/Experiments/deep_cv_football/soft_record_video/camera_left_20251109_174533.mp4 \
    --video2 ~/Experiments/deep_cv_football/soft_record_video/camera_right_20251109_174533.mp4 2>&1 | \
    grep "Extracting tiles"
```

**Expected**: Should see repeated messages:
```
Extracting tiles from NV12 panorama (5700x1900)
```

---

## Same Steps for nvdsstitch and nvdsvirtualcam

If nvdsstitch or nvdsvirtualcam also need rebuilding:

### nvdsstitch
```bash
cd ~/ds_pipeline/my_steach
make clean
make -j$(nproc)
make install
```

### nvdsvirtualcam
```bash
cd ~/ds_pipeline/my_virt_cam/src
make clean
make -j$(nproc)
make install
```

---

## Verification Checklist

After rebuilding all plugins:

- [ ] `make install` completed successfully for nvtilebatcher
- [ ] `gst-inspect-1.0 nvtilebatcher` shows `format: { RGBA, NV12 }`
- [ ] GStreamer cache cleared (`rm -rf ~/.cache/gstreamer-1.0/`)
- [ ] NV12 pipeline starts without caps negotiation errors
- [ ] nvtilebatcher receives buffers (debug logs show "Extracting tiles")
- [ ] Ball detection works (inference produces detections)
- [ ] Virtualcam follows ball in NV12 mode

---

## Troubleshooting

### Issue: "make: command not found"
Install build tools:
```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

### Issue: "NVCC not found"
Verify CUDA path:
```bash
ls -la /usr/local/cuda-12.6/bin/nvcc
```

If missing, check CUDA installation or update Makefile `NVCC` path.

### Issue: "Package gstreamer-1.0 was not found"
Install GStreamer development packages:
```bash
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev
```

### Issue: Plugin still shows RGBA-only after rebuild
1. Verify source code has correct template (gstnvtilebatcher.cpp:44)
2. Check plugin was copied to correct location:
   ```bash
   ls -lh ~/.local/share/gstreamer-1.0/plugins/libnvtilebatcher.so
   ```
3. Clear cache again:
   ```bash
   rm -rf ~/.cache/gstreamer-1.0/
   ```
4. Restart shell or run:
   ```bash
   export GST_PLUGIN_PATH=~/.local/share/gstreamer-1.0/plugins:$GST_PLUGIN_PATH
   ```

---

## Root Cause Analysis

The caps negotiation failure occurred because:

1. **Phase 3 (c891fbf)**: Updated nvtilebatcher source code to accept NV12
   - ✅ Pad template updated: `format={ RGBA, NV12 }`
   - ✅ CUDA kernel added for NV12→RGB conversion
   - ✅ Code committed to repository

2. **Missing Step**: Plugin was never rebuilt on Jetson
   - ❌ Old binary still installed (RGBA-only)
   - ❌ Caps query returns old template

3. **Pipeline Impact**:
   ```
   nvdsstitch (NV12) → tee → frame-filter (NV12) → nvtilebatcher (expects RGBA) ✗
   ```
   Result: Zero buffers flow to nvtilebatcher → zero detections → no ball tracking

4. **Fix**: Rebuild nvtilebatcher to pick up Phase 3 changes
   - New binary advertises `format={ RGBA, NV12 }`
   - Caps negotiation succeeds
   - NV12 buffers flow through pipeline
   - Ball detection works

---

**Last Updated**: 2025-11-23
**Issue**: nvtilebatcher caps negotiation failure in NV12 mode
**Fix**: Rebuild plugin with Phase 3 NV12 support
