# Stereo Camera Calibration Module

**See main documentation**: `/home/user/ds_pipeline/CLAUDE.MD` Section 4

## Quick Reference

**Location**: `/home/user/ds_pipeline/calibration/`
**Primary Method**: Essential Matrix (wide-angle stereo)
**Camera Angle**: 85° separation
**Pattern**: 8×6 chessboard (25mm squares)

## Calibration Results

### Individual Cameras
- **Left Camera**: RMS = 0.180 px (49 images)
- **Right Camera**: RMS = 0.198 px (63 images)

### Stereo Configuration
- **Method**: cv2.findEssentialMat() with RANSAC
- **Measured Angle**: 85.19° (target: 85°)
- **Inlier Ratio**: 3.0% (expected for wide-angle)
- **Rotation**: Yaw=-85.82°, Pitch=0.38°, Roll=-21.29°

## Key Files

- `stereo_essential_matrix.py` - Essential matrix calibration (PRIMARY)
- `stereo_calibration.py` - Standard stereo (narrow baseline only)
- `test_stiching.py` - Integration test with stitching
- `calibration_result_standard.pkl` - Binary output for pipeline

## Camera Parameters

From **Sony IMX678-AAQR1 (L100A lens)**:
- Resolution: 3840×2160 (4K)
- FOV: 100° horizontal, 55° vertical, 114° diagonal
- Distortion: -35.8% (F-Tan-Theta model)
- Aperture: F/2.7
- Operating Temp: -30°C to +85°C

## Usage

### Capture Calibration Images
```bash
python3 stereo_capture.py --left-cam 0 --right-cam 1 --count 20
```

### Run Calibration
```bash
python3 stereo_essential_matrix.py
```

### Test Stitching
```bash
python3 test_stiching.py calibration_result_standard.pkl left.jpg right.jpg
```

## Technical Notes

**Why Essential Matrix?**
- Standard cv2.stereoCalibrate() only works for narrow baselines (<20°)
- Essential matrix method handles wide-angle configurations (85°)
- RANSAC robust to outliers from severe distortion differences

**Limitations**:
- Translation vector T is normalized (unknown real baseline)
- Requires manual measurement or additional calibration for absolute scale

For complete documentation see: `CLAUDE.MD`
