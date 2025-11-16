#!/usr/bin/env python3
"""
Calculate what pitch should be for different ball_y positions
"""

# Config values (from nvdsvirtualcam_config.h)
LAT_MIN_CONFIG = -32.0  # Bottom of panorama
LAT_MAX_CONFIG = +22.0  # Top of panorama

# In C++, these are SWAPPED for the formula:
LAT_MIN_formula = LAT_MAX_CONFIG  # +22° (TOP)
LAT_MAX_formula = LAT_MIN_CONFIG  # -32° (BOTTOM)

PANORAMA_HEIGHT = 1900

def calc_pitch(ball_y):
    """Calculate pitch from ball_y using the C++ formula"""
    # C++ formula: *out_pitch = LAT_MIN - (y / (pano_h - 1)) * (LAT_MIN - LAT_MAX);
    pitch = LAT_MIN_formula - (ball_y / (PANORAMA_HEIGHT - 1)) * (LAT_MIN_formula - LAT_MAX_formula)
    return pitch

print("=" * 70)
print("🎯 PITCH VALUES FOR DIFFERENT BALL_Y POSITIONS")
print("=" * 70)
print(f"Panorama config: LAT_MIN={LAT_MIN_CONFIG}°, LAT_MAX={LAT_MAX_CONFIG}°")
print(f"Height: {LAT_MAX_CONFIG - LAT_MIN_CONFIG}° = 54°")
print()
print(f"{'ball_y (px)':<15} {'% from top':<15} {'pitch (°)':<15} {'Description'}")
print("=" * 70)

test_positions = [
    (100, "Top limit (min_y)"),
    (300, ""),
    (500, ""),
    (700, ""),
    (950, "CENTER (current)"),
    (1200, ""),
    (1400, ""),
    (1600, ""),
    (1800, "Bottom limit (max_y)"),
]

for y, desc in test_positions:
    pitch = calc_pitch(y)
    fraction = y / PANORAMA_HEIGHT
    print(f"{y:<15} {fraction:<15.2f} {pitch:<15.1f} {desc}")

print("=" * 70)
print()
print("📌 Key findings:")
print(f"  • At ball_y=950 (center): pitch = {calc_pitch(950):.1f}°")
print(f"  • At ball_y=100 (top limit): pitch = {calc_pitch(100):.1f}°")
print(f"  • At ball_y=1800 (bottom limit): pitch = {calc_pitch(1800):.1f}°")
print()
print("✅ If ball moves UP (W key), ball_y DECREASES → pitch INCREASES")
print("✅ Camera should be able to reach pitch ≈ +19.4° when ball at y=100")
print()
