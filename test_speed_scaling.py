#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify ball speed scaling logic.

Tests that ball_scale increases with movement distance:
- Slow movement (< 50px) → scale = 1.0x
- Medium movement (100-300px) → scale = 1.5x
- Fast movement (500+px) → scale = 2.5x (clamped)
"""

import sys
sys.path.insert(0, '/home/nvidia/ds_pipeline')

from new_week.core.camera_trajectory_history import CameraTrajectoryHistory
from new_week.core.players_history import PlayersHistory


def test_speed_scaling():
    """Тест масштабирования мяча по скорости движения."""

    print("\n" + "="*80)
    print("TEST: Ball speed scaling (ball_scale based on movement distance)")
    print("="*80)

    # Create histories
    camera_traj = CameraTrajectoryHistory(max_gap=3.0, outlier_threshold=300)
    players_hist = PlayersHistory()

    # Create synthetic player data
    players_data = [
        {'x': 100, 'y': 200},
        {'x': 200, 'y': 300}
    ]

    # Add players history
    for ts in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
        players_hist.add_players(players_data, ts)

    # Scenario: Ball with different speeds
    # t=10.0: Ball at (500, 400)
    # t=11.0: Ball at (530, 400) - distance=30px (slow, scale=1.0)
    # t=12.0: Ball at (700, 400) - distance=170px (medium, scale~1.4)
    # t=13.0: Ball at (1000, 400) - distance=300px (fast, scale~2.2)
    # t=14.0: Ball at (1500, 400) - distance=500px (very fast, scale=2.5 clamped)
    # t=15.0: Ball at (1600, 400) - distance=100px (medium, scale~1.2)

    ball_history = {
        10.0: [0, 0, 100, 100, 0.9, 0, 500, 400, 0, 0, False],   # Start
        11.0: [0, 0, 100, 100, 0.9, 0, 530, 400, 0, 0, False],   # +30px (slow)
        12.0: [0, 0, 100, 100, 0.9, 0, 700, 400, 0, 0, False],   # +170px (medium)
        13.0: [0, 0, 100, 100, 0.9, 0, 1000, 400, 0, 0, False],  # +300px (fast)
        14.0: [0, 0, 100, 100, 0.9, 0, 1500, 400, 0, 0, False],  # +500px (very fast)
        15.0: [0, 0, 100, 100, 0.9, 0, 1600, 400, 0, 0, False],  # +100px (medium)
    }

    print(f"\n📋 Ball history with different speeds:")
    for ts in sorted(ball_history.keys()):
        det = ball_history[ts]
        print(f"   t={ts:.1f}: Ball at x={det[6]:.0f}")

    print(f"\n🎯 Expected distances and scales:")
    print(f"   t=10→11: distance=30px   → scale ≈ 1.0 (< 50px threshold)")
    print(f"   t=11→12: distance=170px  → scale ≈ 1.4 (linear: 50-500 range)")
    print(f"   t=12→13: distance=300px  → scale ≈ 1.8 (linear: 50-500 range)")
    print(f"   t=13→14: distance=500px  → scale ≈ 2.5 (at max threshold)")
    print(f"   t=14→15: distance=100px  → scale ≈ 1.2 (linear: 50-500 range)")

    # Process
    print(f"\n⚙️  Processing...")
    camera_traj.populate_camera_trajectory_from_ball_history(ball_history, players_hist, fps=30)

    # Verify results
    print(f"\n📊 Results:")
    print(f"   Total points: {len(camera_traj.camera_trajectory)}")
    print(f"   Source breakdown: {camera_traj.get_stats()['sources']}")

    print(f"\n📍 Ball points AFTER speed scaling:")
    times = sorted(camera_traj.camera_trajectory.keys())
    ball_points = []

    for ts in times:
        point = camera_traj.camera_trajectory[ts]
        if point['source_type'] == 'ball':
            scale = point.get('ball_scale', 'N/A')
            x = point['x']
            print(f"   t={ts:5.2f}: x={x:7.0f}, ball_scale={scale:.2f}" if isinstance(scale, float)
                  else f"   t={ts:5.2f}: x={x:7.0f}, ball_scale={scale}")
            ball_points.append((ts, x, scale))

    # Analysis
    print(f"\n🔬 Speed scaling analysis:")

    if len(ball_points) >= 2:
        print(f"   Ball points with scales: {len(ball_points)}")

        # Check scaling pattern
        scales = [p[2] for p in ball_points if isinstance(p[2], float)]
        if scales:
            print(f"   Scale range: [{min(scales):.2f}, {max(scales):.2f}]")
            print(f"   Scales in order: {[f'{s:.2f}' for s in scales]}")

            # Verify expected behavior
            if scales[0] <= 1.0:
                print(f"   ✅ First scale (slow) ≤ 1.0")
            else:
                print(f"   ⚠️  First scale should be ≤ 1.0")

            if max(scales) <= 2.5:
                print(f"   ✅ Max scale ≤ 2.5 (clamped)")
            else:
                print(f"   ⚠️  Max scale exceeds 2.5")

            if scales[-1] < max(scales):
                print(f"   ✅ Scale varies with distance (not constant)")
            else:
                print(f"   ⚠️  Scales should vary with movement distance")

    print("\n✅ Test complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_speed_scaling()
