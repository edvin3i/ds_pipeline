#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify camera trajectory gap filling logic.
Tests that gaps > 3s are filled with uniformly spaced player COM positions.
"""

import sys
sys.path.insert(0, '/home/nvidia/ds_pipeline')

from new_week.core.camera_trajectory_history import CameraTrajectoryHistory
from new_week.core.players_history import PlayersHistory

def test_gap_filling():
    """Тест заполнения разрывов >3s player COM."""

    print("\n" + "="*80)
    print("TEST: Gap filling with player center-of-mass (>3s gaps)")
    print("="*80)

    # Create histories
    camera_traj = CameraTrajectoryHistory(max_gap=3.0, outlier_threshold=300)
    players_hist = PlayersHistory()

    # Create synthetic player data (two players)
    # Players will be at fixed positions for simplicity
    players_data = [
        {'x': 100, 'y': 200},
        {'x': 200, 'y': 300}
    ]
    # COM should be (150, 250)

    # Add players history at multiple timestamps
    for ts in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
        players_hist.add_players(players_data, ts)

    # Create synthetic ball history with a 4-second gap (gap > max_gap=3.0)
    ball_history = {
        10.0: [0, 0, 100, 100, 0.9, 0, 500, 400, 0, 0, False],  # Ball at t=10
        11.0: [0, 0, 100, 100, 0.9, 0, 510, 410, 0, 0, False],  # Ball at t=11
        15.0: [0, 0, 100, 100, 0.9, 0, 600, 500, 0, 0, False],  # Ball recovered at t=15 (4s gap!)
    }

    print(f"\n📋 Ball history:")
    for ts, det in ball_history.items():
        print(f"   t={ts:.1f}: Ball at ({det[6]:.0f}, {det[7]:.0f})")

    print(f"\n👥 Player COM: (150, 250) [avg of two players]")

    print(f"\n🔍 Expected gap: t=11.0 → t=15.0 (4.0 seconds > 3.0 threshold)")
    print(f"   → Should fill with player COM points every 0.5s (15 frames at 30fps)")
    print(f"   → Expected points: t=11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5")
    print(f"   → Plus blend at t≈14.25 (85% into gap)")

    # Process
    print(f"\n⚙️  Processing...")
    camera_traj.populate_camera_trajectory_from_ball_history(ball_history, players_hist, fps=30)

    # Verify results
    print(f"\n📊 Results:")
    print(f"   Total points: {len(camera_traj.camera_trajectory)}")
    print(f"   Source breakdown: {camera_traj.get_stats()['sources']}")

    print(f"\n📍 Trajectory points:")
    for ts in sorted(camera_traj.camera_trajectory.keys()):
        point = camera_traj.camera_trajectory[ts]
        source = point['source_type']
        x, y = point['x'], point['y']
        conf = point['confidence']
        print(f"   t={ts:5.2f}: ({x:6.0f}, {y:6.0f}) [source={source:20s} conf={conf:.2f}]")

    # Analyze gap filling
    print(f"\n🔬 Gap analysis:")
    player_points = [ts for ts, p in camera_traj.camera_trajectory.items()
                     if p['source_type'] == 'player']

    if player_points:
        print(f"   Player COM points: {len(player_points)}")
        for ts in player_points:
            point = camera_traj.camera_trajectory[ts]
            print(f"     t={ts:.2f}: ({point['x']:.0f}, {point['y']:.0f})")

        if len(player_points) > 1:
            intervals = [player_points[i+1] - player_points[i] for i in range(len(player_points)-1)]
            avg_interval = sum(intervals) / len(intervals)
            print(f"   Spacing between points: avg={avg_interval:.2f}s (expected ~0.5s)")

    blend_points = [ts for ts, p in camera_traj.camera_trajectory.items()
                    if p['source_type'] == 'blend']
    if blend_points:
        print(f"   Blend transition points: {len(blend_points)}")
        for ts in blend_points:
            point = camera_traj.camera_trajectory[ts]
            print(f"     t={ts:.2f}: ({point['x']:.0f}, {point['y']:.0f})")

    print("\n✅ Test complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    test_gap_filling()
