#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify camera trajectory movement filtering.

Tests that temporary reversals (ball moves 300px right then back)
are filtered out while sustained movements are preserved.
"""

import sys
sys.path.insert(0, '/home/nvidia/ds_pipeline')

from new_week.core.camera_trajectory_history import CameraTrajectoryHistory
from new_week.core.players_history import PlayersHistory

def test_reversal_filtering():
    """Тест фильтрования временных движений (разворотов)."""

    print("\n" + "="*80)
    print("TEST: Movement filtering (temporary reversals)")
    print("="*80)

    # Create histories
    camera_traj = CameraTrajectoryHistory(max_gap=3.0, outlier_threshold=300)
    players_hist = PlayersHistory()

    # Create synthetic player data
    players_data = [
        {'x': 100, 'y': 200},
        {'x': 200, 'y': 300}
    ]
    # COM should be (150, 250)

    # Add players history
    for ts in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
        players_hist.add_players(players_data, ts)

    # Scenario: Ball makes REVERSAL (300px right, then 300px back)
    # t=10.0: Ball at (500, 400)
    # t=10.5: Ball at (800, 400)  <- moved 300px RIGHT
    # t=11.0: Ball at (500, 400)  <- moved 300px LEFT (back to start!)
    # t=12.0: Ball at (600, 400)  <- moves to new position (sustained)
    # t=13.0: Ball at (700, 400)  <- continues moving RIGHT

    ball_history = {
        10.0: [0, 0, 100, 100, 0.9, 0, 500, 400, 0, 0, False],  # Start at 500
        10.5: [0, 0, 100, 100, 0.9, 0, 800, 400, 0, 0, False],  # Move RIGHT 300px
        11.0: [0, 0, 100, 100, 0.9, 0, 500, 400, 0, 0, False],  # Back to 500 (REVERSAL!)
        12.0: [0, 0, 100, 100, 0.9, 0, 600, 400, 0, 0, False],  # New position 600
        13.0: [0, 0, 100, 100, 0.9, 0, 700, 400, 0, 0, False],  # Continue RIGHT (sustained)
    }

    print(f"\n📋 Ball history (with REVERSAL at t=10.5-11.0):")
    for ts in sorted(ball_history.keys()):
        det = ball_history[ts]
        print(f"   t={ts:.1f}: Ball at x={det[6]:.0f}")

    print(f"\n🎯 Expected behavior:")
    print(f"   - t=10.0→10.5: Move 300px RIGHT (sustained? no, reversal coming)")
    print(f"   - t=10.5→11.0: Back 300px LEFT (REVERSAL! filter out intermediate point)")
    print(f"   - t=11.0→13.0: Move RIGHT gradually (sustained movement, keep all)")

    # Process
    print(f"\n⚙️  Processing...")
    camera_traj.populate_camera_trajectory_from_ball_history(ball_history, players_hist, fps=30)

    # Verify results
    print(f"\n📊 Results:")
    print(f"   Total points: {len(camera_traj.camera_trajectory)}")
    print(f"   Source breakdown: {camera_traj.get_stats()['sources']}")

    print(f"\n📍 Trajectory points AFTER filtering:")
    for ts in sorted(camera_traj.camera_trajectory.keys()):
        point = camera_traj.camera_trajectory[ts]
        source = point['source_type']
        x = point['x']
        print(f"   t={ts:5.2f}: x={x:6.0f} [source={source:15s}]")

    # Analysis
    print(f"\n🔬 Movement analysis:")

    times = sorted(camera_traj.camera_trajectory.keys())
    ball_points = []

    for ts in times:
        point = camera_traj.camera_trajectory[ts]
        if point['source_type'] == 'ball':
            ball_points.append((ts, point['x']))

    if ball_points:
        print(f"   Ball trajectory points: {len(ball_points)}")
        for ts, x in ball_points:
            print(f"     t={ts:.2f}: x={x:.0f}")

        # Check for reversal filtering
        if len(ball_points) >= 2:
            # Find if reversal was detected
            has_reversal = False
            for i in range(len(ball_points) - 1):
                t1, x1 = ball_points[i]
                t2, x2 = ball_points[i + 1]

                if i > 0:
                    t_prev, x_prev = ball_points[i - 1]
                    # If went right then left (or left then right) - reversal
                    if (x1 > x_prev and x2 < x1) or (x1 < x_prev and x2 > x1):
                        has_reversal = True

            if has_reversal:
                print(f"\n⚠️  WARNING: Reversal detected in trajectory!")
            else:
                print(f"\n✅ PASS: No reversals - temporary movements filtered correctly")

    print("\n✅ Test complete!")
    print("="*80 + "\n")


def test_sustained_movement():
    """Тест что sustained movements не фильтруются."""

    print("\n" + "="*80)
    print("TEST: Sustained movement preservation")
    print("="*80)

    camera_traj = CameraTrajectoryHistory(max_gap=3.0, outlier_threshold=300)
    players_hist = PlayersHistory()

    players_data = [
        {'x': 100, 'y': 200},
        {'x': 200, 'y': 300}
    ]

    for ts in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
        players_hist.add_players(players_data, ts)

    # Scenario: Ball moves SUSTAINED (500→600→700→800, no reversal)
    ball_history = {
        10.0: [0, 0, 100, 100, 0.9, 0, 500, 400, 0, 0, False],
        11.0: [0, 0, 100, 100, 0.9, 0, 600, 400, 0, 0, False],
        12.0: [0, 0, 100, 100, 0.9, 0, 700, 400, 0, 0, False],
        13.0: [0, 0, 100, 100, 0.9, 0, 800, 400, 0, 0, False],
    }

    print(f"\n📋 Ball history (sustained movement):")
    for ts in sorted(ball_history.keys()):
        det = ball_history[ts]
        print(f"   t={ts:.1f}: Ball at x={det[6]:.0f}")

    print(f"\n⚙️  Processing...")
    camera_traj.populate_camera_trajectory_from_ball_history(ball_history, players_hist, fps=30)

    print(f"\n📊 Results:")
    times = sorted(camera_traj.camera_trajectory.keys())
    ball_points = [
        (ts, camera_traj.camera_trajectory[ts]['x'])
        for ts in times
        if camera_traj.camera_trajectory[ts]['source_type'] == 'ball'
    ]

    print(f"   Ball points: {len(ball_points)}")
    if len(ball_points) >= 4:
        print(f"   ✅ PASS: All ball points preserved (sustained movement not filtered)")
    else:
        print(f"   ⚠️  WARNING: Some ball points were filtered (expected 4, got {len(ball_points)})")

    for ts, x in ball_points:
        print(f"     t={ts:.2f}: x={x:.0f}")

    print("\n✅ Test complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_reversal_filtering()
    test_sustained_movement()
