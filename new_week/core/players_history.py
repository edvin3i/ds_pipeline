#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Players history management for fallback tracking."""


class PlayersHistory:
    """История позиций игроков для синхронизации analysis→display."""

    def __init__(self, history_duration=10.0):
        self.history_duration = history_duration
        self.detections = {}  # {timestamp: [list of player detections]}

    def add_players(self, players_list, timestamp):
        """Сохранить список игроков для timestamp."""
        if players_list:
            self.detections[timestamp] = players_list
            self._cleanup_old(timestamp)

    def get_players_for_timestamp(self, ts):
        """Получить игроков для ближайшего timestamp."""
        if not self.detections:
            return None

        # Находим ближайший timestamp
        timestamps = list(self.detections.keys())
        closest_ts = min(timestamps, key=lambda t: abs(t - ts))

        # Если слишком старые данные
        if abs(closest_ts - ts) > 0.5:
            return None

        return self.detections[closest_ts]

    def calculate_center_of_mass(self, ts):
        """Вычисляет центр масс игроков для timestamp."""
        players = self.get_players_for_timestamp(ts)
        if not players or len(players) == 0:
            return None

        xs = [p['x'] for p in players]
        ys = [p['y'] for p in players]

        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _cleanup_old(self, current_ts):
        """Удаляет старые данные."""
        cutoff = current_ts - self.history_duration
        self.detections = {
            ts: players
            for ts, players in self.detections.items()
            if ts >= cutoff
        }
