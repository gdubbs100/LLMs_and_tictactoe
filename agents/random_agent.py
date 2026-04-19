from __future__ import annotations

import random


class RandomAgent:
    def __init__(self, name: str = "random", seed: int | None = None):
        self.name = name
        self._rng = random.Random(seed)

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        return self._rng.choice(valid_actions)
