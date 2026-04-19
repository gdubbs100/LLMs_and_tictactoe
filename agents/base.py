from __future__ import annotations

from typing import Protocol


class Agent(Protocol):
    """An agent chooses an action given the current board and legal moves."""

    name: str

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        ...
