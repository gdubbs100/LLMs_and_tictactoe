from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class Agent(Protocol):
    """An agent chooses an action given the current board and legal moves."""

    name: str

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        ...


class LLMAgent(ABC):
    """Base class for LLM-backed agents.

    Subclasses must populate ``self.last_interaction`` after each ``act()`` call
    as a list of attempt dicts, each with keys ``agent``, ``prompt``, ``thinking``,
    and ``response``. The game runner reads this when ``log_interactions`` is True.
    """

    name: str
    log_interactions: bool = True
    last_interaction: list[dict]

    @abstractmethod
    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        ...
