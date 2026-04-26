from __future__ import annotations

from typing import Callable

from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.alphabeta_agent import AlphaBetaAgent
from environment import TicTacToeEnv


AgentFactory = Callable[[TicTacToeEnv], Agent]


def _random_factory(**kwargs) -> AgentFactory:
    def make(env: TicTacToeEnv) -> Agent:
        return RandomAgent(**kwargs)
    return make


def _alphabeta_factory(**kwargs) -> AgentFactory:
    def make(env: TicTacToeEnv) -> Agent:
        return AlphaBetaAgent(env, **kwargs)
    return make


REGISTRY: dict[str, Callable[..., AgentFactory]] = {
    "random": _random_factory,
    "alphabeta": _alphabeta_factory,
}


def make_agent_factory(type_: str, **kwargs) -> AgentFactory:
    if type_ not in REGISTRY:
        raise KeyError(f"unknown agent type {type_!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[type_](**kwargs)
