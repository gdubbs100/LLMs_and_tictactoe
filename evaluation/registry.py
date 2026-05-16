from __future__ import annotations

from typing import Callable

from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.hf_llm_agent import HFLLMAgent
from agents.ollama_agent import OllamaAgent
from agents.fallback_agent import FallbackAgent
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


def _hf_factory(**kwargs) -> AgentFactory:
    def make(env: TicTacToeEnv) -> Agent:
        return HFLLMAgent(**kwargs)
    return make


def _hf_fallback_factory(**kwargs) -> AgentFactory:
    primary_cfg = kwargs.get("primary", {})
    other = {k: v for k, v in kwargs.items() if k != "primary"}

    def make(env: TicTacToeEnv) -> Agent:
        primary = HFLLMAgent(**primary_cfg)
        return FallbackAgent(primary=primary, **other)
    return make


def _ollama_factory(**kwargs) -> AgentFactory:
    def make(env: TicTacToeEnv) -> Agent:
        return OllamaAgent(**kwargs)
    return make


def _ollama_fallback_factory(**kwargs) -> AgentFactory:
    primary_cfg = kwargs.get("primary", {})
    other = {k: v for k, v in kwargs.items() if k != "primary"}

    def make(env: TicTacToeEnv) -> Agent:
        return FallbackAgent(primary=OllamaAgent(**primary_cfg), **other)
    return make


REGISTRY: dict[str, Callable[..., AgentFactory]] = {
    "random": _random_factory,
    "alphabeta": _alphabeta_factory,
    "hf": _hf_factory,
    "hf_fallback": _hf_fallback_factory,
    "ollama": _ollama_factory,
    "ollama_fallback": _ollama_fallback_factory,
}


def make_agent_factory(type_: str, **kwargs) -> AgentFactory:
    if type_ not in REGISTRY:
        raise KeyError(f"unknown agent type {type_!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[type_](**kwargs)
