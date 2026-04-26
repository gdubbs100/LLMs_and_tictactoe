from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evaluation.registry import AgentFactory, make_agent_factory


@dataclass
class AgentConfig:
    name: str
    type: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class TournamentConfig:
    name: str
    agents: list[AgentConfig]
    games_per_pair: int = 50
    k_factor: float = 32.0
    initial_rating: float = 1000.0
    seed: int | None = None

    def agent_factories(self) -> dict[str, AgentFactory]:
        out: dict[str, AgentFactory] = {}
        for a in self.agents:
            out[a.name] = make_agent_factory(a.type, **a.kwargs)
        return out


def load_config(path: str | Path) -> TournamentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    agents = [AgentConfig(**a) for a in raw["agents"]]
    return TournamentConfig(
        name=raw["name"],
        agents=agents,
        games_per_pair=raw.get("games_per_pair", 50),
        k_factor=raw.get("k_factor", 32.0),
        initial_rating=raw.get("initial_rating", 1000.0),
        seed=raw.get("seed"),
    )
