from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from environment.tictactoe_env import BoardSpec
from evaluation.registry import AgentFactory, make_agent_factory


def _parse_board_spec(d: dict | None) -> BoardSpec:
    if not d:
        return BoardSpec()
    return BoardSpec(**{**d, "pieces": tuple(d["pieces"])} if "pieces" in d else d)


@dataclass
class AgentConfig:
    name: str
    type: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class MatchConfig:
    p1: AgentConfig
    p2: AgentConfig
    n_games: int = 1
    board_spec: BoardSpec = field(default_factory=BoardSpec)
    env_kwargs: dict = field(default_factory=dict)

    def agent_factories(self) -> tuple[AgentFactory, AgentFactory]:
        fac1 = make_agent_factory(self.p1.type, **self.p1.kwargs)
        fac2 = make_agent_factory(self.p2.type, **self.p2.kwargs)
        return fac1, fac2


@dataclass
class TournamentConfig:
    name: str
    agents: list[AgentConfig]
    games_per_pair: int = 50
    k_factor: float = 32.0
    initial_rating: float = 1000.0
    seed: int | None = None
    board_spec: BoardSpec = field(default_factory=BoardSpec)
    env_kwargs: dict = field(default_factory=dict)

    def agent_factories(self) -> dict[str, AgentFactory]:
        out: dict[str, AgentFactory] = {}
        for a in self.agents:
            out[a.name] = make_agent_factory(a.type, **a.kwargs)
        return out


def load_match_config(path: str | Path) -> MatchConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return MatchConfig(
        p1=AgentConfig(**raw["p1"]),
        p2=AgentConfig(**raw["p2"]),
        n_games=raw.get("n_games", 1),
        board_spec=_parse_board_spec(raw.get("board_spec")),
        env_kwargs=raw.get("env", {}),
    )


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
        board_spec=_parse_board_spec(raw.get("board_spec")),
        env_kwargs=raw.get("env", {}),
    )
