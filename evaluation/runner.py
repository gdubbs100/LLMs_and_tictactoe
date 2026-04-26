from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from agents.play import ResultSpec, play_game
from environment import TicTacToeEnv
from evaluation.registry import AgentFactory


def run_matches(
    agent_a: tuple[str, AgentFactory],
    agent_b: tuple[str, AgentFactory],
    n_games: int,
    alternate_starts: bool = True,
    verbose: bool = False,
) -> list[ResultSpec]:
    """Play n_games between two agents. By default alternates who starts."""
    results: list[ResultSpec] = []
    name_a, fac_a = agent_a
    name_b, fac_b = agent_b
    for i in range(n_games):
        env = TicTacToeEnv()
        a = fac_a(env)
        a.name = name_a
        b = fac_b(env)
        b.name = name_b
        if alternate_starts and i % 2 == 1:
            agents = (b, a)
        else:
            agents = (a, b)
        results.append(play_game(env, agents, verbose=verbose))
    return results


def save_results(results: Iterable[ResultSpec], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "agent_names": list(r.agent_names) if r.agent_names else None,
                "actions": r.actions,
                "boards": [list(b) for b in r.boards],
                "player": r.player,
                "winner": r.winner,
                "invalid": r.invalid,
                "final_reward": r.final_reward,
                "outcome": r.outcome,
            }) + "\n")


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
