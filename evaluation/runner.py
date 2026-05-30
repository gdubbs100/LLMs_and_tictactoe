from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

from agents.ollama_agent import OllamaUnavailableError
from agents.play import ResultSpec, play_game
from environment import TicTacToeEnv
from environment.tictactoe_env import BoardSpec
from evaluation.registry import AgentFactory

logger = logging.getLogger(__name__)

_GAME_RETRY_WAIT = 30


def run_matches(
    agent_a: tuple[str, AgentFactory],
    agent_b: tuple[str, AgentFactory],
    n_games: int,
    alternate_starts: bool = True,
    verbose: bool = False,
    board_spec: BoardSpec | None = None,
    env_kwargs: dict | None = None,
) -> list[ResultSpec]:
    """Play n_games between two agents. By default alternates who starts.

    Retries each game once on Ollama 500 errors (after _GAME_RETRY_WAIT seconds).
    Raises OllamaUnavailableError if the retry also fails.
    """
    results: list[ResultSpec] = []
    name_a, fac_a = agent_a
    name_b, fac_b = agent_b
    for i in range(n_games):
        swapped = alternate_starts and i % 2 == 1

        def _setup():
            env = TicTacToeEnv(board_spec=board_spec, **(env_kwargs or {}))
            a = fac_a(env)
            a.name = name_a
            b = fac_b(env)
            b.name = name_b
            return env, ((b, a) if swapped else (a, b))

        env, agents = _setup()
        try:
            result = play_game(env, agents, verbose=verbose)
        except OllamaUnavailableError as e:
            logger.warning(
                "Ollama 500 on game %d/%d (%s vs %s): %s — retrying in %ds...",
                i + 1, n_games, name_a, name_b, e, _GAME_RETRY_WAIT,
            )
            time.sleep(_GAME_RETRY_WAIT)
            env, agents = _setup()
            try:
                result = play_game(env, agents, verbose=verbose)
                logger.info("Retry succeeded for game %d/%d (%s vs %s).", i + 1, n_games, name_a, name_b)
            except OllamaUnavailableError:
                logger.error(
                    "Ollama still unavailable after retry for game %d/%d (%s vs %s). Aborting match.",
                    i + 1, n_games, name_a, name_b,
                )
                raise

        results.append(result)
        outcome = "draw" if result.winner is None else f"{result.agent_names[result.winner]} wins"
        remaining = n_games - i - 1
        logger.info(
            "[%d/%d] %s vs %s → %s  (%d left)",
            i + 1, n_games, name_a, name_b, outcome, remaining,
        )
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
                "llm_interactions": {str(k): v for k, v in (r.llm_interactions or {}).items()},
                "size": r.size,
                "win_length": r.win_length,
            }) + "\n")


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
