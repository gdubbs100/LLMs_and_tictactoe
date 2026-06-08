from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from agents.ollama_agent import OllamaUnavailableError
from agents.play import ResultSpec
from environment.tictactoe_env import BoardSpec
from evaluation.metrics import MatchStats, score_results
from evaluation.registry import AgentFactory
from evaluation.runner import load_results, result_from_dict, run_matches

logger = logging.getLogger(__name__)

_OLLAMA_DEFER_WAIT = 30


@dataclass
class TournamentResult:
    games: list[ResultSpec] = field(default_factory=list)
    final_ratings: dict[str, float] = field(default_factory=dict)
    rating_history: list[dict] = field(default_factory=list)  # {game_idx, name, rating}
    pair_stats: list[dict] = field(default_factory=list)
    agent_stats: dict[str, MatchStats] = field(default_factory=dict)


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _outcome_score(result: ResultSpec, idx: int) -> float:
    if result.winner is None:
        return 0.5
    return 1.0 if result.winner == idx else 0.0


@dataclass
class Tournament:
    agents: dict[str, AgentFactory]
    games_per_pair: int = 50
    k_factor: float = 32.0
    initial_rating: float = 1000.0
    alternate_starts: bool = True
    board_spec: BoardSpec = field(default_factory=BoardSpec)
    env_kwargs: dict = field(default_factory=dict)

    def run(self, games_path: str | Path | None = None, resume: bool = False) -> TournamentResult:
        ratings = {name: self.initial_rating for name in self.agents}
        history: list[dict] = []
        all_games: list[ResultSpec] = []
        game_counter = 0
        pair_done: dict[frozenset, int] = {}

        for name, r in ratings.items():
            history.append({"game_idx": 0, "name": name, "rating": r})

        games_path = Path(games_path) if games_path is not None else None

        def _apply_elo(r: ResultSpec) -> None:
            """Update ratings/history/counter for one game (live or replayed)."""
            nonlocal game_counter
            name_a, name_b = r.agent_names
            score_a = _outcome_score(r, 0)
            score_b = 1.0 - score_a
            exp_a = _expected(ratings[name_a], ratings[name_b])
            exp_b = 1.0 - exp_a
            ratings[name_a] += self.k_factor * (score_a - exp_a)
            ratings[name_b] += self.k_factor * (score_b - exp_b)
            game_counter += 1
            history.append({"game_idx": game_counter, "name": name_a, "rating": ratings[name_a]})
            history.append({"game_idx": game_counter, "name": name_b, "rating": ratings[name_b]})

        if games_path is not None and games_path.exists():
            if resume:
                for d in load_results(games_path):
                    r = result_from_dict(d)
                    _apply_elo(r)
                    all_games.append(r)
                    key = frozenset(r.agent_names)
                    pair_done[key] = pair_done.get(key, 0) + 1
                logger.info("Resuming tournament: %d completed games loaded.", len(all_games))
            else:
                games_path.unlink()

        pairs = list(combinations(self.agents.keys(), 2))
        n_pairs = len(pairs)
        deferred: list[tuple[str, str]] = []

        def _process_pair(name_a: str, name_b: str, label: str) -> None:
            start = pair_done.get(frozenset((name_a, name_b)), 0)
            if start >= self.games_per_pair:
                logger.info("%s: %s vs %s — already complete, skipping.", label, name_a, name_b)
                return
            logger.info("%s: %s vs %s (from game %d)", label, name_a, name_b, start + 1)
            results = run_matches(
                (name_a, self.agents[name_a]),
                (name_b, self.agents[name_b]),
                n_games=self.games_per_pair,
                alternate_starts=self.alternate_starts,
                verbose=False,
                board_spec=self.board_spec,
                env_kwargs=self.env_kwargs,
                out_path=games_path,
                start_index=start,
            )
            for r in results:
                _apply_elo(r)
            all_games.extend(results)

        for pair_idx, (name_a, name_b) in enumerate(pairs):
            try:
                _process_pair(name_a, name_b, f"Pair {pair_idx + 1}/{n_pairs}")
            except OllamaUnavailableError:
                logger.warning(
                    "Ollama unavailable for %s vs %s — deferring. "
                    "%d non-deferred pairs remaining.",
                    name_a, name_b, n_pairs - pair_idx - 1 - len(deferred),
                )
                deferred.append((name_a, name_b))

        if deferred:
            logger.info(
                "%d pair(s) deferred due to Ollama errors. Waiting %ds before retrying...",
                len(deferred), _OLLAMA_DEFER_WAIT,
            )
            time.sleep(_OLLAMA_DEFER_WAIT)
            for retry_idx, (name_a, name_b) in enumerate(deferred):
                try:
                    _process_pair(name_a, name_b, f"Retry {retry_idx + 1}/{len(deferred)}")
                except OllamaUnavailableError:
                    logger.error(
                        "Ollama still unavailable for %s vs %s after wait — skipping pair.",
                        name_a, name_b,
                    )
            logger.info("Tournament ending — %d pair(s) may have been skipped.", len(deferred))

        pair_groups: dict[frozenset, list[ResultSpec]] = {}
        for r in all_games:
            pair_groups.setdefault(frozenset(r.agent_names), []).append(r)
        pair_stats: list[dict] = []
        for group in pair_groups.values():
            stats = score_results(group)
            names = list(stats.keys())
            for name, s in stats.items():
                row = s.to_dict()
                row["opponent"] = next(n for n in names if n != name) if len(names) > 1 else name
                pair_stats.append(row)

        agent_stats = score_results(all_games)
        return TournamentResult(
            games=all_games,
            final_ratings=ratings,
            rating_history=history,
            pair_stats=pair_stats,
            agent_stats=agent_stats,
        )


def save_tournament(result: TournamentResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # games.jsonl is written incrementally during the run; only aggregates here.
    pd.DataFrame(
        [{"name": n, "rating": r} for n, r in result.final_ratings.items()]
    ).sort_values("rating", ascending=False).to_csv(out_dir / "ratings.csv", index=False)

    pd.DataFrame(result.rating_history).to_csv(out_dir / "rating_history.csv", index=False)
    pd.DataFrame(result.pair_stats).to_csv(out_dir / "pair_stats.csv", index=False)
    pd.DataFrame([s.to_dict() for s in result.agent_stats.values()]).to_csv(
        out_dir / "agent_stats.csv", index=False
    )
