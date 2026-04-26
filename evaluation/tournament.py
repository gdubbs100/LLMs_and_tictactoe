from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from agents.play import ResultSpec
from evaluation.metrics import MatchStats, score_results
from evaluation.registry import AgentFactory
from evaluation.runner import run_matches


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

    def run(self) -> TournamentResult:
        ratings = {name: self.initial_rating for name in self.agents}
        history: list[dict] = []
        all_games: list[ResultSpec] = []
        pair_stats: list[dict] = []

        # Snapshot initial ratings
        for name, r in ratings.items():
            history.append({"game_idx": 0, "name": name, "rating": r})

        game_counter = 0
        for name_a, name_b in combinations(self.agents.keys(), 2):
            results = run_matches(
                (name_a, self.agents[name_a]),
                (name_b, self.agents[name_b]),
                n_games=self.games_per_pair,
                alternate_starts=self.alternate_starts,
                verbose=False,
            )
            for r in results:
                # idx 0/1 in this game maps to whichever agent was placed there
                idx_a = r.agent_names.index(name_a)
                idx_b = 1 - idx_a
                score_a = _outcome_score(r, idx_a)
                score_b = 1.0 - score_a
                exp_a = _expected(ratings[name_a], ratings[name_b])
                exp_b = 1.0 - exp_a
                ratings[name_a] += self.k_factor * (score_a - exp_a)
                ratings[name_b] += self.k_factor * (score_b - exp_b)
                game_counter += 1
                history.append({"game_idx": game_counter, "name": name_a, "rating": ratings[name_a]})
                history.append({"game_idx": game_counter, "name": name_b, "rating": ratings[name_b]})
            all_games.extend(results)

            pair = score_results(results)
            for name, s in pair.items():
                row = s.to_dict()
                row["opponent"] = name_b if name == name_a else name_a
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

    with (out_dir / "games.jsonl").open("w") as f:
        for r in result.games:
            f.write(json.dumps({
                "agent_names": list(r.agent_names),
                "actions": r.actions,
                "boards": [list(b) for b in r.boards],
                "player": r.player,
                "winner": r.winner,
                "invalid": r.invalid,
                "outcome": r.outcome,
            }) + "\n")

    pd.DataFrame(
        [{"name": n, "rating": r} for n, r in result.final_ratings.items()]
    ).sort_values("rating", ascending=False).to_csv(out_dir / "ratings.csv", index=False)

    pd.DataFrame(result.rating_history).to_csv(out_dir / "rating_history.csv", index=False)
    pd.DataFrame(result.pair_stats).to_csv(out_dir / "pair_stats.csv", index=False)
    pd.DataFrame([s.to_dict() for s in result.agent_stats.values()]).to_csv(
        out_dir / "agent_stats.csv", index=False
    )
