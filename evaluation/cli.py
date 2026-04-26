from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from evaluation.config import load_config
from evaluation.metrics import score_results
from evaluation.plots import (
    plot_elo_history,
    plot_optimality,
    plot_outcome_rates,
    plot_pairwise_winrate,
)
from evaluation.registry import make_agent_factory
from evaluation.runner import run_matches, save_results
from evaluation.tournament import Tournament, save_tournament


def cmd_match(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fac_a = make_agent_factory(args.p1)
    fac_b = make_agent_factory(args.p2)
    results = run_matches((args.p1, fac_a), (args.p2, fac_b), n_games=args.n)
    save_results(results, out / "games.jsonl")
    stats = score_results(results)
    plot_outcome_rates(stats, out / "outcome_rates.png")
    plot_optimality(stats, out / "optimality.png")
    print(f"Wrote {len(results)} games + plots to {out}")


def cmd_tournament(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")

    tourney = Tournament(
        agents=cfg.agent_factories(),
        games_per_pair=cfg.games_per_pair,
        k_factor=cfg.k_factor,
        initial_rating=cfg.initial_rating,
    )
    result = tourney.run()
    save_tournament(result, out)

    plot_outcome_rates(result.agent_stats, out / "outcome_rates.png")
    plot_optimality(result.agent_stats, out / "optimality.png")
    plot_elo_history(result.rating_history, out / "elo_history.png")
    plot_pairwise_winrate(result.pair_stats, out / "pairwise_winrate.png")

    print(f"Tournament '{cfg.name}' done. {len(result.games)} games written to {out}")
    print("Final ratings:")
    for name, r in sorted(result.final_ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {r:7.1f}")


def main() -> None:
    p = argparse.ArgumentParser(prog="evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="run N games between two agents")
    m.add_argument("--p1", required=True, help="agent type for player 1")
    m.add_argument("--p2", required=True, help="agent type for player 2")
    m.add_argument("-n", type=int, default=100)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_match)

    t = sub.add_parser("tournament", help="run a round-robin ELO tournament")
    t.add_argument("--config", required=True, help="YAML config path")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tournament)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
