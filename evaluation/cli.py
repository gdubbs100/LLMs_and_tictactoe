from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.ollama_agent import OllamaUnavailableError
from evaluation.config import load_config, load_match_config
from evaluation.metrics import aggregate_token_usage, collect_move_tokens, score_results
from evaluation.plots import (
    plot_elo_history,
    plot_optimality,
    plot_outcome_rates,
    plot_pairwise_winrate,
    plot_token_usage,
)
from evaluation.runner import run_matches, save_results
from evaluation.tournament import Tournament, save_tournament

logger = logging.getLogger(__name__)


def cmd_match(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_match_config(args.config)
    fac_a, fac_b = cfg.agent_factories()
    try:
        results = run_matches((cfg.p1.name, fac_a), (cfg.p2.name, fac_b), n_games=cfg.n_games)
    except OllamaUnavailableError as e:
        logger.error("Match ended early — Ollama unavailable: %s", e)
        print("Match ended early due to Ollama error. No results saved.")
        return
    save_results(results, out / "games.jsonl")
    stats = score_results(results)
    plot_outcome_rates(stats, out / "outcome_rates.png")
    plot_optimality(stats, out / "optimality.png")
    token_stats = aggregate_token_usage(results)
    move_tok = collect_move_tokens(results)
    if token_stats:
        plot_token_usage(token_stats, move_tok, out / "token_usage.png")
    print(f"Wrote {len(results)} games + plots to {out}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if args.results:
        cmd += ["--", "--results", args.results]
    subprocess.run(cmd)


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
    token_stats = aggregate_token_usage(result.games)
    move_tok = collect_move_tokens(result.games)
    if token_stats:
        plot_token_usage(token_stats, move_tok, out / "token_usage.png")

    print(f"Tournament '{cfg.name}' done. {len(result.games)} games written to {out}")
    print("Final ratings:")
    for name, r in sorted(result.final_ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {r:7.1f}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(prog="evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="run games between two agents")
    m.add_argument("--config", required=True, help="YAML config path")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_match)

    t = sub.add_parser("tournament", help="run a round-robin ELO tournament")
    t.add_argument("--config", required=True, help="YAML config path")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tournament)

    d = sub.add_parser("dashboard", help="launch the replay dashboard")
    d.add_argument("--results", default="results/", help="path to results dir or .jsonl file")
    d.set_defaults(func=cmd_dashboard)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
