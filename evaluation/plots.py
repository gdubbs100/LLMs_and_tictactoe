from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation.metrics import MatchStats


def _stats_to_df(stats: dict[str, MatchStats]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in stats.values()])


def plot_outcome_rates(stats: dict[str, MatchStats], path: str | Path) -> plt.Figure:
    df = _stats_to_df(stats)
    long = df.melt(
        id_vars="agent_name",
        value_vars=["win_rate", "draw_rate", "loss_rate", "invalid_rate"],
        var_name="outcome",
        value_name="rate",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=long, x="agent_name", y="rate", hue="outcome", ax=ax)
    ax.set_title("Outcome rates per agent")
    ax.set_ylabel("rate")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_optimality(stats: dict[str, MatchStats], path: str | Path) -> plt.Figure:
    df = _stats_to_df(stats).sort_values("optimality_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="agent_name", y="optimality_rate", ax=ax, color="steelblue")
    ax.set_title("Optimality rate per agent")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_elo_history(history: list[dict] | pd.DataFrame, path: str | Path) -> plt.Figure:
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="game_idx", y="rating", hue="name", ax=ax)
    ax.set_title("ELO over games")
    ax.set_xlabel("game")
    ax.set_ylabel("rating")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_pairwise_winrate(pair_stats: list[dict] | pd.DataFrame, path: str | Path) -> plt.Figure:
    df = pd.DataFrame(pair_stats) if not isinstance(pair_stats, pd.DataFrame) else pair_stats
    matrix = df.pivot_table(
        index="agent_name", columns="opponent", values="win_rate", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Pairwise win rate (row vs column)")
    fig.tight_layout()
    fig.savefig(path)
    return fig
