from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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


def plot_token_usage(
    token_stats: dict[str, dict],
    move_tokens: dict[str, list[int]],
    path: str | Path,
) -> plt.Figure | None:
    """Two-panel token usage plot.

    Left: stacked bar of total thinking + output tokens per agent.
    Right: boxplot of per-move output token distribution with mean marked.
    """
    agents = [a for a in token_stats if token_stats[a]["output"] + token_stats[a]["thinking"] > 0]
    if not agents:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- stacked bar: total tokens ---
    x = np.arange(len(agents))
    think_vals = [token_stats[a]["thinking"] for a in agents]
    out_vals = [token_stats[a]["output"] for a in agents]
    ax1.bar(x, think_vals, label="thinking", color="steelblue")
    ax1.bar(x, out_vals, bottom=think_vals, label="output", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=15, ha="right")
    ax1.set_title("Total tokens per agent")
    ax1.set_ylabel("tokens")
    ax1.legend()

    # --- boxplot: per-move distribution ---
    box_agents = [a for a in agents if move_tokens.get(a)]
    box_data = [move_tokens[a] for a in box_agents]
    ax2.boxplot(
        box_data, tick_labels=box_agents, showmeans=True, meanline=True,
        meanprops={"color": "red", "linewidth": 1.5},
        medianprops={"color": "black"},
    )
    ax2.set_title("Tokens per move (output incl. thinking)")
    ax2.set_ylabel("tokens per move")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=1.5, label="mean"),
            Line2D([0], [0], color="black", linewidth=1.5, label="median"),
        ],
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(path)
    return fig
