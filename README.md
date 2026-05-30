# LLMs and Tic-Tac-Toe

An experimentation framework for evaluating how well LLMs play tic-tac-toe (and NxN generalisations) against an optimal alpha-beta agent, random baselines, and each other.

## Setup

```bash
uv sync
```

## Running experiments

All experiments run through `evaluation.cli` and are defined by YAML configs in `configs/`. Use `uv run` so the project environment is used automatically.

### Match — two agents head-to-head

```bash
uv run python -m evaluation.cli match --config configs/alphabeta_vs_gemma_AB_match.yaml --out runs/alphabeta_vs_gemma_AB
```

Writes `games.jsonl` plus `outcome_rates.png`, `optimality.png`, and (for LLM agents) `token_usage.png` to the `--out` directory.

### Tournament — round-robin with Elo ratings

```bash
uv run python -m evaluation.cli tournament --config configs/tournament-3x3.yaml --out runs/tournament-3x3
```

Plays every agent pair, then writes results and plots to the `--out` directory (see table below).

### Dashboard — interactive replay of results

```bash
uv run python -m evaluation.cli dashboard --results runs/tournament-3x3
```

Launches the Streamlit replay app. `--results` accepts a results directory or a single `.jsonl` file (defaults to `results/`).

## Agents

| Type | Description |
|------|-------------|
| `alphabeta` | Optimal agent (negamax + alpha-beta pruning). Any board size; size > 3 searches on-the-fly so the first game is slow while its cache warms. |
| `random` | Random baseline; accepts `seed` kwarg |
| `ollama` | Ollama-served model (local or cloud) |
| `hf` | HuggingFace local model |
| `ollama_fallback` / `hf_fallback` | LLM agent wrapped with corrective retries on illegal moves |

> If an agent plays on an occupied square it **loses immediately**.

## Configs

Match and tournament configs live in `configs/`. To create new ones, use the `create-experiment-config` skill, or copy an existing config and edit it — the full field reference lives in `.claude/skills/create-experiment-config/assets/`.

## Tournament outputs

| File | Contents |
|------|----------|
| `games.jsonl` | All games (loadable with pandas) |
| `ratings.csv` | Final Elo ratings |
| `rating_history.csv` | Elo after each game |
| `agent_stats.csv` | Per-agent win/draw/loss/optimality |
| `pair_stats.csv` | Head-to-head breakdown |
| `config.yaml` | Saved config for reproducibility |
| `outcome_rates.png` | Win/draw/loss bars per agent |
| `optimality.png` | Optimal move % per agent |
| `elo_history.png` | Elo progression over games |
| `pairwise_winrate.png` | Agent vs agent heatmap |

## Analysing results

Load in `scratch.ipynb`:

```python
import pandas as pd

games   = pd.read_json("runs/tournament-3x3/games.jsonl", lines=True)
ratings = pd.read_csv("runs/tournament-3x3/ratings.csv")
history = pd.read_csv("runs/tournament-3x3/rating_history.csv")
stats   = pd.read_csv("runs/tournament-3x3/agent_stats.csv")
```

Or re-run plots programmatically:

```python
from evaluation.plots import plot_elo_history
fig = plot_elo_history(history, path="custom_plot.png")
fig  # renders inline in notebook
```

## Adding a new agent

1. Implement the agent (needs a `name` attr and `act(observation, valid_actions, player_idx) -> int`).
2. Register in `evaluation/registry.py`:

```python
REGISTRY["my_agent"] = lambda **kwargs: lambda env: MyAgent(**kwargs)
```

3. Reference it in configs as `type: my_agent`.
