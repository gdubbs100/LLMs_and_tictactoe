# LLMs and Tic-Tac-Toe

A framework for building and evaluating tic-tac-toe agents, with an optimal AlphaBeta agent as reference baseline.

## Setup

```bash
uv sync
```

## Agents

| Type | Description |
|------|-------------|
| `alphabeta` | Optimal agent (negamax + alpha-beta pruning) |
| `random` | Random baseline; accepts `seed` kwarg |

> If an agent plays on an occupied square it **loses immediately**.

---

## Running Experiments

### Match — quick 2-agent comparison

```bash
python -m evaluation.cli match --p1 alphabeta --p2 random -n 100 --out runs/ab_vs_rand/
```

Options:
- `--p1`, `--p2` — agent types (from table above)
- `-n` — number of games
- `--out` — output directory

Outputs: `games.jsonl`, `outcome_rates.png`, `optimality.png`

---

### Tournament — round-robin with ELO

Define your agents in a YAML config (copy `configs/example_tournament.yaml`):

```yaml
name: my_experiment
games_per_pair: 50   # higher = slower but more stable ELO
k_factor: 32         # ELO update sensitivity
initial_rating: 1000

agents:
  - name: alphabeta
    type: alphabeta
  - name: random_a
    type: random
    kwargs: {seed: 1}
  - name: random_b
    type: random
    kwargs: {seed: 2}
```

Run:

```bash
python -m evaluation.cli tournament --config configs/my_experiment.yaml --out runs/my_experiment/
```

Outputs:

| File | Contents |
|------|----------|
| `games.jsonl` | All games (loadable with pandas) |
| `ratings.csv` | Final ELO ratings |
| `rating_history.csv` | ELO after each game |
| `agent_stats.csv` | Per-agent win/draw/loss/optimality |
| `pair_stats.csv` | Head-to-head breakdown |
| `config.yaml` | Saved config for reproducibility |
| `outcome_rates.png` | Win/draw/loss bars per agent |
| `optimality.png` | Optimal move % per agent |
| `elo_history.png` | ELO progression over games |
| `pairwise_winrate.png` | Agent vs agent heatmap |

---

## Analysing Results

Load in `scratch.ipynb`:

```python
import pandas as pd

games   = pd.read_json("runs/my_experiment/games.jsonl", lines=True)
ratings = pd.read_csv("runs/my_experiment/ratings.csv")
history = pd.read_csv("runs/my_experiment/rating_history.csv")
stats   = pd.read_csv("runs/my_experiment/agent_stats.csv")
```

Or re-run plots programmatically:

```python
from evaluation.plots import plot_elo_history
fig = plot_elo_history(history, path="custom_plot.png")
fig  # renders inline in notebook
```

---

## Adding a New Agent

1. Implement the agent (needs `name` attr and `act(observation, valid_actions, player_idx) -> int`).
2. Register in `evaluation/registry.py`:

```python
REGISTRY["my_agent"] = lambda **kwargs: lambda env: MyAgent(**kwargs)
```

3. Use in configs or CLI as `type: my_agent` / `--p1 my_agent`.
