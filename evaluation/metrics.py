from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agents.alphabeta_agent import action_value
from agents.play import ResultSpec
from environment.board import Board, current_player, is_full, winner


@dataclass
class MatchStats:
    agent_name: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    invalid_moves: int = 0
    optimal_actions: int = 0
    total_actions: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0

    @property
    def invalid_rate(self) -> float:
        return self.invalid_moves / self.games if self.games else 0.0

    @property
    def optimality_rate(self) -> float:
        return self.optimal_actions / self.total_actions if self.total_actions else 0.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "invalid_moves": self.invalid_moves,
            "optimal_actions": self.optimal_actions,
            "total_actions": self.total_actions,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "invalid_rate": self.invalid_rate,
            "optimality_rate": self.optimality_rate,
        }


def is_optimal_action(board: Board, action: int) -> bool:
    """An action is optimal iff its value equals the max value at this state."""
    if winner(board) or is_full(board):
        return False
    chosen = action_value(board, action)
    best = max(action_value(board, a) for a in range(9))
    return abs(chosen - best) < 1e-9


def count_optimal_actions(result: ResultSpec) -> dict[int, tuple[int, int]]:
    """Returns {player_idx: (optimal, total)} for each agent in the game."""
    counts = {0: [0, 0], 1: [0, 0]}
    for board, action, player in zip(result.boards, result.actions, result.player):
        counts[player][1] += 1
        if is_optimal_action(board, action):
            counts[player][0] += 1
    return {p: (c[0], c[1]) for p, c in counts.items()}


def score_results(results: Iterable[ResultSpec]) -> dict[str, MatchStats]:
    """Aggregate per-agent stats. Uses agent_names recorded on each result."""
    stats: dict[str, MatchStats] = {}
    for r in results:
        if r.agent_names is None:
            raise ValueError("ResultSpec.agent_names is required for scoring")
        for idx, name in enumerate(r.agent_names):
            s = stats.setdefault(name, MatchStats(agent_name=name))
            s.games += 1
            if r.winner is None:
                s.draws += 1
            elif r.winner == idx:
                s.wins += 1
            else:
                s.losses += 1
                if r.invalid and r.player and r.player[-1] == idx:
                    s.invalid_moves += 1
        opt = count_optimal_actions(r)
        for idx, name in enumerate(r.agent_names):
            o, t = opt[idx]
            stats[name].optimal_actions += o
            stats[name].total_actions += t
    return stats
