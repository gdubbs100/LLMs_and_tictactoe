from __future__ import annotations

from functools import lru_cache

from environment.board import (
    Board,
    EMPTY,
    apply_move,
    current_player,
    is_full,
    valid_actions,
    winner,
)
from environment.tictactoe_env import TicTacToeEnv

import numpy as np

# Center, corners, edges — better move ordering yields more cutoffs.
MOVE_ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)


@lru_cache(maxsize=None)
def negamax(board: Board, alpha: int = -2, beta: int = 2, discount_rate: float = 0.9) -> int:
    """Value of `board` from the perspective of the player to move.

    +1 = wins with best play, 0 = draw, -1 = loses.
    """
    if winner(board):
        # Previous player just won, so the player to move has already lost.
        return -1 * discount_rate**(np.array(board) > 0).sum()
    if is_full(board):
        return 0

    value = -2
    for i in MOVE_ORDER:
        v = action_value(board, i, alpha=-beta, beta=-alpha, discount_rate=discount_rate)
        if v > value:
            value = v
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    return value


def action_value(
    board: Board,
    action: int,
    alpha: float = -2,
    beta: float = 2,
    discount_rate: float = 0.9,
) -> float:
    """Value of taking `action` in `board`, from the mover's perspective.

    Invalid actions (occupied cells) auto-lose for the mover.
    """
    if board[action] != 0:
        return -1
    player = current_player(board)
    return -negamax(apply_move(board, action, player), -beta, -alpha, discount_rate)


def best_action(board: Board, discount_rate: float = 0.9) -> int:
    best_v, best_a = -3, -1
    for i in MOVE_ORDER:
        v = action_value(board, i, discount_rate=discount_rate)
        if v > best_v:
            best_v, best_a = v, i
    return best_a


def build_policy() -> dict[Board, int]:
    """Map every reachable non-terminal board to the optimal action."""
    policy: dict[Board, int] = {}
    seen: set[Board] = set()
    stack: list[Board] = [EMPTY]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        if winner(b) or is_full(b):
            continue
        policy[b] = best_action(b)
        player = current_player(b)
        for i in valid_actions(b):
            stack.append(apply_move(b, i, player))
    return policy


class AlphaBetaAgent:
    """Optimal tic-tac-toe agent backed by a precomputed lookup table."""

    name = "alphabeta"

    _policy: dict[Board, int] | None = None

    def __init__(self, env: TicTacToeEnv):
        self.env = env
        if AlphaBetaAgent._policy is None:
            AlphaBetaAgent._policy = build_policy()

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        return AlphaBetaAgent._policy[self.env.board]
