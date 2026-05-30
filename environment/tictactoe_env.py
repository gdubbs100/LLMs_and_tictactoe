from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from environment.board import (
    Board,
    apply_move,
    is_full,
    make_lines,
    valid_actions as board_valid_actions,
    winner,
)


@dataclass(frozen=True)
class BoardSpec:
    """Tokens used to render the board.

    ``pieces`` is a pair of distinct tokens for player 0 and player 1, each
    different from ``empty``. Pieces and empty may be multi-character strings;
    cells are padded to a common width on render so the grid stays aligned.
    Boundary chars must be single characters.
    """

    pieces: tuple[str, str] = ("X", "O")
    empty: str = " "
    h_boundary: str = "-"
    v_boundary: str = "|"

    def __post_init__(self) -> None:
        tokens = [self.pieces[0], self.pieces[1], self.empty]
        if any(len(t) < 1 for t in tokens):
            raise ValueError("pieces and empty must be non-empty strings")
        if any(len(c) != 1 for c in (self.h_boundary, self.v_boundary)):
            raise ValueError("boundary chars must each be a single character")
        if len(set(tokens)) != 3:
            raise ValueError("pieces and empty must all be distinct")

    @property
    def cell_width(self) -> int:
        return max(len(self.pieces[0]), len(self.pieces[1]), len(self.empty))

    @property
    def charset(self) -> frozenset[str]:
        chars: set[str] = {self.h_boundary, self.v_boundary, "\n"}
        for token in (self.pieces[0], self.pieces[1], self.empty):
            chars.update(token)
        return frozenset(chars)


class TicTacToeEnv(gym.Env):
    """Two-player NxN tic-tac-toe environment with a text-rendered board.

    Actions are integers in [0, size*size-1] indexing cells in row-major order.
    Observations are the str rendering of the board. Players alternate turns.

    Args:
        size:       Board dimension (default 3 for standard 3x3).
        win_length: Pieces in a row needed to win (default = size).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        board_spec: BoardSpec | None = None,
        render_mode: str | None = None,
        invalid_move_reward: float = -1.0,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
        size: int = 3,
        win_length: int | None = None,
    ):
        super().__init__()

        self.board_spec = board_spec if board_spec is not None else BoardSpec()
        self.render_mode = render_mode
        self.invalid_move_reward = invalid_move_reward
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.size = size
        self.win_length = win_length if win_length is not None else size
        self._lines = make_lines(self.size, self.win_length)

        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Text(max_length=512, charset=self.board_spec.charset)

        self._board: Board = (0,) * (size * size)
        self._current_player: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------ core

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._board = (0,) * (self.size * self.size)
        self._current_player = 0
        self._done = False
        return self._render_text(), self._info(winner_piece=0, invalid=False)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"action {action!r} is not in the action space Discrete({self.size ** 2})")

        mover = self._current_player

        if self._board[action] != 0:
            self._done = True
            return (
                self._render_text(),
                self.invalid_move_reward,
                True,
                False,
                self._info(winner_piece=0, invalid=True, mover=mover),
            )

        self._board = apply_move(self._board, action, mover)
        winner_piece = winner(self._board, self._lines)

        if winner_piece:
            reward = self.win_reward
            terminated = True
        elif is_full(self._board):
            reward = self.draw_reward
            terminated = True
        else:
            reward = 0.0
            terminated = False
            self._current_player = 1 - mover

        self._done = terminated
        return (
            self._render_text(),
            reward,
            terminated,
            False,
            self._info(winner_piece=winner_piece, invalid=False, mover=mover),
        )

    def render(self):
        text = self._render_text()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    # ------------------------------------------------------------------ helpers

    @property
    def board(self) -> Board:
        return self._board

    def valid_actions(self) -> list[int]:
        return board_valid_actions(self._board)

    def _info(self, *, winner_piece: int, invalid: bool, mover: int | None = None):
        spec = self.board_spec
        info = {
            "current_player": self._current_player,
            "valid_actions": self.valid_actions(),
            "invalid_move": invalid,
            "winner": spec.pieces[winner_piece - 1] if winner_piece else None,
        }
        if mover is not None:
            info["mover"] = mover
        return info

    def _render_text(self) -> str:
        spec = self.board_spec
        width = spec.cell_width
        cells = [(spec.empty if v == 0 else spec.pieces[v - 1]).center(width) for v in self._board]

        def row(r: int) -> str:
            sep = f" {spec.v_boundary} "
            return sep.join(f" {cells[r * self.size + c]} " for c in range(self.size))

        rows = [row(r) for r in range(self.size)]
        divider = spec.h_boundary * len(rows[0])
        return f"\n{divider}\n".join(rows)
