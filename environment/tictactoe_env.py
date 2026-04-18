from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class BoardSpec:
    """Characters used to render the board.

    All fields must be single characters. ``pieces`` is a pair of distinct
    characters for player 0 and player 1 respectively, and must also differ
    from ``empty``.
    """

    pieces: tuple[str, str] = ("X", "O")
    empty: str = " "
    h_boundary: str = "-"
    v_boundary: str = "|"

    def __post_init__(self) -> None:
        chars = [self.pieces[0], self.pieces[1], self.empty, self.h_boundary, self.v_boundary]
        if any(len(c) != 1 for c in chars):
            raise ValueError("pieces, empty, and boundary chars must each be a single character")
        if len({self.pieces[0], self.pieces[1], self.empty}) != 3:
            raise ValueError("pieces and empty must all be distinct")

    @property
    def charset(self) -> frozenset[str]:
        return frozenset({self.pieces[0], self.pieces[1], self.empty, self.h_boundary, self.v_boundary, "\n"})


class TicTacToeEnv(gym.Env):
    """Two-player tic-tac-toe environment with a text-rendered board.

    Actions are integers in [0, 8] indexing cells in row-major order:

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8

    Observations are the ``str`` rendering of the board. Players alternate
    turns; ``info["current_player"]`` before ``step`` tells you whose move
    it is, and after ``step`` ``info["mover"]`` identifies who just moved.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        board_spec: BoardSpec | None = None,
        render_mode: str | None = None,
        invalid_move_reward: float = -1.0,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
    ):
        super().__init__()

        self.board_spec = board_spec if board_spec is not None else BoardSpec()
        self.render_mode = render_mode
        self.invalid_move_reward = invalid_move_reward
        self.win_reward = win_reward
        self.draw_reward = draw_reward

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Text(max_length=256, charset=self.board_spec.charset)

        self._board: list[str] = [self.board_spec.empty] * 9
        self._current_player: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------ core

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._board = [self.board_spec.empty] * 9
        self._current_player = 0
        self._done = False
        return self._render_text(), self._info(winner=None, invalid=False)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"action {action!r} is not in the action space Discrete(9)")

        mover = self._current_player
        empty = self.board_spec.empty

        if self._board[action] != empty:
            self._done = True
            return (
                self._render_text(),
                self.invalid_move_reward,
                True,
                False,
                self._info(winner=None, invalid=True, mover=mover),
            )

        self._board[action] = self.board_spec.pieces[mover]
        winner = self._check_winner()
        board_full = empty not in self._board

        if winner is not None:
            reward = self.win_reward
            terminated = True
        elif board_full:
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
            self._info(winner=winner, invalid=False, mover=mover),
        )

    def render(self):
        text = self._render_text()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    # ------------------------------------------------------------------ helpers

    def valid_actions(self) -> list[int]:
        empty = self.board_spec.empty
        return [i for i, c in enumerate(self._board) if c == empty]

    def _info(self, *, winner, invalid, mover=None):
        info = {
            "current_player": self._current_player,
            "valid_actions": self.valid_actions(),
            "invalid_move": invalid,
            "winner": winner,
        }
        if mover is not None:
            info["mover"] = mover
        return info

    def _check_winner(self) -> str | None:
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        b = self._board
        empty = self.board_spec.empty
        for a, c, d in lines:
            if b[a] != empty and b[a] == b[c] == b[d]:
                return b[a]
        return None

    def _render_text(self) -> str:
        spec = self.board_spec

        def row(r: int) -> str:
            cells = [self._board[r * 3 + c] for c in range(3)]
            sep = f" {spec.v_boundary} "
            return f" {cells[0]}{sep}{cells[1]}{sep}{cells[2]} "

        rows = [row(r) for r in range(3)]
        divider = spec.h_boundary * len(rows[0])
        return f"\n{divider}\n".join(rows)
