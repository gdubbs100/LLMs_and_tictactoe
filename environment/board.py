from __future__ import annotations

# Board: tuple of size*size ints in row-major order (NxN, size>=3).
#   0 = empty, 1 = player 0's piece, 2 = player 1's piece.
#
#   For the default 3x3 board cells are indexed:
#       0 | 1 | 2
#       ---------
#       3 | 4 | 5
#       ---------
#       6 | 7 | 8
#
# `EMPTY` and `LINES` below are 3x3 conveniences for the precomputed fast path
# (winner()'s default lines, AlphaBetaAgent.build_policy). NxN callers pass
# lines from make_lines(size, win_length) instead.

Board = tuple[int, ...]

EMPTY: Board = (0,) * 9

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def make_lines(size: int, win_length: int) -> tuple:
    """Generate all winning lines for an NxN board with the given win_length."""
    lines = []
    for r in range(size):
        for c in range(size - win_length + 1):
            lines.append(tuple(r * size + c + i for i in range(win_length)))
    for c in range(size):
        for r in range(size - win_length + 1):
            lines.append(tuple((r + i) * size + c for i in range(win_length)))
    for r in range(size - win_length + 1):
        for c in range(size - win_length + 1):
            lines.append(tuple((r + i) * size + (c + i) for i in range(win_length)))
    for r in range(size - win_length + 1):
        for c in range(win_length - 1, size):
            lines.append(tuple((r + i) * size + (c - i) for i in range(win_length)))
    return tuple(lines)


def winner(board: Board, lines: tuple = LINES) -> int:
    """Return the winning piece (1 or 2), or 0 if no winner."""
    for line in lines:
        piece = board[line[0]]
        if piece and all(board[i] == piece for i in line):
            return piece
    return 0


def is_full(board: Board) -> bool:
    return 0 not in board


def valid_actions(board: Board) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


def current_player(board: Board) -> int:
    """Player to move (0 or 1). Player 0 always moves first."""
    return 0 if board.count(1) == board.count(2) else 1


def apply_move(board: Board, action: int, player: int) -> Board:
    return board[:action] + (player + 1,) + board[action + 1:]
