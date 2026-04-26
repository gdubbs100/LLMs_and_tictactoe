from __future__ import annotations

# Board: tuple of 9 ints in row-major order.
#   0 = empty, 1 = player 0's piece, 2 = player 1's piece.
#
#   0 | 1 | 2
#   ---------
#   3 | 4 | 5
#   ---------
#   6 | 7 | 8

Board = tuple[int, ...]

EMPTY: Board = (0,) * 9

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def winner(board: Board) -> int:
    """Return the winning piece (1 or 2), or 0 if no winner."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
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
