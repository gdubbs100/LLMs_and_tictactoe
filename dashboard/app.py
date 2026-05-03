from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Allow `streamlit run dashboard/app.py` to find sibling packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.board_view import render_board
from dashboard.controls import init_state, render_controls, reset_playback
from dashboard.llm_panel import render_llm_panel
from dashboard.loader import (
    find_jsonl_files,
    game_label,
    load_games,
    player_interaction,
)

PLAY_INTERVAL_SEC = 1.0


def _default_results_dir() -> str:
    if "--results" in sys.argv:
        i = sys.argv.index("--results")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "results/"


def main() -> None:
    st.set_page_config(page_title="Tic-Tac-Toe Replay", layout="wide")
    init_state()

    st.title("Tic-Tac-Toe Game Replay")

    # Top row: file selection
    top = st.columns([2, 2, 3])
    root = top[0].text_input(
        "Results directory or .jsonl file",
        value=st.session_state.get("root_path", _default_results_dir()),
    )
    st.session_state.root_path = root

    files = find_jsonl_files(Path(root))
    if not files:
        st.warning(f"No .jsonl files found under `{root}`.")
        return

    file_labels = [str(f.relative_to(_PROJECT_ROOT)) if f.is_relative_to(_PROJECT_ROOT) else str(f) for f in files]
    file_choice = top[1].selectbox(
        "File", options=range(len(files)), format_func=lambda i: file_labels[i], key="file_choice"
    )
    chosen_file = files[file_choice]

    if st.session_state.get("loaded_file") != str(chosen_file):
        st.session_state.games = load_games(chosen_file)
        st.session_state.loaded_file = str(chosen_file)
        reset_playback()

    games = st.session_state.games
    if not games:
        st.warning("Selected file has no games.")
        return

    game_choice = top[2].selectbox(
        "Game",
        options=range(len(games)),
        format_func=lambda i: game_label(games[i], i),
        key="game_choice",
        on_change=reset_playback,
    )
    game = games[game_choice]

    boards = game.get("boards") or []
    actions = game.get("actions") or []
    players = game.get("player") or []
    agent_names = game.get("agent_names") or ["Player 0", "Player 1"]
    n_moves = len(actions)

    move_idx = min(st.session_state.move_idx, n_moves)
    st.session_state.move_idx = move_idx

    last_action = actions[move_idx - 1] if move_idx > 0 else None
    game_over = move_idx == n_moves
    current_board = boards[move_idx] if move_idx < len(boards) else boards[-1]

    # Main content: board on left, two player panels on right
    left, p0_col, p1_col = st.columns([2, 1, 1])

    with left:
        render_board(
            board=list(current_board),
            last_action=last_action,
            agent_names=agent_names,
            outcome=game.get("outcome"),
            game_over=game_over,
        )
        render_controls(n_moves)

    with p0_col:
        render_llm_panel(
            interaction=player_interaction(game, 0, move_idx),
            player_name=agent_names[0],
            piece="X",
        )
    with p1_col:
        render_llm_panel(
            interaction=player_interaction(game, 1, move_idx),
            player_name=agent_names[1],
            piece="O",
        )

    # Auto-play: advance one frame per rerun while playing.
    if st.session_state.playing:
        if move_idx >= n_moves:
            st.session_state.playing = False
        else:
            time.sleep(PLAY_INTERVAL_SEC)
            st.session_state.move_idx = move_idx + 1
            st.rerun()


main()
