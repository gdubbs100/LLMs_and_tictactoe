from __future__ import annotations

import streamlit as st


def init_state() -> None:
    st.session_state.setdefault("move_idx", 0)
    st.session_state.setdefault("playing", False)


def reset_playback() -> None:
    st.session_state.move_idx = 0
    st.session_state.playing = False


def _step_back() -> None:
    st.session_state.move_idx = max(0, st.session_state.move_idx - 1)
    st.session_state.playing = False


def _step_forward(n_moves: int) -> None:
    st.session_state.move_idx = min(n_moves, st.session_state.move_idx + 1)
    st.session_state.playing = False


def _toggle_play() -> None:
    st.session_state.playing = not st.session_state.playing


def render_controls(n_moves: int) -> None:
    cols = st.columns([1, 1, 1, 2])
    cols[0].button(
        "⏪ Back",
        on_click=_step_back,
        disabled=st.session_state.move_idx == 0,
        use_container_width=True,
    )
    play_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    cols[1].button(
        play_label,
        on_click=_toggle_play,
        disabled=st.session_state.move_idx >= n_moves,
        use_container_width=True,
    )
    cols[2].button(
        "Step ⏩",
        on_click=_step_forward,
        args=(n_moves,),
        disabled=st.session_state.move_idx >= n_moves,
        use_container_width=True,
    )
    cols[3].markdown(
        f"<div style='text-align:center;padding-top:0.5rem'>"
        f"Move <b>{st.session_state.move_idx}</b> / {n_moves}</div>",
        unsafe_allow_html=True,
    )
