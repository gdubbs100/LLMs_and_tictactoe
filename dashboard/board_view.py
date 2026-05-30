from __future__ import annotations

import streamlit as st

PIECE_HTML = {
    0: '<span style="color:#bbb;font-size:1rem">{i}</span>',
    1: '<span style="color:#1f6feb;font-weight:700">X</span>',
    2: '<span style="color:#d1242f;font-weight:700">O</span>',
}

def _cell_style(size: int) -> str:
    px = 90 if size <= 3 else 70 if size == 4 else 55
    font = "2.5rem" if size <= 3 else "2rem" if size == 4 else "1.5rem"
    return (
        f"width:{px}px;height:{px}px;border:2px solid #333;"
        f"text-align:center;vertical-align:middle;font-size:{font};"
        "font-family:monospace;"
    )


HIGHLIGHT_BG = "background:#fff3a8;"


def render_board(
    board: list[int],
    last_action: int | None,
    agent_names: tuple[str, str] | list,
    outcome: str | None,
    game_over: bool,
) -> None:
    name_x, name_o = agent_names[0], agent_names[1]
    title = f"**{name_x}** (X) &nbsp; vs &nbsp; **{name_o}** (O)"
    st.markdown(f"<div style='text-align:center;font-size:1.1rem'>{title}</div>", unsafe_allow_html=True)

    size = int(round(len(board) ** 0.5))
    cell_style = _cell_style(size)

    rows_html = []
    for r in range(size):
        cells = []
        for c in range(size):
            i = r * size + c
            piece = PIECE_HTML[board[i]].format(i=i)
            style = cell_style + (HIGHLIGHT_BG if i == last_action else "")
            cells.append(f'<td style="{style}">{piece}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        '<table style="border-collapse:collapse;margin:1rem auto;">'
        + "".join(rows_html)
        + "</table>"
    )
    st.markdown(table, unsafe_allow_html=True)

    if game_over and outcome:
        st.markdown(
            f"<div style='text-align:center;font-size:1rem;color:#444'>"
            f"<b>Result:</b> {outcome}</div>",
            unsafe_allow_html=True,
        )
