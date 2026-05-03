from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

SECTION_STYLE = (
    "padding:0.6rem 0.8rem;margin-bottom:0.6rem;border-radius:6px;"
    "font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.85rem;"
    "white-space:pre-wrap;word-wrap:break-word;"
)
LABEL_STYLE = "font-weight:700;font-size:0.75rem;text-transform:uppercase;opacity:0.7;margin-bottom:0.25rem;"
HEADER_STYLE = (
    "font-weight:700;font-size:0.8rem;padding:0.25rem 0.5rem;border-radius:4px;"
    "margin-bottom:0.5rem;margin-top:{top}rem;"
)

# Per-attempt color scheme: [primary, fallback, ...]
ATTEMPT_COLORS = [
    {"prompt": "#d0e8ff", "thinking": "#e8d0ff", "response": "#d0ffd8", "header": "#e8f0ff"},  # blue/purple/green
    {"prompt": "#ffe8d0", "thinking": "#d0fff0", "response": "#ffffd0", "header": "#fff4e8"},  # orange/teal/yellow
]


def _section(label: str, text: str, bg: str) -> str:
    safe = html.escape(text) if text else "<i>(empty)</i>"
    return (
        f'<div style="{SECTION_STYLE}background:{bg}">'
        f'<div style="{LABEL_STYLE}">{label}</div>'
        f'{safe}</div>'
    )


def _attempt_header(label: str, bg: str, top: float) -> str:
    style = HEADER_STYLE.format(top=top) + f"background:{bg};"
    return f'<div style="{style}">{html.escape(label)}</div>'


def render_llm_panel(interaction: dict | None, player_name: str, piece: str) -> None:
    st.markdown(f"**{player_name}** &nbsp; ({piece})")

    if not interaction:
        st.caption("No LLM data for this turn.")
        return

    # Normalize old flat format (pre-fallback results)
    if "attempts" not in interaction:
        interaction = {"attempts": [interaction], "move_valid": interaction.get("move_valid")}

    attempts = interaction["attempts"]
    move_valid = interaction.get("move_valid")
    multi = len(attempts) > 1

    all_sections: list[str] = []
    for i, attempt in enumerate(attempts):
        colors = ATTEMPT_COLORS[i % len(ATTEMPT_COLORS)]
        if multi:
            label = f"Attempt {i + 1}: {attempt.get('agent', player_name)}"
            all_sections.append(_attempt_header(label, colors["header"], top=0.8 if i > 0 else 0))

        all_sections.append(_section("Prompt", attempt.get("prompt", ""), colors["prompt"]))
        thinking = attempt.get("thinking", "")
        if thinking:
            all_sections.append(_section("Thinking", thinking, colors["thinking"]))
        all_sections.append(_section("Response", attempt.get("response", ""), colors["response"]))

    if move_valid is False:
        all_sections.append(
            '<div style="color:#a00;font-size:0.8rem;margin-top:0.4rem">'
            "&#9888; This move was invalid.</div>"
        )

    body = (
        '<div style="font-family:system-ui,sans-serif;color:#222;'
        'padding:0.4rem;">' + "".join(all_sections) + "</div>"
    )
    components.html(body, height=440, scrolling=True)
