from __future__ import annotations

from pathlib import Path

from evaluation.runner import load_results


def find_jsonl_files(root: Path) -> list[Path]:
    """Return all .jsonl files under ``root`` (or [root] if root is a file)."""
    root = Path(root)
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.jsonl"))


def load_games(path: Path) -> list[dict]:
    return load_results(path)


def game_label(game: dict, idx: int) -> str:
    names = game.get("agent_names") or ["?", "?"]
    outcome = game.get("outcome") or "?"
    return f"{idx}: {names[0]} vs {names[1]} — {outcome}"


def player_interaction(game: dict, player_idx: int, move_idx: int) -> dict | None:
    """Return the LLM interaction shown when board state is at ``move_idx``.

    ``move_idx`` is the index into ``boards`` (0 = initial). The interaction
    returned is the one that produced the board at ``move_idx``, i.e. the move
    just played by ``player_idx``. Returns None if no such interaction exists.

    Returns a dict with keys ``attempts`` (list) and ``move_valid`` (bool).
    Old results stored a flat dict; those are normalized to the same shape.
    """
    if move_idx <= 0:
        return None
    players = game.get("player") or []
    if move_idx > len(players) or players[move_idx - 1] != player_idx:
        return None
    interactions = (game.get("llm_interactions") or {}).get(str(player_idx)) or []
    turn_count = players[:move_idx].count(player_idx)
    if turn_count == 0 or turn_count > len(interactions):
        return None
    entry = interactions[turn_count - 1]
    # Normalize old flat format (no "attempts" key)
    if "attempts" not in entry:
        entry = {"attempts": [entry], "move_valid": entry.get("move_valid")}
    return entry
