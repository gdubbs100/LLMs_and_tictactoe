from __future__ import annotations


class HumanAgent:
    """Prompts the user via stdin for a cell index in [0, 8]."""

    def __init__(self, name: str = "human"):
        self.name = name

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        print(observation)
        print(f"Your turn ({self.name}, player {player_idx}). Valid moves: {valid_actions}")
        while True:
            raw = input("Cell [0-8]: ").strip()
            try:
                action = int(raw)
            except ValueError:
                print("  not an integer, try again")
                continue
            if action not in valid_actions:
                print("  not a legal move, try again")
                continue
            return action
