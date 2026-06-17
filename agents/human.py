from __future__ import annotations

from environment.tictactoe_env import TicTacToeEnv


class HumanAgent:
    """Prompts the user via stdin for a cell index in [0, n-1] for an NxN board."""

    def __init__(self, env: TicTacToeEnv, name: str = "human"):
        self.name = name
        self._n = env.action_space.n

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        print(observation)
        print(f"Your turn ({self.name}, player {player_idx}). Valid moves: {valid_actions}")
        while True:
            raw = input(f"Cell [0-{self._n - 1}]: ").strip()
            try:
                action = int(raw)
            except ValueError:
                print("  not an integer, try again")
                continue
            if action not in valid_actions:
                print("  not a legal move, try again")
                continue
            return action
