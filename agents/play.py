from __future__ import annotations

from environment import TicTacToeEnv
from agents.base import Agent


def play_game(
    env: TicTacToeEnv,
    agents: tuple[Agent, Agent],
    verbose: bool = True,
) -> dict:
    """Run one episode between two agents. Returns a result summary."""
    obs, info = env.reset()
    if verbose:
        print(f"{agents[0].name} (X) vs {agents[1].name} (O)\n")

    while True:
        player = info["current_player"]
        action = agents[player].act(obs, info["valid_actions"], player)
        obs, reward, terminated, _, info = env.step(action)

        if verbose:
            print(f"\n{agents[info['mover']].name} played {action}")
            print(obs)

        if terminated:
            break

    winner_char = info["winner"]
    if info["invalid_move"]:
        outcome = f"{agents[info['mover']].name} made an invalid move and lost"
    elif winner_char is None:
        outcome = "draw"
    else:
        outcome = f"{agents[info['mover']].name} ({winner_char}) wins"

    if verbose:
        print(f"\nResult: {outcome}")

    return {
        "winner": info["mover"] if winner_char is not None else None,
        "invalid": info["invalid_move"],
        "final_reward": reward,
        "outcome": outcome,
    }
