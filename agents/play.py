from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from environment import TicTacToeEnv
from environment.board import Board
from agents.base import Agent, LLMAgent


@dataclass
class ResultSpec:
    actions: List[int] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    boards: List[Board] = field(default_factory=list)
    player: List[int] = field(default_factory=list)
    agent_names: Tuple[str, str] | None = None
    winner: int | None = None
    invalid: bool | None = None
    final_reward: float | None = None
    outcome: str | None = None
    llm_interactions: dict = field(default_factory=dict)


def play_game(
    env: TicTacToeEnv,
    agents: tuple[Agent, Agent],
    verbose: bool = True,
) -> ResultSpec:
    """Run one episode between two agents. Returns a result summary."""

    action_log: list[int] = []
    observation_log: list[str] = []
    board_log: list[Board] = []
    player_log: list[int] = []
    llm_log: dict[int, list[dict]] = {0: [], 1: []}

    obs, info = env.reset()
    observation_log.append(obs)
    board_log.append(env.board)
    if verbose:
        print(f"{agents[0].name} (X) vs {agents[1].name} (O)\n")

    while True:
        player = info["current_player"]
        action = agents[player].act(obs, info["valid_actions"], player)

        if not (0 <= action <= 8):
            info = {"invalid_move": True, "mover": player, "winner": None, "current_player": player, "valid_actions": []}
            reward = env.invalid_move_reward
            terminated = True
        else:
            obs, reward, terminated, _, info = env.step(action)

        action_log.append(action)
        observation_log.append(obs)
        board_log.append(env.board)
        player_log.append(player)

        if isinstance(agents[player], LLMAgent) and agents[player].log_interactions:
            llm_log[player].append({
                "attempts": list(agents[player].last_interaction),
                "move_valid": not info.get("invalid_move", False),
            })

        if verbose:
            print(f"\n{agents[info['mover']].name} played {action}")
            print(obs)

        if terminated:
            break

    winner_char = info["winner"]
    mover = info["mover"]
    if info["invalid_move"]:
        outcome = f"{agents[mover].name} made an invalid move and lost"
        winner_idx: int | None = 1 - mover
    elif winner_char is None:
        outcome = "draw"
        winner_idx = None
    else:
        outcome = f"{agents[mover].name} ({winner_char}) wins"
        winner_idx = mover

    if verbose:
        print(f"\nResult: {outcome}")

    return ResultSpec(
        actions=action_log,
        observations=observation_log,
        boards=board_log,
        player=player_log,
        agent_names=(agents[0].name, agents[1].name),
        winner=winner_idx,
        invalid=info["invalid_move"],
        final_reward=reward,
        outcome=outcome,
        llm_interactions=llm_log,
    )
