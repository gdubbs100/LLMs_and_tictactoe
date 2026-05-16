from __future__ import annotations

from agents.base import LLMAgent
from agents.hf_llm_agent import HFLLMAgent


class HFFallbackAgent(LLMAgent):
    """Wraps an HFLLMAgent and retries on invalid moves.

    Strategies:
      correction_prompt — fresh call quoting the invalid response and asking again.
      thinking_pass     — fresh non-thinking call seeded with the primary's prior thinking.
    """

    def __init__(
        self,
        primary: HFLLMAgent,
        strategy: str = "correction_prompt",
        max_retries: int = 1,
        name: str | None = None,
    ):
        self.primary = primary
        self.name = name or primary.name
        self.strategy = strategy
        self.max_retries = max_retries
        self.log_interactions = True
        self.last_interaction: list[dict] = []

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        action = self.primary.act(observation, valid_actions, player_idx)
        attempts: list[dict] = list(self.primary.last_interaction)

        for _ in range(self.max_retries):
            if action in valid_actions:
                break
            attempts.append(self._retry(observation, valid_actions, attempts[-1]))
            try:
                action = int(attempts[-1]["response"])
            except (ValueError, TypeError):
                action = -1

        self.last_interaction = attempts
        return action

    def _retry(self, observation: str, valid_actions: list[int], prev: dict) -> dict:
        valid_str = f" Valid moves: {valid_actions}." if self.primary.pass_valid_actions else ""

        if self.strategy == "correction_prompt":
            prompt = (
                f"State:\n{observation}\n\n"
                f"{prev['response']!r} is not a valid move. "
                f"Choose the correct integer for an unoccupied space.{valid_str}"
            )
            et = False
        elif self.strategy == "thinking_pass":
            note = "\nNote: the above reasoning was cut off. Complete the analysis and give your final answer." if prev.get("thinking_truncated", False) else ""
            prompt = f"Analysis:\n{prev.get('thinking', '')}{note}\n\nState:\n{observation}.{valid_str}"
            et = False
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

        messages = [self.primary.messages[0], {"role": "user", "content": prompt}]
        thinking, response = self.primary.generate(messages, enable_thinking=et)
        return {
            "agent": f"{self.primary.name} ({self.strategy})",
            "prompt": prompt,
            "thinking": thinking,
            "response": response,
        }
