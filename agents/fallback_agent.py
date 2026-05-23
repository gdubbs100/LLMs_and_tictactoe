from __future__ import annotations

from agents.base import LLMAgent
from agents.chat_agent import ChatAgent


class FallbackAgent(LLMAgent):
    """Wraps any ChatAgent. If the primary's response is invalid, retry with a
    corrective prompt tailored to the failure mode. Fallback calls never think."""

    def __init__(
        self,
        primary: ChatAgent,
        max_retries: int = 1,
        name: str | None = None,
    ):
        self.primary = primary
        self.name = name or primary.name
        self.max_retries = max_retries
        self.log_interactions = True
        self.last_interaction: list[dict] = []

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        action = self.primary.act(observation, valid_actions, player_idx)
        attempts: list[dict] = list(self.primary.last_interaction)

        retries_left = self.max_retries
        while action not in valid_actions and retries_left > 0:
            reason = self._classify(attempts[-1])
            attempts.append(self._fallback(observation, valid_actions, attempts[-1], reason))
            try:
                action = int(attempts[-1]["response"])
            except (ValueError, TypeError):
                action = -1
            retries_left -= 1

        self.last_interaction = attempts
        return action

    @staticmethod
    def _classify(prev: dict) -> str:
        if prev.get("thinking_truncated"):
            return "thinking_truncated"
        try:
            int(prev["response"])
        except (ValueError, TypeError):
            return "not_integer"
        return "invalid_cell"

    def _fallback(self, observation: str, valid_actions: list[int], prev: dict, reason: str) -> dict:
        prompt = self._build_prompt(observation, valid_actions, prev, reason)
        messages = [self.primary.messages[0], {"role": "user", "content": prompt}]
        thinking, response, tokens = self.primary.generate(messages, enable_thinking=False)
        return {
            "agent": f"{self.primary.name} (fallback)",
            "invalid_reason": reason,
            "prompt": prompt,
            "thinking": thinking,
            "response": response,
            "tokens": tokens,
        }

    def _build_prompt(self, observation: str, valid_actions: list[int], prev: dict, reason: str) -> str:
        valid_str = f" Valid moves: {valid_actions}." if self.primary.pass_valid_actions else ""
        if reason == "thinking_truncated":
            return (
                f"Analysis:\n{prev.get('thinking', '')}\n"
                f"Note: reasoning was cut off. Give your final answer now.\n\n"
                f"Respond with a single integer only, no explanation."
                f"State:\n{observation}. {valid_str}"
            )
        if reason == "not_integer":
            return (
                f"State:\n{observation}\n\n"
                f"Your previous response {prev['response']!r} was not a single integer. "
                f"Respond with a single integer only, no explanation. {valid_str}"
            )
        return (
            f"State:\n{observation}\n\n"
            f"{prev['response']!r} is not an available cell. "
            f"Respond with a single integer only, no explanation. {valid_str}"
        )
