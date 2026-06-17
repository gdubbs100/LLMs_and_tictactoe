from __future__ import annotations

from abc import abstractmethod

from agents.base import LLMAgent


class ChatAgent(LLMAgent):
    """Base for chat-style LLM agents. Subclasses implement generate(); act() is shared."""

    name: str
    enable_thinking: bool
    pass_valid_actions: bool
    stateless: bool
    messages: list[dict]
    log_interactions: bool
    last_interaction: list[dict]
    pieces: tuple[str, str] = ("X", "O")

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> tuple[str, str, dict]:
        """Return (thinking, response, {"input": N, "output": N})."""
        ...

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        user_content = (
            f'You are playing as "{self.pieces[player_idx]}". '
            f'Your opponent is "{self.pieces[1 - player_idx]}".\n\n'
            f"State:\n{observation}"
        )
        if self.pass_valid_actions:
            user_content += f"\n\nValid moves:\n{valid_actions}"

        user_msg = {"role": "user", "content": user_content}

        if self.stateless:
            messages = [self.messages[0], user_msg]
        else:
            self.messages.append(user_msg)
            messages = self.messages

        thinking, content, tokens = self.generate(messages)

        thinking_truncated = False
        if self.enable_thinking and not thinking and content and content.strip():
            try:
                int(content)
            except (ValueError, TypeError):
                thinking = content + "</think>"
                content = ""
                thinking_truncated = True

        if not self.stateless:
            self.messages.append({"role": "assistant", "content": content})

        self.last_interaction = [{
            "agent": self.name,
            "prompt": user_content,
            "thinking": thinking,
            "response": content,
            "thinking_truncated": thinking_truncated,
            "tokens": tokens,
        }]

        try:
            return int(content)
        except (ValueError, TypeError):
            return -1
