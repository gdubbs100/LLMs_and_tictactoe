from __future__ import annotations

import ollama

from agents.chat_agent import ChatAgent


class OllamaUnavailableError(Exception):
    """Raised when Ollama returns a 500 error."""


class OllamaAgent(ChatAgent):

    def __init__(
        self,
        model_name: str,
        base_prompt: str,
        enable_thinking: bool = False,
        pass_valid_actions: bool = True,
        stateless: bool = True,
        log_interactions: bool = True,
        pieces: tuple[str, str] = ("X", "O"),
    ):
        self.model_name = model_name
        self.name = model_name
        self.enable_thinking = enable_thinking
        self.pass_valid_actions = pass_valid_actions
        self.stateless = stateless
        self.log_interactions = log_interactions
        self.pieces = tuple(pieces)
        self.last_interaction: list[dict] = []
        self.messages = [{"role": "system", "content": base_prompt}]

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> tuple[str, str, dict]:
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        try:
            resp = ollama.chat(model=self.model_name, messages=messages, think=et)
        except ollama.ResponseError as e:
            if e.status_code == 500:
                raise OllamaUnavailableError(
                    f"Ollama 500 for {self.model_name}: {e}"
                ) from e
            raise
        return (
            resp.message.thinking or "",
            resp.message.content or "",
            {"input": resp.prompt_eval_count or 0, "output": resp.eval_count or 0},
        )
