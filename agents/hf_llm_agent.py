import os

from agents.chat_agent import ChatAgent
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

QWEN_THINK_END_TOKEN = 151668  # </think>


class HFLLMAgent(ChatAgent):

    def __init__(
        self,
        model_name: str,
        base_prompt: str,
        enable_thinking: bool = False,
        pass_valid_actions: bool = True,
        stateless: bool = True,
        max_new_tokens: int = 256,
        verbose: bool = False,
        log_interactions: bool = True,
        pieces: tuple[str, str] = ("X", "O"),
    ):
        self.model_name = model_name
        self.name = model_name
        self.enable_thinking = enable_thinking
        self.pass_valid_actions = pass_valid_actions
        self.stateless = stateless
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.log_interactions = log_interactions
        self.pieces = tuple(pieces)
        self.last_interaction: list[dict] = []

        token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            token=token,
        )
        self.streamer = TextStreamer(self.tokenizer) if verbose else None

        self.messages = [{"role": "system", "content": base_prompt}]

    def generate(self, messages: list[dict], enable_thinking: bool | None = None) -> tuple[str, str, dict]:
        """Run the model on explicit messages. Returns (thinking, response, tokens)."""
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=et,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = int(model_inputs.input_ids.shape[1])
        generated_ids = self.model.generate(
            **model_inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][input_tokens:].tolist()
        output_tokens = len(output_ids)

        index = 0
        if et:
            try:
                index = len(output_ids) - output_ids[::-1].index(QWEN_THINK_END_TOKEN)
            except ValueError:
                index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip() if et else ""
        return thinking, content, {"input": input_tokens, "output": output_tokens}

