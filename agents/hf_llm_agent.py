from agents.base import LLMAgent
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

QWEN_THINK_END_TOKEN = 151668  # </think>


class HFLLMAgent(LLMAgent):

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
    ):
        self.model_name = model_name
        self.name = model_name
        self.enable_thinking = enable_thinking
        self.pass_valid_actions = pass_valid_actions
        self.stateless = stateless
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.log_interactions = log_interactions
        self.last_interaction: list[dict] = []

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.streamer = TextStreamer(self.tokenizer) if verbose else None

        self.messages = [{"role": "system", "content": base_prompt}]

    def generate(self, messages: list[dict], enable_thinking: bool | None = None) -> tuple[str, str]:
        """Run the model on explicit messages. Returns (thinking, response)."""
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=et,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        index = 0
        if et:
            try:
                index = len(output_ids) - output_ids[::-1].index(QWEN_THINK_END_TOKEN)
            except ValueError:
                index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip() if et else ""
        return thinking, content

    def act(self, observation: str, valid_actions: list[int], player_idx: int) -> int:
        user_content = f"State:\n{observation}"
        if self.pass_valid_actions:
            user_content += f"\n\nValid moves:\n{valid_actions}"

        user_msg = {"role": "user", "content": user_content}

        if self.stateless:
            messages = [self.messages[0], user_msg]
        else:
            self.messages.append(user_msg)
            messages = self.messages

        thinking, content = self.generate(messages)

        # Thinking enabled but no </think> found and output isn't a valid answer:
        # the model ran out of tokens mid-think. Treat entire output as truncated thinking.
        thinking_truncated = False
        if self.enable_thinking and not thinking and content and content.strip():
            try:
                int(content)  # already a valid answer — leave it alone
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
        }]

        try:
            return int(content)
        except (ValueError, TypeError):
            return -1  # out-of-range sentinel — play.py treats as invalid move
