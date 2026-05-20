from typing import Optional
import os as _os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_QWEN_INSTRUCT = _os.environ.get(
    "QWEN_INSTRUCT",
    "Given a citizen complaint in Russian to a Russian city administration, "
    "retrieve the most relevant department/subdepartment description.",
)

_QWEN_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class DecoderCrossEncoder:
    def __init__(
        self,
        model_name: str,
        max_length: int = 1024,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        instruct: str = _QWEN_INSTRUCT,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.instruct = instruct
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if torch_dtype is None:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        self.config = self.model.config

        self._prefix_tokens = self.tokenizer(_QWEN_PREFIX, add_special_tokens=False).input_ids
        self._suffix_tokens = self.tokenizer(_QWEN_SUFFIX, add_special_tokens=False).input_ids
        self._content_max = max_length - len(self._prefix_tokens) - len(self._suffix_tokens)

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def _build_input_ids(self, query: str, document: str) -> list[int]:
        content = f"<Instruct>: {self.instruct}\n<Query>: {query}\n<Document>: {document}"
        content_ids = self.tokenizer(
            content, add_special_tokens=False, truncation=True, max_length=self._content_max,
        ).input_ids
        return self._prefix_tokens + content_ids + self._suffix_tokens

    @torch.no_grad()
    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 4,
        activation_fn=None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        all_scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            input_ids_list = [self._build_input_ids(q, d) for q, d in batch]
            max_len = max(len(x) for x in input_ids_list)
            pad_id = self.tokenizer.pad_token_id
            padded, attn = [], []
            for ids in input_ids_list:
                pad_count = max_len - len(ids)
                padded.append([pad_id] * pad_count + ids)
                attn.append([0] * pad_count + [1] * len(ids))
            input_ids = torch.tensor(padded, device=self.device)
            attention_mask = torch.tensor(attn, device=self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_token_logits = outputs.logits[:, -1, :]
            yes_logits = last_token_logits[:, self.yes_token_id]
            no_logits = last_token_logits[:, self.no_token_id]
            scores = (yes_logits - no_logits).float().cpu().numpy()
            if activation_fn is not None:
                scores = activation_fn(torch.from_numpy(scores)).numpy()
            all_scores.extend(scores.tolist())
        return np.array(all_scores, dtype=np.float32)
