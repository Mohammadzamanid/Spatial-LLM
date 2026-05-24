"""
src/data/tokenizer.py
Wraps a HuggingFace tokenizer and adds spatial-context injection.
Spatial context is prepended as a structured string so the LLM sees
coordinate information as part of the prompt.
"""

from transformers import AutoTokenizer


SPATIAL_PROMPT_TEMPLATE = (
    "[SPATIAL CONTEXT] Location: {lat:.6f}°N, {lon:.6f}°E\n"
    "[QUESTION] {question}\n"
    "[ANSWER]"
)


class SpatialTokenizer:
    """
    Wraps HuggingFace tokenizer with spatial prompt injection.
    Coordinates are embedded by the CoordinateEmbedder model, but
    we also include them as text for the text pathway as a fallback.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def encode_spatial(
        self,
        question: str,
        lat: float,
        lon: float,
        answer: str | None = None,
    ) -> dict:
        """
        Build a tokenized input with spatial context prepended.
        If answer is provided, it is appended for teacher-forcing during training.
        """
        prompt = SPATIAL_PROMPT_TEMPLATE.format(
            lat=lat, lon=lon, question=question
        )
        if answer is not None:
            full_text = prompt + " " + answer
        else:
            full_text = prompt

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if answer is not None:
            # Mask prompt tokens in labels (only supervise the answer)
            prompt_ids = self.tokenizer(
                prompt, return_tensors="pt"
            )["input_ids"]
            prompt_len = prompt_ids.shape[1]
            labels = encoded["input_ids"].clone()
            labels[0, :prompt_len] = -100  # ignore_index
            encoded["labels"] = labels

        return {k: v.squeeze(0) for k, v in encoded.items()}

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
