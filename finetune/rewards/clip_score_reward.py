from __future__ import annotations

from typing import Any, List

import torch

from .base import RewardModel


class CLIPScoreRewardModel(RewardModel):
    """CLIPScore-like reward using CLIP text-image cosine similarity."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "openai/clip-vit-large-patch14",
    ):
        self.name = "clip_score"
        self.device = device
        self.model_name = model_name

        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, prompts: List[str], images: List[Any]) -> torch.Tensor:
        if len(prompts) != len(images):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
            )
        if len(images) == 0:
            return torch.empty(0, dtype=torch.float32)

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_features = self._as_feature_tensor(image_features, "image_features")
        text_features = self._as_feature_tensor(text_features, "text_features")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (image_features * text_features).sum(dim=-1)
        return scores.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _as_feature_tensor(output: Any, name: str) -> torch.Tensor:
        if torch.is_tensor(output):
            return output

        # Some transformers versions return model output dataclasses.
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        if isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]

        raise TypeError(f"Unexpected {name} type: {type(output)}")
