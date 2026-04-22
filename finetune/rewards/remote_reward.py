from __future__ import annotations

import base64
import io
from typing import Any, List

import requests
import torch

from .base import RewardModel


class RemoteRewardModel(RewardModel):
    """Generic remote reward client for the root reward service /score API."""

    def __init__(
        self,
        name: str,
        service_url: str,
        timeout: float = 1200.0,
        response_text: str = "",
    ):
        self.name = name
        self.service_url = self._normalize_service_url(service_url)
        self.timeout = float(timeout)
        self.response_text = response_text

    def score_batch(self, prompts: List[str], images: List[Any], metadatas: List[dict] = None) -> torch.Tensor:
        if metadatas is None:
            metadatas = [{} for _ in range(len(images))]

        if len(prompts) != len(images) or len(prompts) != len(metadatas):
            raise ValueError(
                f"prompts/images/metadatas length mismatch: {len(prompts)} vs {len(images)} vs {len(metadatas)}"
            )
        if len(images) == 0:
            return torch.empty(0, dtype=torch.float32)

        scores = []
        for prompt, image, metadata in zip(prompts, images, metadatas):
            if image is None:
                raise ValueError(f"{self.name} reward received None image.")
            payload = {
                "prompt": prompt,
                "response": self.response_text,
                "generated_images": [self._encode_image(image)],
                "metadata": metadata,
            }
            response = requests.post(
                self.service_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            score = data.get("score")
            if not isinstance(score, (int, float)):
                raise RuntimeError(
                    f"Malformed reward service response from {self.service_url}: {data}"
                )
            scores.append(float(score))
        return torch.tensor(scores, dtype=torch.float32)

    @staticmethod
    def _normalize_service_url(url: str) -> str:
        normalized = str(url or "").strip()
        if not normalized:
            raise ValueError("reward service_url must not be empty for RemoteRewardModel.")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        normalized = normalized.rstrip("/")
        if not normalized.endswith("/score"):
            normalized = f"{normalized}/score"
        return normalized

    @staticmethod
    def _encode_image(image: Any) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
