from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from typing import Any, List
from urllib import request

import torch

from .base import RewardModel


class HPSv3RewardModel(RewardModel):
    """HPSv3 reward model wrapper for text-to-image scoring."""

    def __init__(self, device: str = "cuda", service_url: str = ""):
        self.name = "hpsv3"
        self.device = device
        self.service_url = service_url.strip()
        self._inferencer = None
        if not self.service_url:
            try:
                from hpsv3 import HPSv3RewardInferencer
            except ImportError as e:
                raise ImportError(
                    "hpsv3 is not installed locally and no reward service_url is set. "
                    "Install hpsv3 or pass --reward_service_url."
                ) from e
            self._inferencer = HPSv3RewardInferencer(device=device)

    def score_batch(self, prompts: List[str], images: List[Any]) -> torch.Tensor:
        if self.service_url:
            return self._score_batch_remote(prompts=prompts, images=images)
        return self._score_batch_local(prompts=prompts, images=images)

    def _score_batch_local(self, prompts: List[str], images: List[Any]) -> torch.Tensor:
        if len(prompts) != len(images):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
            )
        if len(images) == 0:
            return torch.empty(0, dtype=torch.float32)

        with tempfile.TemporaryDirectory(prefix="hpsv3_reward_") as tmpdir:
            image_paths: List[str] = []
            for idx, image in enumerate(images):
                if image is None:
                    raise ValueError("hpsv3 reward received None image.")
                path = os.path.join(tmpdir, f"sample_{idx:06d}.png")
                image.save(path)
                image_paths.append(path)

            raw_scores = self._call_hpsv3(prompts=prompts, image_paths=image_paths)

        values = [self._to_scalar(item) for item in raw_scores]
        return torch.tensor(values, dtype=torch.float32)

    def _score_batch_remote(self, prompts: List[str], images: List[Any]) -> torch.Tensor:
        if len(prompts) != len(images):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
            )
        encoded_images = []
        for image in images:
            if image is None:
                raise ValueError("hpsv3 reward received None image.")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        payload = {
            "prompts": prompts,
            "images_base64": encoded_images,
        }
        endpoint = self.service_url.rstrip("/") + "/score"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"hpsv3 reward service error: HTTP {resp.status}")
            raw = json.loads(resp.read().decode("utf-8"))
        scores = raw.get("scores", None)
        if scores is None:
            raise RuntimeError(f"Malformed hpsv3 service response: {raw}")
        values = [self._to_scalar(item) for item in scores]
        return torch.tensor(values, dtype=torch.float32)

    def _call_hpsv3(self, prompts: List[str], image_paths: List[str]):
        # Different hpsv3 versions expose slightly different signatures.
        if hasattr(self._inferencer, "reward"):
            fn = self._inferencer.reward
        elif hasattr(self._inferencer, "score"):
            fn = self._inferencer.score
        else:
            raise AttributeError("hpsv3 inferencer has neither `reward` nor `score`.")

        errors = []
        call_patterns = [
            lambda: fn(image_paths, prompts),
            lambda: fn(prompts, image_paths=image_paths),
            lambda: fn(prompts, image_paths),
        ]
        for call in call_patterns:
            try:
                return call()
            except Exception as e:  # pragma: no cover - compatibility fallback
                errors.append(f"{type(e).__name__}: {e}")
        raise RuntimeError(
            "Failed to call hpsv3 inferencer with known signatures. "
            + " | ".join(errors)
        )

    @staticmethod
    def _to_scalar(score_item: Any) -> float:
        # Common hpsv3 output format: [mu, sigma]
        if isinstance(score_item, (list, tuple)) and score_item:
            return HPSv3RewardModel._to_scalar(score_item[0])
        if torch.is_tensor(score_item):
            return float(score_item.detach().cpu().flatten()[0].item())
        if isinstance(score_item, dict):
            for key in ("mu", "score", "reward"):
                if key in score_item:
                    return float(score_item[key])
        return float(score_item)
