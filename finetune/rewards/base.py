from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import torch


class RewardModel(ABC):
    """Base interface for text-to-image reward models."""

    name: str

    @abstractmethod
    def score_batch(self, prompts: List[str], images: List[Any]) -> torch.Tensor:
        """
        Return per-sample reward scores with shape [batch_size].

        Args:
            prompts: Text prompts (one per image).
            images: Decoded images (typically PIL.Image).
        """
        raise NotImplementedError
