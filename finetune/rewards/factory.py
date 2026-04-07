from __future__ import annotations

import logging
from typing import List, Optional

from .base import RewardModel
from .clip_score_reward import CLIPScoreRewardModel
from .hpsv3_reward import HPSv3RewardModel
from .remote_reward import RemoteRewardModel


def build_reward_models(
    reward_names: List[str],
    device: str = "cuda",
    service_url: str = "",
    service_timeout: float = 1200.0,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    logger: Optional[logging.Logger] = None,
) -> List[RewardModel]:
    if logger is None:
        logger = logging.getLogger(__name__)

    models: List[RewardModel] = []
    for name in reward_names:
        key = name.lower()
        if service_url:
            models.append(
                RemoteRewardModel(
                    name=key,
                    service_url=service_url,
                    timeout=service_timeout,
                )
            )
            logger.info(
                "Initialized reward model '%s' via generic reward service_url=%s",
                key,
                service_url,
            )
        elif key == "hpsv3":
            models.append(HPSv3RewardModel(device=device, service_url=service_url))
        elif key in ("clip", "clip_score"):
            models.append(CLIPScoreRewardModel(device=device, model_name=clip_model_name))
        else:
            raise ValueError(
                f"Unknown reward model: '{name}'. Supported reward models: hpsv3, clip_score"
            )
        if not service_url:
            logger.info("Initialized reward model '%s' on device=%s", key, device)

    return models
