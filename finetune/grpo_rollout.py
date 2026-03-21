"""
GRPO rollout for GLM-Image.

This module implements the sampling (rollout) phase of Group Relative Policy
Optimization for the AR encoder of GLM-Image.  It is designed as a standalone
building block; reward scoring and the GRPO loss will be added later.

Key design decisions
--------------------
* Only the AR encoder produces stochastic rollouts (via nucleus/top-k sampling
  with `do_sample=True`).  The diffusion decoder is deterministic for a given
  set of prior tokens when the decoder seed is fixed.
* For every prompt in a group we:
  1. Run the AR model `G` times with different random seeds to get `G` different
     discrete-token sequences.
  2. Decode each token sequence through the frozen DiT + VAE pipeline with a
     *fixed* decoder seed (shared within the group) so that visual differences
     are attributable solely to the AR encoder.
* The batch sampler ensures that all `G` rollouts for a single prompt land on
  the same device / micro-batch so that group-level statistics (advantages) can
  be computed locally without cross-device communication.

Data format
-----------
Input JSONL -- one JSON object per line::

    {"prompt": "a cat sitting on ...", "metadata": {"height": 1024, "width": 1024}}

`metadata.height` and `metadata.width` must be divisible by 32.

Usage example (rollout only)
----------------------------
>>> from grpo_rollout import GRPORolloutConfig, GRPORolloutEngine
>>> cfg = GRPORolloutConfig(
...     model_path="/data/GLM-Image",
...     prompt_jsonl="prompts.jsonl",
...     group_size=4,
... )
>>> engine = GRPORolloutEngine(cfg)
>>> for group in engine.rollout():
...     # group is a GRPOGroup with .prompt, .images, .token_ids, ...
...     pass
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from rewards import build_reward_models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPORolloutConfig:
    """All knobs for the GRPO rollout phase."""

    # Model
    model_path: str = "zai-org/GLM-Image"
    torch_dtype: str = "bfloat16"

    # LoRA (optional -- load a LoRA checkpoint on top of the base model)
    lora_path: Optional[str] = None

    # Prompt dataset
    prompt_jsonl: str = "prompts.jsonl"
    default_height: int = 1024
    default_width: int = 1024

    # GRPO sampling
    group_size: int = 4  # G: number of rollouts per prompt
    ar_temperature: float = 0.9
    ar_top_p: float = 0.75

    # Decoder
    num_inference_steps: int = 50
    guidance_scale: float = 1.5
    decoder_seed: int = 42  # fixed across the group

    # Batching
    # `prompts_per_batch` is the number of *distinct prompts* processed together.
    # Total AR forward calls per batch = prompts_per_batch * group_size
    # Total decoder calls per batch   = prompts_per_batch * group_size
    prompts_per_batch: int = 1
    num_workers: int = 2

    # Device / precision
    device: str = "cuda"

    # Misc
    seed: int = 0
    output_dir: str = "./outputs/grpo"
    skip_decode: bool = False  # If True, skip the DiT+VAE decode (useful for
    # debugging or when only token-level rewards are
    # needed).
    reward_models: str = ""  # Comma-separated reward model names, e.g. "hpsv3"
    reward_device: Optional[str] = None
    reward_service_url: str = ""
    reward_clip_model: str = "openai/clip-vit-large-patch14"

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.torch_dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Prompt dataset
# ---------------------------------------------------------------------------


class PromptDataset(Dataset):
    """
    Reads a JSONL file of prompts with optional height/width metadata.

    Each line: {"prompt": "...", "metadata": {"height": H, "width": W}}
    """

    def __init__(
        self,
        jsonl_path: str,
        default_height: int = 1024,
        default_width: int = 1024,
    ):
        super().__init__()
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "prompt" not in obj:
                    raise ValueError(
                        f"Line {line_no} in {jsonl_path} missing 'prompt' field"
                    )
                meta = obj.get("metadata", {})
                h = meta.get("height", default_height)
                w = meta.get("width", default_width)
                if h % 32 != 0 or w % 32 != 0:
                    raise ValueError(
                        f"Line {line_no}: height={h} and width={w} must be "
                        f"divisible by 32"
                    )
                self.items.append({"prompt": obj["prompt"], "height": h, "width": w})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Group-aware batch sampler
# ---------------------------------------------------------------------------


class GroupBatchSampler(Sampler):
    """
    Yields batches of prompt indices such that each batch contains exactly
    `prompts_per_batch` prompts.  The caller is responsible for expanding
    each prompt index into `group_size` rollout instances.

    Within an epoch the prompts are visited in a fixed (optionally shuffled)
    order.  This sampler is intentionally simple -- it does *not* try to
    bucket by resolution because all rollouts within a group share the same
    resolution, and cross-group padding is expected to be minor when
    `prompts_per_batch` is small (typically 1-2).
    """

    def __init__(
        self,
        dataset_size: int,
        prompts_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset_size = dataset_size
        self.prompts_per_batch = prompts_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))

        batch: List[int] = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.prompts_per_batch:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.prompts_per_batch
        return (self.dataset_size + self.prompts_per_batch - 1) // self.prompts_per_batch


# ---------------------------------------------------------------------------
# Data structures for rollout outputs
# ---------------------------------------------------------------------------


@dataclass
class RolloutSample:
    """One AR rollout for a single prompt."""

    # Discrete token ids produced by the AR model *before* upsampling.
    # Shape: [num_tokens_d32]  (i.e. the "small" representation from the AR)
    token_ids_d32: torch.Tensor

    # Upsampled token ids fed to the DiT decoder.
    # Shape: [num_tokens_upsampled]
    token_ids_upsampled: torch.Tensor

    # Log-probabilities of each generated token under the policy at sampling
    # time.  Shape: [num_generated_tokens]
    log_probs: torch.Tensor

    # The decoded image (PIL or tensor).  None when `skip_decode=True`.
    image: Any = None

    # The random seed used for the AR model on this rollout.
    ar_seed: int = 0

    # Optional reward details for this sample.
    reward: Optional[float] = None
    reward_breakdown: Optional[Dict[str, float]] = None


@dataclass
class GRPOGroup:
    """All rollout samples for one prompt (a "group" in GRPO terms)."""

    prompt: str
    height: int
    width: int
    samples: List[RolloutSample] = field(default_factory=list)

    # Placeholder for reward scores (filled in later by the reward module).
    rewards: Optional[torch.Tensor] = None  # shape [G]

    # Placeholder for advantages (filled in later by the GRPO loss module).
    advantages: Optional[torch.Tensor] = None  # shape [G]


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------


class GRPORolloutEngine:
    """
    Orchestrates the rollout phase:
      1. Load model + pipeline components.
      2. Iterate over prompt batches.
      3. For each prompt, run `group_size` AR rollouts.
      4. Optionally decode each rollout through the DiT + VAE.
    """

    def __init__(self, config: GRPORolloutConfig):
        self.config = config
        self._setup_logging()
        self._load_model()
        self._load_reward_models()
        self._build_dataloader()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> None:
        """Load the full GLM-Image pipeline and optionally apply LoRA."""
        from diffusers import GlmImagePipeline
        from transformers import GlmImageForConditionalGeneration

        cfg = self.config
        self.logger.info(f"Loading pipeline from {cfg.model_path} ...")

        self.pipe = GlmImagePipeline.from_pretrained(
            cfg.model_path,
            torch_dtype=cfg.dtype,
        )
        self.ar_model = self.pipe.vision_language_encoder
        self.processor = self.pipe.processor

        # Optionally load LoRA
        if cfg.lora_path:
            self.logger.info(f"Loading LoRA from {cfg.lora_path}")
            from peft import PeftModel

            model_for_lora = (
                os.path.join(cfg.model_path, "vision_language_encoder")
                if os.path.exists(os.path.join(cfg.model_path, "vision_language_encoder"))
                else cfg.model_path
            )
            base_model = GlmImageForConditionalGeneration.from_pretrained(
                model_for_lora,
                torch_dtype=cfg.dtype,
                device_map=None,
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, cfg.lora_path)
            merged = peft_model.merge_and_unload()

            if self.pipe is not None:
                self.pipe.vision_language_encoder = merged
            self.ar_model = merged

        self.pipe = self.pipe.to(cfg.device)
        self.ar_model = self.ar_model.to(cfg.device)
        self.ar_model.eval()  # AR model in eval mode for sampling (no dropout)
        self.logger.info("Pipeline loaded.")

    def _build_dataloader(self) -> None:
        cfg = self.config
        self.prompt_dataset = PromptDataset(
            jsonl_path=cfg.prompt_jsonl,
            default_height=cfg.default_height,
            default_width=cfg.default_width,
        )
        self.batch_sampler = GroupBatchSampler(
            dataset_size=len(self.prompt_dataset),
            prompts_per_batch=cfg.prompts_per_batch,
            shuffle=True,
            seed=cfg.seed,
        )
        # We use a trivial collate that just returns a list of dicts --
        # actual tensor batching happens inside the rollout logic.
        self.dataloader = DataLoader(
            self.prompt_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=cfg.num_workers,
            collate_fn=_list_collate,
            pin_memory=False,
        )
        self.logger.info(
            f"Prompt dataset: {len(self.prompt_dataset)} prompts, "
            f"{len(self.batch_sampler)} batches of "
            f"{cfg.prompts_per_batch} prompts x {cfg.group_size} rollouts"
        )

    def _load_reward_models(self) -> None:
        cfg = self.config
        reward_names = [name.strip() for name in cfg.reward_models.split(",") if name.strip()]
        reward_device = cfg.reward_device or cfg.device
        self.reward_models = build_reward_models(
            reward_names=reward_names,
            device=reward_device,
            service_url=cfg.reward_service_url,
            clip_model_name=cfg.reward_clip_model,
            logger=self.logger,
        )
        if self.reward_models:
            self.logger.info(
                "Loaded reward models: %s",
                ", ".join(model.name for model in self.reward_models),
            )

    def _score_group_rewards(self, group: GRPOGroup) -> None:
        if not self.reward_models:
            return

        prompts = [group.prompt] * len(group.samples)
        images = [sample.image for sample in group.samples]
        if any(image is None for image in images):
            raise ValueError(
                "Reward scoring requires decoded images. "
                "Disable --skip_decode when using --reward_models."
            )

        per_model_scores: List[torch.Tensor] = []
        reward_names: List[str] = []
        for reward_model in self.reward_models:
            scores = reward_model.score_batch(prompts=prompts, images=images)
            if scores.numel() != len(group.samples):
                raise ValueError(
                    f"Reward model '{reward_model.name}' returned {scores.numel()} "
                    f"scores for {len(group.samples)} samples."
                )
            per_model_scores.append(scores.to(dtype=torch.float32, device="cpu"))
            reward_names.append(reward_model.name)

        score_matrix = torch.stack(per_model_scores, dim=0)  # [num_models, G]
        group.rewards = score_matrix.mean(dim=0)
        for s_idx, sample in enumerate(group.samples):
            sample.reward = float(group.rewards[s_idx].item())
            sample.reward_breakdown = {
                reward_names[m_idx]: float(score_matrix[m_idx, s_idx].item())
                for m_idx in range(len(reward_names))
            }

    # ------------------------------------------------------------------
    # Core rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self) -> Iterator[List[GRPOGroup]]:
        """
        Yield one list of ``GRPOGroup`` per batch of prompts.

        Each ``GRPOGroup`` contains ``group_size`` ``RolloutSample`` instances
        (one per AR rollout).  Images are decoded only if ``skip_decode`` is
        False.
        """
        cfg = self.config
        device = torch.device(cfg.device)
        base_seed = cfg.seed

        for batch_idx, prompt_batch in enumerate(self.dataloader):
            groups: List[GRPOGroup] = []

            for item in prompt_batch:
                prompt = item["prompt"]
                height = item["height"]
                width = item["width"]
                group = GRPOGroup(prompt=prompt, height=height, width=width)

                # Fixed decoder seed for this prompt group
                decoder_generator = torch.Generator(device=device)
                decoder_generator.manual_seed(cfg.decoder_seed)

                # ----- AR rollouts -----
                for g_idx in range(cfg.group_size):
                    ar_seed = base_seed + batch_idx * cfg.group_size + g_idx

                    # Sample token sequence from the AR model
                    token_ids_d32, token_ids_up, log_probs = (
                        self._ar_sample_single(
                            prompt=prompt,
                            height=height,
                            width=width,
                            seed=ar_seed,
                            device=device,
                        )
                    )

                    # ----- Decoder (optional) -----
                    pil_image = None
                    if not cfg.skip_decode:
                        pil_image = self._decode_single(
                            prompt=prompt,
                            height=height,
                            width=width,
                            prior_token_ids=token_ids_up.unsqueeze(0),
                            decoder_generator=decoder_generator,
                        )
                        # Reset the decoder generator so every rollout in the
                        # group uses the *same* decoder noise.
                        decoder_generator.manual_seed(cfg.decoder_seed)

                    sample = RolloutSample(
                        token_ids_d32=token_ids_d32,
                        token_ids_upsampled=token_ids_up,
                        log_probs=log_probs,
                        image=pil_image,
                        ar_seed=ar_seed,
                    )
                    group.samples.append(sample)

                self._score_group_rewards(group)
                groups.append(group)

            yield groups

    # ------------------------------------------------------------------
    # AR sampling with log-prob collection
    # ------------------------------------------------------------------

    def _ar_sample_single(
        self,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one AR rollout for a single prompt.

        Returns
        -------
        token_ids_d32 : Tensor [num_large_tokens]
            The large-image token ids at the original d32 resolution
            (before upsampling).
        token_ids_upsampled : Tensor [num_upsampled_tokens]
            2x upsampled token ids, ready for the DiT decoder.
        log_probs : Tensor [num_generated_tokens]
            Per-token log probabilities under the current policy for the
            *entire* generated sequence (small + large + EOS).
        """
        cfg = self.config

        # Build the processor inputs (same logic as pipeline.generate_prior_tokens)
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        image_grid_thw = inputs["image_grid_thw"]
        max_new_tokens, large_image_offset, token_h, token_w = (
            GRPORolloutEngine._compute_generation_params(image_grid_thw)
        )

        input_length = inputs["input_ids"].shape[-1]

        # Set AR random seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # --- Generate with output_scores to capture logits ---
        outputs = self.ar_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=cfg.ar_temperature,
            top_p=cfg.ar_top_p,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_ids = outputs.sequences[0, input_length:]  # [num_generated]
        scores = outputs.scores  # tuple of [1, vocab_size] per step

        # Compute per-token log-probs
        log_probs_list = []
        for t, score in enumerate(scores):
            # score: [1, vocab_size]  (raw logits before sampling)
            # Apply the same temperature used at sampling for consistency
            logp = F.log_softmax(score[0] / cfg.ar_temperature, dim=-1)
            token_id = generated_ids[t]
            log_probs_list.append(logp[token_id])
        log_probs = torch.stack(log_probs_list)  # [num_generated]

        # Extract the large-image tokens (d32 resolution)
        num_large_tokens = token_h * token_w
        large_start = large_image_offset
        large_end = large_start + num_large_tokens
        token_ids_d32 = generated_ids[large_start:large_end]

        # Upsample to 2x for the DiT decoder
        token_ids_up = token_ids_d32.view(1, 1, token_h, token_w).float()
        token_ids_up = F.interpolate(token_ids_up, scale_factor=2, mode="nearest")
        token_ids_up = token_ids_up.to(dtype=torch.long).view(-1)

        return token_ids_d32, token_ids_up, log_probs

    # ------------------------------------------------------------------
    # Deterministic decoder
    # ------------------------------------------------------------------

    def _decode_single(
        self,
        prompt: str,
        height: int,
        width: int,
        prior_token_ids: torch.Tensor,
        decoder_generator: torch.Generator,
    ):
        """
        Decode a single set of prior tokens through the DiT + VAE.

        The `decoder_generator` should be seeded *before* calling this method
        and reset *after* so that all calls within a group are deterministic.
        """
        cfg = self.config

        if self.pipe is None:
            raise RuntimeError(
                "Decoder not available: diffusers GLM-Image pipeline could not be loaded. "
                "Run with --skip_decode (or install a diffusers version with GLM-Image)."
            )

        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            prior_token_ids=prior_token_ids,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=decoder_generator,
        )
        return result.images[0]

    # ------------------------------------------------------------------
    # Helpers (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_generation_params(
        image_grid_thw: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """
        Compute AR generation parameters for T2I mode.

        Returns (max_new_tokens, large_image_offset, token_h, token_w).
        """
        grid_sizes = []
        grid_hw = []
        for i in range(image_grid_thw.shape[0]):
            _, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))

        # T2I: grids are [large, small]; generation order is small -> large -> EOS
        total_tokens = sum(grid_sizes)
        max_new_tokens = total_tokens + 1  # +1 for EOS
        large_image_offset = sum(grid_sizes[1:])  # skip small tokens
        token_h, token_w = grid_hw[0]
        return max_new_tokens, large_image_offset, token_h, token_w


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------


def _list_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity collate -- just return the list of dicts."""
    return batch


# ---------------------------------------------------------------------------
# CLI entry point (for testing / standalone rollout)
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GRPO rollout for GLM-Image")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_jsonl", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--prompts_per_batch", type=int, default=1)
    parser.add_argument("--ar_temperature", type=float, default=0.9)
    parser.add_argument("--ar_top_p", type=float, default=0.75)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--decoder_seed", type=int, default=42)
    parser.add_argument("--default_height", type=int, default=1024)
    parser.add_argument("--default_width", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo")
    parser.add_argument("--skip_decode", action="store_true")
    parser.add_argument(
        "--reward_models",
        type=str,
        default="",
        help="Comma-separated reward models (supports: hpsv3, clip_score)",
    )
    parser.add_argument("--reward_device", type=str, default=None)
    parser.add_argument(
        "--reward_service_url",
        type=str,
        default="",
        help="Optional reward service URL, e.g. http://127.0.0.1:8009",
    )
    parser.add_argument(
        "--reward_clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model id used by clip_score reward",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = GRPORolloutConfig(
        model_path=args.model_path,
        prompt_jsonl=args.prompt_jsonl,
        lora_path=args.lora_path,
        group_size=args.group_size,
        prompts_per_batch=args.prompts_per_batch,
        ar_temperature=args.ar_temperature,
        ar_top_p=args.ar_top_p,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        decoder_seed=args.decoder_seed,
        default_height=args.default_height,
        default_width=args.default_width,
        output_dir=args.output_dir,
        skip_decode=args.skip_decode,
        reward_models=args.reward_models,
        reward_device=args.reward_device,
        reward_service_url=args.reward_service_url,
        reward_clip_model=args.reward_clip_model,
        seed=args.seed,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    engine = GRPORolloutEngine(config)

    total_groups = 0
    total_samples = 0
    for batch_groups in engine.rollout():
        for group in batch_groups:
            total_groups += 1
            total_samples += len(group.samples)
            logging.info(
                f"Group {total_groups}: prompt={group.prompt[:60]!r}... "
                f"({group.height}x{group.width}), "
                f"{len(group.samples)} samples"
            )
            if group.rewards is not None:
                for s_idx, sample in enumerate(group.samples):
                    breakdown = sample.reward_breakdown or {}
                    breakdown_text = ", ".join(
                        f"{name}={value:.6f}" for name, value in breakdown.items()
                    )
                    logging.info(
                        f"  Reward sample_{s_idx}: total={sample.reward:.6f}"
                        + (f" ({breakdown_text})" if breakdown_text else "")
                    )

            # Save images if decoded
            if not config.skip_decode:
                group_dir = os.path.join(config.output_dir, f"group_{total_groups:04d}")
                os.makedirs(group_dir, exist_ok=True)
                for s_idx, sample in enumerate(group.samples):
                    if sample.image is not None:
                        img_path = os.path.join(group_dir, f"sample_{s_idx}.png")
                        sample.image.save(img_path)

    logging.info(
        f"Rollout complete: {total_groups} groups, {total_samples} total samples"
    )


if __name__ == "__main__":
    main()
