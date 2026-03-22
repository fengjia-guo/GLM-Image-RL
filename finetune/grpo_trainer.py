#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) trainer for GLM-Image.

This module implements the full GRPO training loop that connects:
  rollout (sampling) → reward scoring → advantage computation → policy update

Algorithm Overview (DeepSeek-R1 style GRPO)
--------------------------------------------
For each prompt q:
  1. Sample G outputs {o_1, ..., o_G} from the current policy π_θ
  2. Score each output with reward model(s) → {r_1, ..., r_G}
  3. Compute group-relative advantages:
       Â_i = (r_i - mean(r)) / (std(r) + ε)
  4. Update policy with clipped surrogate objective + KL penalty:
       L_GRPO = -1/G Σ_i [ min(ρ_i * Â_i, clip(ρ_i, 1-ε, 1+ε) * Â_i)
                           - β * D_KL(π_θ || π_ref) ]
     where ρ_i = π_θ(o_i|q) / π_old(o_i|q)

Key design for GLM-Image:
  - Only the AR encoder is trained (via LoRA); the DiT decoder is frozen.
  - The reference model π_ref is a frozen copy of the initial AR weights.
  - Log-probs are collected per-token during rollout, then used for the
    importance ratio ρ and KL computation.
  - The decoder seed is fixed within each group so visual differences are
    solely attributable to the AR encoder.

Distributed training:
  Uses HuggingFace Accelerate for multi-GPU / mixed-precision support.
  Launch with:
    accelerate launch grpo_trainer.py --model_path ... --prompt_jsonl ...

Usage (single GPU):
    python grpo_trainer.py \
        --model_path /data/GLM-Image \
        --prompt_jsonl prompts.jsonl \
        --reward_models clip_score \
        --num_epochs 3 \
        --group_size 4

Usage (multi-GPU via accelerate):
    accelerate launch [--num_processes N] grpo_trainer.py \
        --model_path /data/GLM-Image \
        --prompt_jsonl prompts.jsonl \
        --reward_models clip_score \
        --num_epochs 3 \
        --group_size 4
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from grpo_rollout import GRPOGroup, GRPORolloutConfig, GRPORolloutEngine, RolloutSample

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOTrainingConfig:
    """Full configuration for GRPO training."""

    # ── Model ──────────────────────────────────────────────────────────
    model_path: str = "zai-org/GLM-Image"
    torch_dtype: str = "bfloat16"

    # ── LoRA ───────────────────────────────────────────────────────────
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_up_proj", "down_proj",
        ]
    )
    resume_lora_path: Optional[str] = None  # Resume from existing LoRA ckpt

    # ── GRPO rollout ───────────────────────────────────────────────────
    prompt_jsonl: str = "prompts.jsonl"
    group_size: int = 4
    prompts_per_batch: int = 1
    ar_temperature: float = 0.9
    ar_top_p: float = 0.75
    num_inference_steps: int = 50
    guidance_scale: float = 1.5
    decoder_seed: int = 42
    default_height: int = 1024
    default_width: int = 1024
    skip_decode: bool = False

    # ── Reward ─────────────────────────────────────────────────────────
    reward_models: str = "clip_score"
    reward_device: Optional[str] = None
    reward_service_url: str = ""
    reward_clip_model: str = "openai/clip-vit-large-patch14"

    # ── GRPO hyper-parameters ──────────────────────────────────────────
    clip_eps: float = 0.2          # PPO-style clipping ε
    kl_coef: float = 0.01         # β coefficient for KL penalty
    advantage_eps: float = 1e-8    # ε for advantage normalisation
    num_epochs: int = 3            # Outer epochs over the prompt dataset
    num_inner_steps: int = 1       # PPO-style inner updates per rollout batch
    gradient_accumulation_steps: int = 1

    # ── Optimiser ──────────────────────────────────────────────────────
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    use_8bit_adam: bool = False

    # ── Gradient checkpointing & precision ─────────────────────────────
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # ── Logging & checkpointing ────────────────────────────────────────
    output_dir: str = "./outputs/grpo"
    logging_steps: int = 1
    save_steps: int = 50
    save_images: bool = True       # Save decoded images for debugging

    # Experiment tracking
    report_to: str = "tensorboard"  # "tensorboard", "wandb", "all", or "none"
    wandb_project: str = "glm-image-grpo"
    wandb_run_name: Optional[str] = None

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 0
    num_workers: int = 2

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.torch_dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Group Relative Policy Optimization advantage:
        Â_i = (r_i - mean(r)) / (std(r) + ε)
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Per-token KL divergence
# ---------------------------------------------------------------------------


def token_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate per-token KL(π_θ || π_ref) — Schulman k3 estimator.
    Only requires log-probs for the *sampled* tokens.
    """
    ratio = ref_log_probs - log_probs
    return torch.exp(ratio) - ratio - 1.0


# ---------------------------------------------------------------------------
# 3D RoPE position_ids for teacher-forcing forward pass
# ---------------------------------------------------------------------------


def build_position_ids_for_teacher_forcing(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build ``position_ids`` of shape ``[3, 1, seq_len]`` for 3D RoPE.

    The model's ``get_rope_index`` is designed for auto-regressive generation
    (incomplete image tokens). During teacher-forcing (all image tokens present),
    it produces a size mismatch. This function manually constructs the correct
    spatial position encoding following the same logic as ``finetune_lora.py``.

    Parameters
    ----------
    input_ids : Tensor [1, seq_len]
        Full token sequence (text prefix + generated image tokens).
    image_grid_thw : Tensor [num_grids, 3]
        Grid dimensions from ``apply_chat_template``. For T2I this is
        ``[[t_large, h_large, w_large], [t_small, h_small, w_small]]``.
    device : torch.device

    Returns
    -------
    position_ids : Tensor [3, 1, seq_len]
        (temporal, height, width) position encoding.
    """
    seq_len = input_ids.shape[-1]
    ids = input_ids[0]  # [seq_len]

    # Token ids for image boundaries
    IMAGE_START = 16384
    IMAGE_END = 16385

    position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long, device=device)

    temporal_list = []
    height_list = []
    width_list = []

    curr_pos = 0

    # Find <image_start>
    img_start_positions = (ids == IMAGE_START).nonzero(as_tuple=True)[0]

    if len(img_start_positions) > 0:
        img_start = img_start_positions[0].item()

        # 1. Text tokens before <image_start>
        text_len = img_start
        if text_len > 0:
            text_pos = torch.arange(curr_pos, curr_pos + text_len, device=device)
            temporal_list.append(text_pos)
            height_list.append(text_pos)
            width_list.append(text_pos)
            curr_pos += text_len

        # 2. <image_start> token
        temporal_list.append(torch.tensor([curr_pos], device=device))
        height_list.append(torch.tensor([curr_pos], device=device))
        width_list.append(torch.tensor([curr_pos], device=device))
        curr_pos += 1

        # 3. Image tokens — T2I generation order is [small] → [large],
        #    but image_grid_thw is stored as [large, small].
        #    Process grids in reverse order to match generation order.
        num_grids = image_grid_thw.shape[0]
        grid_indices = list(reversed(range(num_grids)))

        for g in grid_indices:
            t, h, w = image_grid_thw[g].tolist()
            t, h, w = int(t), int(h), int(w)
            num_tokens = t * h * w

            # Temporal: constant across the entire grid
            img_temporal = torch.full(
                (num_tokens,), curr_pos, device=device, dtype=torch.long
            )
            # Height: repeat each row index w times
            img_height = torch.arange(
                curr_pos, curr_pos + h, device=device
            ).repeat_interleave(w)
            # Width: cycle [0..w-1] for each row
            img_width = torch.arange(
                curr_pos, curr_pos + w, device=device
            ).repeat(h)

            temporal_list.append(img_temporal)
            height_list.append(img_height)
            width_list.append(img_width)

            curr_pos += max(h, w)

        # 4. <image_end> / EOS token (if present in the sequence)
        img_end_positions = (ids == IMAGE_END).nonzero(as_tuple=True)[0]
        if len(img_end_positions) > 0:
            temporal_list.append(torch.tensor([curr_pos], device=device))
            height_list.append(torch.tensor([curr_pos], device=device))
            width_list.append(torch.tensor([curr_pos], device=device))
    else:
        # Pure text (no images)
        text_pos = torch.arange(seq_len, device=device)
        temporal_list.append(text_pos)
        height_list.append(text_pos)
        width_list.append(text_pos)

    full_temporal = torch.cat(temporal_list, dim=0)
    full_height = torch.cat(height_list, dim=0)
    full_width = torch.cat(width_list, dim=0)

    # Pad or truncate to seq_len (should match, but be safe)
    actual_len = full_temporal.shape[0]
    if actual_len < seq_len:
        # Pad with the last position value
        pad_len = seq_len - actual_len
        last_val = curr_pos
        full_temporal = torch.cat([full_temporal, torch.full((pad_len,), last_val, device=device)])
        full_height = torch.cat([full_height, torch.full((pad_len,), last_val, device=device)])
        full_width = torch.cat([full_width, torch.full((pad_len,), last_val, device=device)])

    position_ids[0, 0, :] = full_temporal[:seq_len]
    position_ids[1, 0, :] = full_height[:seq_len]
    position_ids[2, 0, :] = full_width[:seq_len]

    return position_ids


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------


def grpo_loss(
    log_probs_current: torch.Tensor,
    log_probs_old: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the GRPO objective for a single rollout sample.

    L = -Σ_t [ min(ρ_t * Â, clip(ρ_t, 1-ε, 1+ε) * Â) - β * KL_t ]
    """
    log_ratio = log_probs_current - log_probs_old
    ratio = torch.exp(log_ratio)

    adv = advantages
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    kl = token_kl_divergence(log_probs_current, ref_log_probs)
    kl_loss = kl_coef * kl.mean()

    total_loss = policy_loss + kl_loss

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    stats = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "mean_ratio": ratio.mean().item(),
        "mean_kl": kl.mean().item(),
    }
    return total_loss, stats


# ---------------------------------------------------------------------------
# Reference model log-prob computation
# ---------------------------------------------------------------------------


class ReferenceModel:
    """
    Frozen copy of the AR model for computing reference log-probabilities.
    Used solely for the KL penalty term.
    """

    def __init__(self, model: torch.nn.Module, temperature: float = 0.9):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.temperature = temperature

    @torch.no_grad()
    def compute_log_probs(
        self,
        processor,
        prompt: str,
        height: int,
        width: int,
        generated_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass → per-token log-probs."""
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        input_length = inputs["input_ids"].shape[-1]
        full_ids = torch.cat(
            [inputs["input_ids"][0], generated_ids.to(device)], dim=0
        ).unsqueeze(0)
        attn_mask = torch.ones_like(full_ids)

        image_grid_thw = inputs.get("image_grid_thw")
        position_ids = build_position_ids_for_teacher_forcing(
            full_ids, image_grid_thw, device
        )

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            images_per_sample=torch.tensor(
                [image_grid_thw.shape[0]], dtype=torch.long, device=device
            ),
        )

        gen_len = generated_ids.shape[0]
        logits = outputs.logits[0, input_length - 1 : input_length - 1 + gen_len, :]
        log_probs_all = F.log_softmax(logits / self.temperature, dim=-1)
        log_probs = log_probs_all.gather(
            dim=-1, index=generated_ids.to(device).unsqueeze(-1)
        ).squeeze(-1)
        return log_probs


# ---------------------------------------------------------------------------
# Current policy log-prob re-computation (with gradients)
# ---------------------------------------------------------------------------


def recompute_log_probs(
    ar_model: torch.nn.Module,
    processor,
    prompt: str,
    height: int,
    width: int,
    generated_ids: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Re-compute log-probs under the *current* (updated) policy.
    Keeps gradients for backprop.
    """
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=False,
        target_h=height,
        target_w=width,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_length = inputs["input_ids"].shape[-1]
    full_ids = torch.cat(
        [inputs["input_ids"][0], generated_ids.to(device)], dim=0
    ).unsqueeze(0)
    attn_mask = torch.ones_like(full_ids)

    image_grid_thw = inputs.get("image_grid_thw")
    position_ids = build_position_ids_for_teacher_forcing(
        full_ids, image_grid_thw, device
    )

    outputs = ar_model(
        input_ids=full_ids,
        attention_mask=attn_mask,
        position_ids=position_ids,
        image_grid_thw=image_grid_thw,
        images_per_sample=torch.tensor(
            [image_grid_thw.shape[0]], dtype=torch.long, device=device
        ),
    )

    gen_len = generated_ids.shape[0]
    logits = outputs.logits[0, input_length - 1 : input_length - 1 + gen_len, :]
    log_probs_all = F.log_softmax(logits / temperature, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1, index=generated_ids.to(device).unsqueeze(-1)
    ).squeeze(-1)
    return log_probs


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GRPO Trainer (Accelerate-based)
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """
    Full GRPO training loop for GLM-Image with Accelerate support.

    Supports:
      - Multi-GPU data parallelism (via ``accelerate launch``)
      - Mixed precision (bf16/fp16)
      - Gradient accumulation
      - TensorBoard / W&B logging through Accelerate trackers

    Workflow per epoch:
      Phase 1 — Rollout:  AR model in eval mode, sample G images per prompt
                           (runs on each process for its own shard of prompts).
      Phase 2 — Ref:      Compute reference model log-probs (frozen, no grad).
      Phase 3 — Update:   Re-compute current log-probs (with grad), GRPO loss,
                           backward via Accelerator, optimiser step.
    """

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config

        # ── Accelerator ────────────────────────────────────────────
        log_with = self._resolve_log_with(config.report_to)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=log_with,
            project_dir=config.output_dir,
        )

        self._setup_logging()
        set_seed(config.seed)

        # ── Model / rollout engine / LoRA ──────────────────────────
        self._build_rollout_engine()
        self._setup_lora()
        self._build_reference_model()

        # ── Optimiser + scheduler (prepared by accelerator) ────────
        self._setup_optimiser()

        self.global_step = 0
        self.loss_history: List[Dict[str, Any]] = []

    # ================================================================ #
    #  Static helpers                                                    #
    # ================================================================ #

    @staticmethod
    def _resolve_log_with(report_to: str):
        if report_to == "all":
            return ["tensorboard", "wandb"]
        if report_to == "none":
            return []
        return report_to  # "tensorboard" or "wandb"

    # ================================================================ #
    #  Setup                                                            #
    # ================================================================ #

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        if self.accelerator.is_local_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def _build_rollout_engine(self):
        """Construct the rollout engine (loads full pipeline + reward models)."""
        cfg = self.config
        rollout_cfg = GRPORolloutConfig(
            model_path=cfg.model_path,
            torch_dtype=cfg.torch_dtype,
            lora_path=cfg.resume_lora_path,
            prompt_jsonl=cfg.prompt_jsonl,
            group_size=cfg.group_size,
            prompts_per_batch=cfg.prompts_per_batch,
            ar_temperature=cfg.ar_temperature,
            ar_top_p=cfg.ar_top_p,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            decoder_seed=cfg.decoder_seed,
            default_height=cfg.default_height,
            default_width=cfg.default_width,
            skip_decode=cfg.skip_decode,
            reward_models=cfg.reward_models,
            reward_device=cfg.reward_device,
            reward_service_url=cfg.reward_service_url,
            reward_clip_model=cfg.reward_clip_model,
            seed=cfg.seed,
            device=str(self.accelerator.device),
            output_dir=cfg.output_dir,
        )
        self.rollout_engine = GRPORolloutEngine(rollout_cfg)
        self.ar_model = self.rollout_engine.ar_model
        self.processor = self.rollout_engine.processor
        self.logger.info("Rollout engine built.")

    def _setup_lora(self):
        """Apply LoRA adapters to the AR model for training."""
        cfg = self.config
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for GRPO training: pip install peft")

        from peft import PeftModel

        if isinstance(self.ar_model, PeftModel):
            self.logger.info("LoRA already loaded (resumed checkpoint).")
            self.ar_model.train()
            if cfg.gradient_checkpointing:
                self.ar_model.gradient_checkpointing_enable()
                if hasattr(self.ar_model, "enable_input_require_grads"):
                    self.ar_model.enable_input_require_grads()
            return

        if cfg.gradient_checkpointing:
            self.ar_model.gradient_checkpointing_enable()
            if hasattr(self.ar_model, "enable_input_require_grads"):
                self.ar_model.enable_input_require_grads()
            else:
                def _hook(module, input, output):
                    output.requires_grad_(True)
                self.ar_model.get_input_embeddings().register_forward_hook(_hook)

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.ar_model = get_peft_model(self.ar_model, lora_config)
        self.ar_model.print_trainable_parameters()

        # Keep rollout engine's references in sync
        self.rollout_engine.ar_model = self.ar_model
        if self.rollout_engine.pipe is not None:
            self.rollout_engine.pipe.vision_language_encoder = self.ar_model

        self.logger.info("LoRA applied to AR model.")

    def _build_reference_model(self):
        """Create a frozen copy of the AR model for the KL penalty."""
        self.logger.info("Building reference model (frozen copy)...")
        from peft import PeftModel

        if isinstance(self.ar_model, PeftModel):
            ref = copy.deepcopy(self.ar_model)
            ref = ref.merge_and_unload()
        else:
            ref = copy.deepcopy(self.ar_model)

        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

        # Move to current device
        ref = ref.to(self.accelerator.device)

        self.ref_model = ReferenceModel(ref, temperature=self.config.ar_temperature)
        self.logger.info("Reference model ready.")

    def _setup_optimiser(self):
        cfg = self.config

        trainable_params = [p for p in self.ar_model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable_params)
        self.logger.info(f"Trainable parameters: {n_params:,}")

        if cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                opt_cls = bnb.optim.AdamW8bit
            except ImportError:
                self.logger.warning("bitsandbytes not available, falling back to AdamW")
                opt_cls = torch.optim.AdamW
        else:
            opt_cls = torch.optim.AdamW

        self.optimizer = opt_cls(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        # Scheduler created later once we know total steps
        self.lr_scheduler = None

    def _setup_lr_scheduler(self, total_steps: int):
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.logger.info(
            f"LR scheduler: cosine with {warmup_steps} warmup / "
            f"{total_steps} total steps"
        )

    def _prepare_with_accelerator(self):
        """Wrap model, optimiser, and scheduler with Accelerate."""
        if self.lr_scheduler is not None:
            self.ar_model, self.optimizer, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.ar_model, self.optimizer, self.lr_scheduler,
                )
            )
        else:
            self.ar_model, self.optimizer = self.accelerator.prepare(
                self.ar_model, self.optimizer,
            )

    # ================================================================ #
    #  Rollout helpers                                                  #
    # ================================================================ #

    def _enter_rollout_mode(self):
        """Switch AR model to eval mode and point rollout engine at it."""
        self._unwrapped = self.accelerator.unwrap_model(self.ar_model)
        self._unwrapped.eval()
        self.rollout_engine.ar_model = self._unwrapped
        if self.rollout_engine.pipe is not None:
            self.rollout_engine.pipe.vision_language_encoder = self._unwrapped

    def _exit_rollout_mode(self):
        """Restore wrapped model and switch back to train mode."""
        self.rollout_engine.ar_model = self.ar_model
        if self.rollout_engine.pipe is not None:
            self.rollout_engine.pipe.vision_language_encoder = self.ar_model
        self._unwrapped.train()

    # ================================================================ #
    #  Generated-ids cache                                              #
    # ================================================================ #

    @torch.no_grad()
    def _collect_generated_ids(
        self,
        group: GRPOGroup,
        sample: RolloutSample,
    ) -> torch.Tensor:
        """Return cached generated_ids or re-derive deterministically."""
        if sample.generated_ids is not None:
            return sample.generated_ids

        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.ar_model)
        messages = [[{"role": "user", "content": [{"type": "text", "text": group.prompt}]}]]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            target_h=group.height,
            target_w=group.width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]

        torch.manual_seed(sample.ar_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(sample.ar_seed)

        outputs = unwrapped.generate(
            **inputs,
            max_new_tokens=sample.log_probs.shape[0],
            do_sample=True,
            temperature=self.config.ar_temperature,
            top_p=self.config.ar_top_p,
            return_dict_in_generate=True,
        )
        return outputs.sequences[0, input_length:]

    # ================================================================ #
    #  Policy update (single sample)                                    #
    # ================================================================ #

    def _compute_sample_loss(
        self,
        group: GRPOGroup,
        sample: RolloutSample,
        advantage: float,
        ref_log_probs: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss for one sample (keeps gradient graph)."""
        cfg = self.config
        device = self.accelerator.device

        current_log_probs = recompute_log_probs(
            ar_model=self.ar_model,
            processor=self.processor,
            prompt=group.prompt,
            height=group.height,
            width=group.width,
            generated_ids=generated_ids,
            temperature=cfg.ar_temperature,
            device=device,
        )

        old_log_probs = sample.log_probs.detach().to(device)

        min_len = min(
            current_log_probs.shape[0],
            old_log_probs.shape[0],
            ref_log_probs.shape[0],
        )
        current_log_probs = current_log_probs[:min_len]
        old_log_probs = old_log_probs[:min_len]
        ref_lp = ref_log_probs[:min_len].to(device)

        adv_tensor = torch.tensor(advantage, device=device, dtype=torch.float32)

        loss, stats = grpo_loss(
            log_probs_current=current_log_probs,
            log_probs_old=old_log_probs,
            ref_log_probs=ref_lp,
            advantages=adv_tensor,
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
        )
        return loss, stats

    # ================================================================ #
    #  Main training loop                                               #
    # ================================================================ #

    def train(self):
        cfg = self.config

        # ── Estimate total optimiser steps ─────────────────────────
        n_prompts = len(self.rollout_engine.prompt_dataset)
        batches_per_epoch = (
            (n_prompts + cfg.prompts_per_batch - 1) // cfg.prompts_per_batch
        )
        samples_per_epoch = (
            batches_per_epoch * cfg.prompts_per_batch * cfg.group_size
        )
        total_updates = (
            cfg.num_epochs * samples_per_epoch * cfg.num_inner_steps
        ) // cfg.gradient_accumulation_steps
        self._setup_lr_scheduler(max(total_updates, 1))

        # ── Prepare with accelerator ───────────────────────────────
        self._prepare_with_accelerator()

        # ── Init trackers ──────────────────────────────────────────
        tracker_config = {
            k: v for k, v in cfg.__dict__.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        tracker_config.pop("wandb_project")
        self.accelerator.init_trackers(
            project_name=cfg.wandb_project,
            config=tracker_config,
        )

        # ── Banner ─────────────────────────────────────────────────
        self.logger.info("=" * 60)
        self.logger.info("GRPO Training for GLM-Image (Accelerate)")
        self.logger.info("=" * 60)
        self.logger.info(f"  Num processes: {self.accelerator.num_processes}")
        self.logger.info(f"  Device: {self.accelerator.device}")
        self.logger.info(f"  Mixed precision: {cfg.mixed_precision}")
        self.logger.info(f"  Prompts: {n_prompts}")
        self.logger.info(f"  Group size (G): {cfg.group_size}")
        self.logger.info(f"  Epochs: {cfg.num_epochs}")
        self.logger.info(f"  Prompts per batch: {cfg.prompts_per_batch}")
        self.logger.info(f"  Inner steps: {cfg.num_inner_steps}")
        self.logger.info(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
        self.logger.info(f"  Estimated total updates: {total_updates}")
        self.logger.info(f"  Learning rate: {cfg.learning_rate}")
        self.logger.info(f"  Clip eps: {cfg.clip_eps}")
        self.logger.info(f"  KL coef (beta): {cfg.kl_coef}")
        self.logger.info(f"  Reward models: {cfg.reward_models}")
        self.logger.info(f"  Output dir: {cfg.output_dir}")
        self.logger.info("=" * 60)

        # Save config (main process only)
        if self.accelerator.is_local_main_process:
            config_path = os.path.join(cfg.output_dir, "grpo_config.json")
            with open(config_path, "w") as f:
                json.dump(cfg.__dict__, f, indent=2, default=str)

        # ── Epoch loop ─────────────────────────────────────────────
        for epoch in range(cfg.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            self.logger.info(f"{'='*60}")

            epoch_start = time.time()
            self.rollout_engine.config.seed = cfg.seed + epoch * 10000
            self.rollout_engine.batch_sampler.set_epoch(epoch)
            device = self.accelerator.device

            epoch_stats: Dict[str, List[float]] = {
                "policy_loss": [], "kl_loss": [], "total_loss": [],
                "approx_kl": [], "clip_frac": [], "reward": [],
                "advantage": [],
            }
            epoch_rollout_time = 0.0
            epoch_ref_time = 0.0
            epoch_update_time = 0.0
            last_batch_groups = None

            self._enter_rollout_mode()
            for batch_groups in self.rollout_engine.rollout():
                self._exit_rollout_mode()
                last_batch_groups = batch_groups

                # ── Ref log-probs + advantages for this batch ──────
                t_ref = time.time()
                for group in batch_groups:
                    if group.rewards is None:
                        self.logger.warning(
                            f"Group '{group.prompt[:40]}...' has no rewards — "
                            f"skipping."
                        )
                        continue

                    group.advantages = compute_group_advantages(
                        group.rewards, eps=cfg.advantage_eps
                    )

                    for sample in group.samples:
                        gen_ids = self._collect_generated_ids(group, sample)
                        sample._generated_ids = gen_ids
                        sample._ref_log_probs = (
                            self.ref_model.compute_log_probs(
                                processor=self.processor,
                                prompt=group.prompt,
                                height=group.height,
                                width=group.width,
                                generated_ids=gen_ids,
                                device=device,
                            )
                        )
                epoch_ref_time += time.time() - t_ref

                # ── Phase 2: Policy update for this batch ──────────
                t_upd = time.time()
                self.ar_model.train()

                for group in batch_groups:
                    if group.advantages is None:
                        continue

                    for _inner in range(cfg.num_inner_steps):
                        for s_idx, sample in enumerate(group.samples):
                            adv = group.advantages[s_idx].item()
                            ref_lp = sample._ref_log_probs
                            gen_ids = sample._generated_ids

                            with self.accelerator.accumulate(self.ar_model):
                                loss, stats = self._compute_sample_loss(
                                    group=group,
                                    sample=sample,
                                    advantage=adv,
                                    ref_log_probs=ref_lp,
                                    generated_ids=gen_ids,
                                )

                                self.accelerator.backward(loss)

                                if cfg.max_grad_norm > 0:
                                    self.accelerator.clip_grad_norm_(
                                        self.ar_model.parameters(),
                                        cfg.max_grad_norm,
                                    )

                                self.optimizer.step()
                                if self.lr_scheduler is not None:
                                    self.lr_scheduler.step()
                                self.optimizer.zero_grad()

                            if self.accelerator.sync_gradients:
                                self.global_step += 1

                                if self.global_step % cfg.logging_steps == 0:
                                    lr = self.optimizer.param_groups[0]["lr"]
                                    log_metrics = {
                                        "train/policy_loss": stats["policy_loss"],
                                        "train/kl_loss": stats["kl_loss"],
                                        "train/total_loss": stats["total_loss"],
                                        "train/approx_kl": stats["approx_kl"],
                                        "train/clip_frac": stats["clip_frac"],
                                        "train/learning_rate": lr,
                                        "train/epoch": epoch,
                                    }
                                    if sample.reward is not None:
                                        log_metrics["train/reward"] = sample.reward
                                    log_metrics["train/advantage"] = adv

                                    self.accelerator.log(
                                        log_metrics, step=self.global_step
                                    )
                                    self.logger.info(
                                        f"  step={self.global_step} | "
                                        f"loss={stats['total_loss']:.4f} "
                                        f"(policy={stats['policy_loss']:.4f}, "
                                        f"kl={stats['kl_loss']:.4f}) | "
                                        f"clip={stats['clip_frac']:.2f} | "
                                        f"reward={sample.reward or 0:.4f} | "
                                        f"adv={adv:.3f} | "
                                        f"lr={lr:.2e}"
                                    )

                                if (
                                    cfg.save_steps > 0
                                    and self.global_step % cfg.save_steps == 0
                                ):
                                    self._save_checkpoint(
                                        f"step-{self.global_step}"
                                    )

                            for k in (
                                "policy_loss", "kl_loss", "total_loss",
                                "approx_kl", "clip_frac",
                            ):
                                epoch_stats[k].append(stats[k])
                            if sample.reward is not None:
                                epoch_stats["reward"].append(sample.reward)
                            epoch_stats["advantage"].append(adv)

                epoch_update_time += time.time() - t_upd

                # Free batch memory and re-enter rollout mode for next batch
                del batch_groups
                self._enter_rollout_mode()

            self._exit_rollout_mode()
            epoch_time = time.time() - epoch_start

            # ── Epoch summary ──────────────────────────────────────
            def _safe_mean(lst: List[float]) -> float:
                return sum(lst) / len(lst) if lst else 0.0

            summary = {
                "epoch": epoch + 1,
                "policy_loss": _safe_mean(epoch_stats["policy_loss"]),
                "kl_loss": _safe_mean(epoch_stats["kl_loss"]),
                "total_loss": _safe_mean(epoch_stats["total_loss"]),
                "approx_kl": _safe_mean(epoch_stats["approx_kl"]),
                "clip_frac": _safe_mean(epoch_stats["clip_frac"]),
                "mean_reward": _safe_mean(epoch_stats["reward"]),
                "ref_time": epoch_ref_time,
                "update_time": epoch_update_time,
                "epoch_time": epoch_time,
                "global_step": self.global_step,
            }
            self.loss_history.append(summary)

            self.logger.info(f"\nEpoch {epoch + 1} summary:")
            self.logger.info(f"  Total loss: {summary['total_loss']:.4f}")
            self.logger.info(f"  Policy loss: {summary['policy_loss']:.4f}")
            self.logger.info(f"  KL loss: {summary['kl_loss']:.4f}")
            self.logger.info(f"  Mean reward: {summary['mean_reward']:.4f}")
            self.logger.info(f"  Approx KL: {summary['approx_kl']:.4f}")
            self.logger.info(f"  Clip fraction: {summary['clip_frac']:.2%}")
            self.logger.info(
                f"  Time: ref={epoch_ref_time:.1f}s, "
                f"update={epoch_update_time:.1f}s, "
                f"total={epoch_time:.1f}s"
            )

            self._save_checkpoint(f"epoch-{epoch + 1}")
            if cfg.save_images and not cfg.skip_decode and last_batch_groups:
                self._save_epoch_images(epoch + 1, last_batch_groups)

        # ── Finish ─────────────────────────────────────────────────
        self._save_checkpoint("final")
        self._save_loss_history()
        self.accelerator.end_training()
        self.logger.info("\nGRPO training complete!")

    # ================================================================ #
    #  Checkpointing                                                    #
    # ================================================================ #

    def _save_checkpoint(self, name: str):
        if not self.accelerator.is_local_main_process:
            return

        ckpt_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.logger.info(f"Saving checkpoint -> {ckpt_dir}")

        unwrapped = self.accelerator.unwrap_model(self.ar_model)

        from peft import PeftModel

        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(ckpt_dir)
        else:
            torch.save(
                unwrapped.state_dict(), os.path.join(ckpt_dir, "model.pt")
            )

        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

        self.processor.save_pretrained(ckpt_dir)

    def _save_loss_history(self):
        if not self.accelerator.is_local_main_process:
            return
        path = os.path.join(self.config.output_dir, "grpo_loss_history.json")
        with open(path, "w") as f:
            json.dump(self.loss_history, f, indent=2)

    def _save_epoch_images(
        self, epoch: int, batch_groups: List[GRPOGroup]
    ):
        if not self.accelerator.is_local_main_process:
            return
        img_dir = os.path.join(
            self.config.output_dir, f"images_epoch_{epoch}"
        )
        os.makedirs(img_dir, exist_ok=True)
        for g_idx, group in enumerate(batch_groups):
            for s_idx, sample in enumerate(group.samples):
                if sample.image is not None:
                    fname = f"group{g_idx}_sample{s_idx}"
                    if sample.reward is not None:
                        fname += f"_r{sample.reward:.4f}"
                    fname += ".png"
                    sample.image.save(os.path.join(img_dir, fname))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="GRPO Training for GLM-Image (Accelerate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--torch_dtype", type=str, default="bfloat16")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules", nargs="+",
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_up_proj", "down_proj",
        ],
    )
    p.add_argument("--resume_lora_path", type=str, default=None)

    # Rollout
    p.add_argument("--prompt_jsonl", type=str, required=True)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--prompts_per_batch", type=int, default=1)
    p.add_argument("--ar_temperature", type=float, default=0.9)
    p.add_argument("--ar_top_p", type=float, default=0.75)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--decoder_seed", type=int, default=42)
    p.add_argument("--default_height", type=int, default=1024)
    p.add_argument("--default_width", type=int, default=1024)
    p.add_argument("--skip_decode", action="store_true")

    # Reward
    p.add_argument("--reward_models", type=str, default="clip_score")
    p.add_argument("--reward_device", type=str, default=None)
    p.add_argument("--reward_service_url", type=str, default="")
    p.add_argument(
        "--reward_clip_model", type=str,
        default="openai/clip-vit-large-patch14",
    )

    # GRPO
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_inner_steps", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Optimiser
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--use_8bit_adam", action="store_true")

    # Precision
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"],
    )

    # Logging
    p.add_argument("--output_dir", type=str, default="./outputs/grpo")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--no_save_images", action="store_true")
    p.add_argument(
        "--report_to", type=str, default="tensorboard",
        choices=["tensorboard", "wandb", "all", "none"],
    )
    p.add_argument("--wandb_project", type=str, default="glm-image-grpo")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    config = GRPOTrainingConfig(
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        resume_lora_path=args.resume_lora_path,
        prompt_jsonl=args.prompt_jsonl,
        group_size=args.group_size,
        prompts_per_batch=args.prompts_per_batch,
        ar_temperature=args.ar_temperature,
        ar_top_p=args.ar_top_p,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        decoder_seed=args.decoder_seed,
        default_height=args.default_height,
        default_width=args.default_width,
        skip_decode=args.skip_decode,
        reward_models=args.reward_models,
        reward_device=args.reward_device,
        reward_service_url=args.reward_service_url,
        reward_clip_model=args.reward_clip_model,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        num_epochs=args.num_epochs,
        num_inner_steps=args.num_inner_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        use_8bit_adam=args.use_8bit_adam,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_images=not args.no_save_images,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
