#!/usr/bin/env python3
"""
GLM-Image LoRA Finetuning Script

This script provides LoRA (Low-Rank Adaptation) finetuning for GLM-Image's
autoregressive (AR) text model component. The AR model generates discrete
image tokens that are then decoded by the diffusion decoder.

=============================================================================
                         TRAINING ARCHITECTURE
=============================================================================

GLM-Image consists of two main components:
1. AR Model (9B params): Predicts discrete image tokens (0-16383)
2. Diffusion Decoder (7B params): Decodes tokens to pixels (NOT trained here)

Training Flow:
```
Target Image                                        
    ↓                                               
Vision Encoder → VQVAE.encode() → discrete tokens (labels)
                                        ↓
Text Prompt → Tokenizer → input_ids → AR Model → logits
                                        ↓
                              CrossEntropyLoss(logits, labels)
```

Key Details:
- lm_head outputs to vision_vocab_size=16512 (NOT full text vocab)
- VQVAE codebook has 16384 entries (tokens 0-16383)
- Special tokens: 16384=<image_start>, 16385=<image_end>/EOS
- Loss is standard next-token prediction (causal LM)

Why we don't use Diffusion Decoder for training:
- AR model learns to predict the "semantic recipe" (discrete tokens)
- Diffusion decoder is a fixed translator (tokens → pixels)
- Training AR alone is sufficient for style/domain adaptation

=============================================================================

Supported modes:
- Text-to-Image (T2I): Finetune on text-image pairs
- Image-to-Image (I2I): Finetune on image transformation tasks (coming soon)

Usage:
    python finetune_lora.py \\
        --model_path zai-org/GLM-Image \\
        --dataset_subset data_1024_10K \\
        --output_dir ./outputs/glm-image-lora \\
        --lora_rank 8 \\
        --learning_rate 1e-4 \\
        --num_epochs 5

Requirements:
    - transformers (from source)
    - diffusers (from source)  
    - peft
    - accelerate
    - datasets
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Training utilities
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

# PEFT for LoRA
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. Run: pip install peft")

# Local imports
from data_loader import DatasetConfig, create_dataloader, get_dataset

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for LoRA finetuning."""

    # Model
    model_path: str = "zai-org/GLM-Image"
    torch_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"

    # LoRA configuration
    # For image generation AR models, higher rank is recommended (32-128)
    # MLP layers (gate_up_proj, down_proj) are crucial for learning visual styles
    lora_rank: int = 32
    lora_alpha: int = 64  # Typically 2x rank for stable training
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            # Attention layers - affect spatial composition and layout
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP layers - crucial for learning visual styles, textures, and details
            "gate_up_proj",
            "down_proj",
        ]
    )
    lora_modules_to_save: Optional[List[str]] = None

    # Which components to train
    train_text_model: bool = True  # AR text model (recommended)
    train_vision_encoder: bool = False  # Vision encoder (usually frozen)

    # Dataset
    dataset_subset: str = "data_1024_10K"
    dataset_streaming: bool = False  # Set False for proper epoch training
    resolution: int = 1024
    task_type: str = "t2i"  # "t2i" or "i2i"
    local_data_dir: Optional[str] = None
    cache_dir: Optional[str] = None  # Default: finetune/data/

    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    # Batch size: increase for better GPU utilization (requires more VRAM)
    # Effective batch size = batch_size * gradient_accumulation_steps
    batch_size: int = 2  # Increase if GPU memory allows
    gradient_accumulation_steps: int = 2  # Reduce if batch_size increased
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Optimization
    use_8bit_adam: bool = False
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Saving & Logging
    output_dir: str = "./outputs/glm-image-lora"
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500
    resume_from_checkpoint: Optional[str] = None

    # Misc
    seed: int = 42
    num_workers: int = 4

    def __post_init__(self):
        """Validate configuration."""
        if self.lora_alpha < self.lora_rank:
            print(
                f"Warning: lora_alpha ({self.lora_alpha}) < lora_rank ({self.lora_rank}). "
                f"Consider using lora_alpha >= 2 * lora_rank for better adaptation."
            )

        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required. Install with: pip install peft")

    @property
    def dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


# ============================================================================
# Model Setup
# ============================================================================


def load_model_and_processor(config: TrainingConfig):
    """
    Load GLM-Image model and processor.

    Supports two formats:
    1. Transformers format: Single directory with model and processor
    2. Diffusers pipeline format: Separate subdirectories
       - vision_language_encoder/ (AR model)
       - processor/ (processor config)

    Returns:
        model: GlmImageForConditionalGeneration
        processor: GlmImageProcessor
    """
    from transformers import AutoProcessor, GlmImageForConditionalGeneration

    model_path = config.model_path
    logging.info(f"Loading model from {model_path}...")

    # Detect format by checking for model_index.json (diffusers pipeline)
    model_index_path = os.path.join(model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    if is_diffusers_format:
        logging.info("Detected diffusers pipeline format")

        # Load model from vision_language_encoder subdirectory
        model_subdir = os.path.join(model_path, "vision_language_encoder")
        if not os.path.exists(model_subdir):
            raise FileNotFoundError(
                f"vision_language_encoder not found in {model_path}. "
                "Expected diffusers pipeline structure."
            )

        logging.info(f"Loading AR model from {model_subdir}")
        model = GlmImageForConditionalGeneration.from_pretrained(
            model_subdir,
            torch_dtype=config.dtype,
            device_map=None,
            trust_remote_code=True,
        )

        # Load processor from processor subdirectory
        processor_subdir = os.path.join(model_path, "processor")
        if not os.path.exists(processor_subdir):
            raise FileNotFoundError(
                f"processor not found in {model_path}. "
                "Expected diffusers pipeline structure."
            )

        logging.info(f"Loading processor from {processor_subdir}")
        processor = AutoProcessor.from_pretrained(
            processor_subdir,
            trust_remote_code=True,
        )
    else:
        # Standard transformers format
        logging.info("Using transformers format (single directory)")

        model = GlmImageForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=config.dtype,
            device_map=None,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    logging.info(f"Model loaded: {model.__class__.__name__}")
    logging.info(f"Model dtype: {config.dtype}")

    return model, processor


def setup_lora(model, config: TrainingConfig):
    """
    Apply LoRA to the model's AR text component.

    GLM-Image Architecture:
    - model.model.visual: Vision encoder
    - model.model.vqmodel: VQVAE
    - model.model.language_model: AR text model (9B params) - TARGET FOR LORA

    The AR text model has layers with:
    - self_attn: GlmImageTextAttention (q_proj, k_proj, v_proj, o_proj)
    - mlp: GlmImageTextMLP (gate_up_proj, down_proj)
    """
    logging.info("Setting up LoRA...")

    # Build target modules pattern for the text model
    target_modules = []

    if config.train_text_model:
        # Target AR text model attention layers
        # Path: model.language_model.layers.*.self_attn.{q,k,v,o}_proj
        for module_name in config.lora_target_modules:
            target_modules.append(module_name)

    if config.train_vision_encoder:
        # Vision encoder uses different attention structure
        # Path: model.visual.blocks.*.attn.{qkv, proj}
        target_modules.extend(["qkv", "proj"])

    logging.info(f"LoRA target modules: {target_modules}")

    # Create LoRA config
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        modules_to_save=config.lora_modules_to_save,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Enable gradient checkpointing before applying PEFT
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # For LoRA, we need to enable input gradients
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def freeze_non_lora_weights(model, config: TrainingConfig):
    """
    Freeze all weights except LoRA adapters.

    This is automatically handled by PEFT, but we add extra freezing
    for components we definitely don't want to train.
    """
    # Freeze VQVAE (always frozen - it's for tokenization only)
    if hasattr(model, "base_model"):
        base = model.base_model.model
    else:
        base = model.model

    if hasattr(base, "vqmodel"):
        for param in base.vqmodel.parameters():
            param.requires_grad = False
        logging.info("Frozen: VQVAE")

    # Optionally freeze vision encoder
    if not config.train_vision_encoder and hasattr(base, "visual"):
        for param in base.visual.parameters():
            param.requires_grad = False
        logging.info("Frozen: Vision encoder")


# ============================================================================
# Training Loop
# ============================================================================


class GlmImageLoraTrainer:
    """
    Trainer class for GLM-Image LoRA finetuning.

    Handles:
    - Model loading and LoRA setup
    - Dataset preparation
    - Training loop with gradient accumulation
    - Checkpointing and logging
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()

        # Set seed for reproducibility
        set_seed(config.seed)

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=config.output_dir,
        )

        # Load model and processor
        self.model, self.processor = load_model_and_processor(config)

        # Setup LoRA
        self.model = setup_lora(self.model, config)
        freeze_non_lora_weights(self.model, config)

        # Setup dataset
        self.train_dataloader = self.setup_dataset()

        # Setup optimizer and scheduler
        self.optimizer, self.lr_scheduler = self.setup_optimizer()

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

        self.global_step = 0
        self.start_epoch = 0

        # Initialize loss history for plotting
        self.loss_history = []

        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _save_loss_history(self):
        """Save loss history to JSON file for plot_loss.py."""
        loss_file = os.path.join(self.config.output_dir, "loss_history.json")
        with open(loss_file, "w") as f:
            json.dump(self.loss_history, f, indent=2)

    def setup_dataset(self) -> DataLoader:
        """Setup training dataset and dataloader."""
        logging.info("Setting up dataset...")

        dataset_config = DatasetConfig(
            subset=self.config.dataset_subset,
            streaming=self.config.dataset_streaming,
            resolution=self.config.resolution,
            task_type=self.config.task_type,
            local_data_dir=self.config.local_data_dir,
            cache_dir=self.config.cache_dir,  # Use cache_dir from training config
        )

        logging.info(f"Dataset cache directory: {dataset_config.cache_dir}")

        dataset = get_dataset(dataset_config)

        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        logging.info(f"Dataset: {dataset_config.dataset_name}")
        logging.info(f"Subset: {dataset_config.subset}")
        logging.info(f"Batch size: {self.config.batch_size}")

        return dataloader

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        logging.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
        )

        # Optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logging.warning("bitsandbytes not available, using standard AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warmup
        num_training_steps = (
            len(self.train_dataloader) * self.config.num_epochs
        ) // self.config.gradient_accumulation_steps

        from transformers import get_cosine_schedule_with_warmup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logging.info(f"Total training steps: {num_training_steps}")
        logging.info(f"Warmup steps: {self.config.warmup_steps}")

        return optimizer, lr_scheduler

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Training Flow for GLM-Image T2I:
        =================================

        1. Target Image → Vision Encoder → VQVAE.encode() → discrete tokens (labels)
        2. Text Prompt → Tokenizer → input_ids (with image placeholders)
        3. Replace placeholders with target image tokens
        4. Shift for causal LM: predict next token
        5. Loss = CrossEntropy(logits, shifted_labels)

        Input sequence format:
            [text_tokens] <image_start> [image_tokens...] <image_end>

        Labels (for loss):
            [-100, -100, ...] -100 [image_tokens...] <image_end>
            (Only image tokens contribute to loss)
        """
        prompts = batch["prompt"]
        images = batch["image"]  # Target images as tensors [B, C, H, W]

        device = self.accelerator.device
        batch_size = images.shape[0]

        # Get the unwrapped model for accessing components
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "base_model"):
            # PEFT wrapped model
            base_model = unwrapped.base_model.model.model  # GlmImageModel
            peft_model = unwrapped
        else:
            base_model = unwrapped.model
            peft_model = unwrapped

        # ================================================================
        # Step 1: Encode target images to discrete tokens using VQVAE
        # ================================================================

        # Process images through image processor
        image_inputs = self.processor.image_processor(
            images=[img for img in images],  # Convert tensor to list of tensors
            return_tensors="pt",
        )
        pixel_values = image_inputs["pixel_values"].to(device)
        image_grid_thw = image_inputs["image_grid_thw"].to(
            device
        )  # [B, 3] for T2I (one grid per image)

        # Encode through vision encoder + VQVAE to get discrete tokens
        with torch.no_grad():
            # Vision encoder
            image_features = base_model.get_image_features(
                pixel_values, image_grid_thw
            ).pooler_output
            image_features = torch.cat(image_features, dim=0)

            # VQVAE quantization → discrete token indices
            large_image_tokens = base_model.get_image_tokens(
                image_features, image_grid_thw
            )
            # Shape: [total_patches] with values in [0, 16383]

        # ================================================================
        # Step 1.5: Create small image tokens for T2I mode
        # ================================================================
        # T2I generates two token sequences: [large H×W] + [small H/2 × W/2]
        # Small tokens are obtained by nearest-neighbor downsampling

        is_t2i = self.config.task_type == "t2i"

        if is_t2i:
            # Calculate small image grid dimensions (half of large)
            small_grids = []
            for i in range(batch_size):
                t, h, w = image_grid_thw[i].tolist()
                small_grids.append([t, h // 2, w // 2])
            small_grid_thw = torch.tensor(small_grids, device=device, dtype=torch.long)

            # Create small image tokens via nearest-neighbor downsampling
            small_image_tokens_list = []
            token_offset = 0
            for i in range(batch_size):
                t, h, w = image_grid_thw[i].tolist()
                num_tokens = t * h * w
                large_tokens_i = large_image_tokens[
                    token_offset : token_offset + num_tokens
                ]
                token_offset += num_tokens

                # Reshape to 2D and downsample
                large_2d = large_tokens_i.view(1, 1, h, w).float()
                small_2d = F.interpolate(
                    large_2d, size=(h // 2, w // 2), mode="nearest"
                )
                small_tokens_i = small_2d.view(-1).long()
                small_image_tokens_list.append(small_tokens_i)

            small_image_tokens = torch.cat(small_image_tokens_list, dim=0)

            # Combine grids: [large_grid, small_grid] for each sample
            # T2I mode: image_grid_thw should have 2 grids per sample
            combined_grids = []
            for i in range(batch_size):
                combined_grids.append(image_grid_thw[i : i + 1])
                combined_grids.append(small_grid_thw[i : i + 1])
            image_grid_thw = torch.cat(combined_grids, dim=0)  # [B*2, 3]

            # Combine tokens: [large_tokens, small_tokens] for each sample
            target_image_tokens = torch.cat(
                [large_image_tokens, small_image_tokens], dim=0
            )
        else:
            # I2I mode: only large image tokens
            target_image_tokens = large_image_tokens

        # ================================================================
        # Step 2: Create input sequence with text + image tokens
        # ================================================================

        # Get special token IDs from config
        config = (
            peft_model.config if hasattr(peft_model, "config") else base_model.config
        )
        if hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = config

        image_start_token_id = getattr(text_config, "image_start_token_id", 16384)
        image_end_token_id = getattr(text_config, "image_end_token_id", 16385)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        # Tokenize text prompts
        text_inputs = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        text_input_ids = text_inputs["input_ids"].to(device)
        text_attention_mask = text_inputs["attention_mask"].to(device)

        # Calculate image tokens per sample
        # T2I: each sample has 2 grids (large + small), I2I: each sample has 1 grid
        grids_per_sample = 2 if is_t2i else 1
        tokens_per_sample = []
        for i in range(batch_size):
            total_tokens = 0
            for j in range(grids_per_sample):
                grid_idx = i * grids_per_sample + j
                h, w = (
                    image_grid_thw[grid_idx, 1].item(),
                    image_grid_thw[grid_idx, 2].item(),
                )
                total_tokens += h * w
            tokens_per_sample.append(total_tokens)

        # For T2I, we need to interleave small and large tokens per sample
        # CRITICAL: GLM-Image generates tokens in order: [small] → [large] → [EOS]
        # This matches pipeline's _extract_large_image_tokens which uses offset=small_tokens
        #
        # Currently target_image_tokens is [all_large_tokens, all_small_tokens]
        # We need to reorder to [sample0_small, sample0_large, sample1_small, sample1_large, ...]
        if is_t2i:
            reordered_tokens = []
            large_offset = 0
            small_offset = len(large_image_tokens)
            for i in range(batch_size):
                # IMPORTANT: Small tokens come FIRST in generation order!
                # Small tokens for sample i
                small_h = image_grid_thw[i * 2 + 1, 1].item()
                small_w = image_grid_thw[i * 2 + 1, 2].item()
                num_small = small_h * small_w
                reordered_tokens.append(
                    target_image_tokens[small_offset : small_offset + num_small]
                )
                small_offset += num_small

                # Large tokens for sample i (come after small)
                large_h = image_grid_thw[i * 2, 1].item()
                large_w = image_grid_thw[i * 2, 2].item()
                num_large = large_h * large_w
                reordered_tokens.append(
                    target_image_tokens[large_offset : large_offset + num_large]
                )
                large_offset += num_large
            target_image_tokens = torch.cat(reordered_tokens, dim=0)

        # Build full sequence: [text] <image_start> [image_tokens] <image_end>
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        token_offset = 0
        for i in range(batch_size):
            # Text tokens for this sample (without padding)
            text_len = text_attention_mask[i].sum().item()
            text_ids = text_input_ids[i, :text_len]

            # Image tokens for this sample (large + small for T2I)
            num_img_tokens = tokens_per_sample[i]
            img_tokens = target_image_tokens[
                token_offset : token_offset + num_img_tokens
            ]
            token_offset += num_img_tokens

            # Build input_ids: [text] <image_start> [image_tokens] <image_end>
            input_ids = torch.cat(
                [
                    text_ids,
                    torch.tensor([image_start_token_id], device=device),
                    img_tokens,
                    torch.tensor([image_end_token_id], device=device),
                ]
            )

            # Build labels: [-100 for text] -100 [image_tokens] <image_end>
            # Note: We predict image tokens and EOS, not the text or <image_start>
            labels = torch.cat(
                [
                    torch.full(
                        (len(text_ids) + 1,), -100, device=device, dtype=torch.long
                    ),  # text + <image_start>
                    img_tokens,
                    torch.tensor([image_end_token_id], device=device),
                ]
            )

            # Attention mask (all ones for valid tokens)
            attn_mask = torch.ones(len(input_ids), device=device, dtype=torch.long)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attn_mask)

        # Pad sequences to same length (LEFT PADDING for causal LM)
        # GLM-Image uses left padding so all sequences end at the same position
        max_len = max(len(ids) for ids in all_input_ids)

        padded_input_ids = torch.full(
            (batch_size, max_len), pad_token_id, device=device, dtype=torch.long
        )
        padded_labels = torch.full(
            (batch_size, max_len), -100, device=device, dtype=torch.long
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_len), device=device, dtype=torch.long
        )

        for i in range(batch_size):
            seq_len = len(all_input_ids[i])
            # Left padding: place content at the end, padding at the beginning
            padded_input_ids[i, -seq_len:] = all_input_ids[i]
            padded_labels[i, -seq_len:] = all_labels[i]
            padded_attention_mask[i, -seq_len:] = all_attention_masks[i]

        # ================================================================
        # Step 2.5: Compute position_ids manually for 3D RoPE
        # ================================================================
        # The model's get_rope_index is designed for generation (incomplete images).
        # For training, we have complete image tokens, so we compute position_ids manually.
        #
        # Position_ids shape: [3, batch_size, seq_len] for (temporal, height, width)
        # - Text tokens: sequential positions [0, 1, 2, ...]
        # - Image tokens: 2D spatial encoding with position_temporal fixed,
        #                 position_height and position_width varying

        position_ids = torch.zeros(
            3, batch_size, max_len, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            valid_len = int(padded_attention_mask[i].sum().item())
            start_idx = max_len - valid_len  # Due to left padding

            curr_pos = 0  # Current position counter

            # Get the unpadded sequence for this sample
            curr_input_ids = all_input_ids[i]

            # Find image boundaries
            image_start_positions = (curr_input_ids == image_start_token_id).nonzero(
                as_tuple=True
            )[0]

            # Build position_ids for each segment
            temporal_list = []
            height_list = []
            width_list = []

            grid_offset = i * grids_per_sample  # Starting grid index for this sample

            if len(image_start_positions) > 0:
                img_start = image_start_positions[0].item()

                # 1. Text tokens before <image_start>
                text_len = img_start
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

                # 3. Image tokens - iterate through all grids for this sample
                # CRITICAL: For T2I, generation order is [small] → [large]
                # but image_grid_thw is [large, small] per sample
                # So we need to process grids in reverse order for T2I
                if is_t2i:
                    grid_indices = list(reversed(range(grids_per_sample)))
                else:
                    grid_indices = list(range(grids_per_sample))

                for g in grid_indices:
                    grid_idx = grid_offset + g
                    t, h, w = image_grid_thw[grid_idx].tolist()
                    num_tokens = t * h * w

                    # Temporal: constant for the entire image
                    img_temporal = torch.full(
                        (num_tokens,), curr_pos, device=device, dtype=torch.long
                    )

                    # Height: repeats each row index W times
                    img_height = torch.arange(
                        curr_pos, curr_pos + h, device=device
                    ).repeat_interleave(w)

                    # Width: cycles [0, 1, ..., W-1] for each row
                    img_width = torch.arange(
                        curr_pos, curr_pos + w, device=device
                    ).repeat(h)

                    temporal_list.append(img_temporal)
                    height_list.append(img_height)
                    width_list.append(img_width)

                    curr_pos += max(h, w)  # Advance position by max dimension

                # 4. <image_end> token
                temporal_list.append(torch.tensor([curr_pos], device=device))
                height_list.append(torch.tensor([curr_pos], device=device))
                width_list.append(torch.tensor([curr_pos], device=device))
            else:
                # No images, just text
                seq_len = len(curr_input_ids)
                text_pos = torch.arange(curr_pos, curr_pos + seq_len, device=device)
                temporal_list.append(text_pos)
                height_list.append(text_pos)
                width_list.append(text_pos)

            # Concatenate all segments
            full_temporal = torch.cat(temporal_list, dim=0)
            full_height = torch.cat(height_list, dim=0)
            full_width = torch.cat(width_list, dim=0)

            # Place in padded position_ids (respecting left padding)
            position_ids[0, i, start_idx:] = full_temporal
            position_ids[1, i, start_idx:] = full_height
            position_ids[2, i, start_idx:] = full_width

        # ================================================================
        # Step 3: Forward pass and compute loss
        # ================================================================

        # Now we pass manually computed position_ids to bypass get_rope_index
        # which doesn't handle training scenarios with all image tokens present

        # NOTE: Do NOT pass labels to model.forward() because:
        # The model's loss_function uses text_config.vocab_size (168064)
        # but lm_head outputs vision_vocab_size (16512), causing shape mismatch.
        # We compute loss manually with correct vocab_size.

        outputs = self.model(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            images_per_sample=torch.full(
                (batch_size,), grids_per_sample, dtype=torch.long, device=device
            ),
            # labels=padded_labels,  # Don't pass labels - compute loss manually
        )

        # Compute loss manually with correct vision_vocab_size
        # Shift logits and labels for causal LM loss
        # logits: [B, seq_len, vision_vocab_size=16512]
        # labels: [B, seq_len]
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = padded_labels[..., 1:].contiguous()

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def train(self):
        """Main training loop."""
        logging.info("=" * 50)
        logging.info("Starting training...")
        logging.info(f"  Epochs: {self.config.num_epochs}")
        logging.info(f"  Batch size: {self.config.batch_size}")
        logging.info(
            f"  Gradient accumulation: {self.config.gradient_accumulation_steps}"
        )
        logging.info(
            f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )
        logging.info("=" * 50)

        # Training loop
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Compute loss
                    loss = self.compute_loss(batch)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update tracking
                epoch_loss += loss.detach().item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = self.lr_scheduler.get_last_lr()[0]

                    # Log to accelerator (TensorBoard)
                    self.accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                        },
                        step=self.global_step,
                    )

                    # Save to loss history for plot_loss.py
                    if self.accelerator.is_local_main_process:
                        self.loss_history.append(
                            {
                                "step": self.global_step,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "lr": current_lr,
                                "epoch": epoch,
                            }
                        )
                        self._save_loss_history()

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logging.info(
                f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}"
            )

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch + 1}")

        # Final save
        self.save_checkpoint("final")
        logging.info("Training completed!")

    def save_checkpoint(self, name: str):
        """Save a checkpoint."""
        if not self.accelerator.is_local_main_process:
            return

        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        logging.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save LoRA weights
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.global_step // len(self.train_dataloader),
            "config": self.config.__dict__,
        }

        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2, default=str)

        # Save processor
        self.processor.save_pretrained(checkpoint_dir)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        logging.info(f"Loading checkpoint from {checkpoint_path}")

        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                training_state = json.load(f)
            self.global_step = training_state.get("global_step", 0)
            self.start_epoch = training_state.get("epoch", 0)

        # Load existing loss history if available
        loss_file = os.path.join(self.config.output_dir, "loss_history.json")
        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                self.loss_history = json.load(f)
            logging.info(f"Loaded {len(self.loss_history)} loss history entries")

        # Load LoRA weights
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.accelerator.unwrap_model(self.model).base_model,
            checkpoint_path,
        )


# ============================================================================
# Inference / Merging Utilities
# ============================================================================


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Merge LoRA weights into the base model and save.

    Supports both transformers and diffusers pipeline formats.

    Args:
        base_model_path: Path to base GLM-Image model (or pipeline root)
        lora_path: Path to trained LoRA weights
        output_path: Path to save merged model
        torch_dtype: Data type for the merged model
    """
    from peft import PeftModel
    from transformers import AutoProcessor, GlmImageForConditionalGeneration

    # Detect format
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    if is_diffusers_format:
        # Load from vision_language_encoder subdirectory
        model_subdir = os.path.join(base_model_path, "vision_language_encoder")
        processor_subdir = os.path.join(base_model_path, "processor")

        logging.info(f"Loading base AR model from {model_subdir}")
        base_model = GlmImageForConditionalGeneration.from_pretrained(
            model_subdir,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        logging.info(f"Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)

        logging.info("Merging weights...")
        merged_model = model.merge_and_unload()

        logging.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(output_path)

        # Save processor
        processor = AutoProcessor.from_pretrained(processor_subdir)
        processor.save_pretrained(output_path)
    else:
        # Standard transformers format
        logging.info(f"Loading base model from {base_model_path}")
        base_model = GlmImageForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        logging.info(f"Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)

        logging.info("Merging weights...")
        merged_model = model.merge_and_unload()

        logging.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(output_path)

        # Also save processor
        processor = AutoProcessor.from_pretrained(base_model_path)
        processor.save_pretrained(output_path)

    logging.info("Done!")


def generate_with_lora(
    model_path: str,
    lora_path: str,
    prompt: str,
    output_path: str = "output.png",
    **generation_kwargs,
):
    """
    Generate an image using a LoRA-finetuned model.

    Args:
        model_path: Path to base GLM-Image model
        lora_path: Path to trained LoRA weights
        prompt: Text prompt for generation
        output_path: Path to save generated image
        **generation_kwargs: Additional generation parameters
    """
    from diffusers import GlmImagePipeline
    from peft import PeftModel
    from transformers import GlmImageForConditionalGeneration

    # Load pipeline
    pipe = GlmImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # CRITICAL: LoRA was trained on GlmImageForConditionalGeneration, so we need to
    # load it the same way to ensure weight names match correctly.
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        model_subdir = os.path.join(model_path, "vision_language_encoder")
    else:
        model_subdir = model_path

    # Load base model and apply LoRA
    base_model = GlmImageForConditionalGeneration.from_pretrained(
        model_subdir,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = peft_model.merge_and_unload()

    # Replace pipeline's encoder with merged model
    pipe.vision_language_encoder = merged_model
    pipe = pipe.to("cuda")

    # Generate
    default_kwargs = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 1.5,
    }
    default_kwargs.update(generation_kwargs)

    image = pipe(prompt=prompt, **default_kwargs).images[0]
    image.save(output_path)

    logging.info(f"Image saved to {output_path}")
    return image


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GLM-Image LoRA Finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="zai-org/GLM-Image",
        help="Path to base GLM-Image model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    # LoRA
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank (r). For image generation, 32-128 recommended",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling factor. Typically 2x rank",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
        help="Modules to apply LoRA to. Include MLP layers for better style learning",
    )

    # Dataset
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="pokemon",
        help="Dataset to use: 'pokemon', 'pixel-art', 'chinese-landscape', 'line-art' or 'data_1024_10K'",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--task_type", type=str, default="t2i", choices=["t2i", "i2i"])
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=None,
        help="Path to local dataset (optional)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded datasets (default: finetune/data/)",
    )

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-GPU batch size. Increase for better GPU utilization (requires more VRAM)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps. Effective batch = batch_size * accumulation",
    )
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Optimization
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/glm-image-lora")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # Utility commands
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights instead of training",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (for merging/inference)",
    )
    parser.add_argument(
        "--merged_output_path", type=str, default=None, help="Path to save merged model"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle utility commands
    if args.merge_lora:
        if not args.lora_path or not args.merged_output_path:
            print("Error: --lora_path and --merged_output_path required for merging")
            sys.exit(1)

        merge_lora_weights(
            base_model_path=args.model_path,
            lora_path=args.lora_path,
            output_path=args.merged_output_path,
        )
        return

    # Create training config from args
    config = TrainingConfig(
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        dataset_subset=args.dataset_subset,
        resolution=args.resolution,
        task_type=args.task_type,
        local_data_dir=args.local_data_dir,
        cache_dir=args.cache_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        use_8bit_adam=args.use_8bit_adam,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Train
    trainer = GlmImageLoraTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
