# GLM-Image LoRA Finetuning

This directory provides tools for finetuning GLM-Image using LoRA (Low-Rank Adaptation), including both **supervised fine-tuning (SFT)** and **reinforcement learning via GRPO**.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python finetune_lora.py \
    --model_path /path/to/GLM-Image \
    --dataset_subset pokemon \
    --output_dir ./outputs/glm-image-lora \
    --num_epochs 5 \
    --batch_size 4 \
    --lora_rank 32
```

### 3. Compare Results

```bash
python compare.py \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/epoch-5 \
    --prompt "A cute dragon"
```

### 4. Visualize Training

```bash
# Static plot
python plot_loss.py --log_dir ./outputs/glm-image-lora

# Live monitoring during training
python plot_loss.py --log_dir ./outputs/glm-image-lora --live

# Export to image
python plot_loss.py --log_dir ./outputs/glm-image-lora --export loss_curve.png
```

## Architecture Overview

GLM-Image consists of two components:

| Component         | Parameters | Role                                      | Trainable     |
| ----------------- | ---------- | ----------------------------------------- | ------------- |
| AR Model          | 9B         | Generates discrete image tokens from text | ✅ Yes (LoRA) |
| Diffusion Decoder | 7B         | Decodes tokens to pixels                  | ❌ Frozen     |

We apply LoRA to the AR model's attention and MLP layers:

- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_up_proj`, `down_proj` (MLP)

## SFT Training Details

### Training Flow

```
Target Image → Vision Encoder → VQVAE.encode() → discrete tokens (labels)
                                                        ↓
Text Prompt → Tokenizer → input_ids → AR Model → logits
                                                        ↓
                                        CrossEntropyLoss(logits, labels)
```

### Token Vocabulary

| Token Range | Purpose                       |
| ----------- | ----------------------------- |
| 0-16383     | VQVAE codebook (image tokens) |
| 16384       | `<image_start>` marker        |
| 16385       | `<image_end>` / EOS           |

## Files

| File               | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `finetune_lora.py` | Main SFT training script                                   |
| `grpo_rollout.py`  | GRPO rollout + optional reward scoring                     |
| `grpo_trainer.py`  | Full GRPO training loop (rollout → reward → policy update) |
| `data_loader.py`   | Dataset loading utilities                                  |
| `compare.py`       | Compare base vs LoRA model outputs                         |
| `plot_loss.py`     | Visualize training loss curves                             |
| `rewards/`         | Reward model implementations (e.g. HPSv3, CLIPScore)      |

## SFT Training Parameters

### Recommended Settings

| Parameter                       | Default | Description                               |
| ------------------------------- | ------- | ----------------------------------------- |
| `--lora_rank`                   | 32      | LoRA rank (8-128, higher = more capacity) |
| `--lora_alpha`                  | 64      | Scaling factor (typically 2× rank)        |
| `--learning_rate`               | 1e-4    | Learning rate                             |
| `--batch_size`                  | 4       | Per-GPU batch size                        |
| `--gradient_accumulation_steps` | 2       | Gradient accumulation                     |
| `--num_epochs`                  | 5       | Number of training epochs                 |

### Dataset Options

You can specify the dataset using `--dataset_subset`. We provide several style-specific datasets for quick demos:

| Subset              | Style     | Samples | Description                                     |
| ------------------- | --------- | ------- | ----------------------------------------------- |
| `pokemon`           | 🐉 Anime  | ~833    | **Default**. High quality Pokemon BLIP captions |
| `pixel-art`         | 👾 Pixel  | ~6K     | Pixel art characters and scenes                 |
| `chinese-landscape` | ⛰️ Ink    | ~1K     | Traditional Chinese landscape painting          |
| `line-art`          | ✏️ Sketch | ~1K     | Black and white line drawings                   |
| `data_1024_10K`     | 📷 Photo  | 10K     | High-quality photorealistic images              |

## Example SFT Training Log

```
2026-01-26 11:37:27 - INFO - Starting training...
2026-01-26 11:37:27 - INFO -   Epochs: 5
2026-01-26 11:37:27 - INFO -   Batch size: 4
2026-01-26 11:37:27 - INFO -   Gradient accumulation: 2
2026-01-26 11:37:27 - INFO -   Effective batch size: 8

Epoch 1/5: 100%|█████████| 2500/2500 [55:35<00:00, 1.33s/it, loss=1.28, lr=9.16e-05]
2026-01-26 12:33:03 - INFO - Epoch 1 completed. Average loss: 1.6421

Epoch 2/5: 100%|█████████| 2500/2500 [55:18<00:00, 1.33s/it, loss=1.15, lr=6.69e-05]
2026-01-26 13:28:25 - INFO - Epoch 2 completed. Average loss: 1.5128

...

2026-01-26 16:14:32 - INFO - Training completed!
```

---

## Reinforcement Learning with GRPO

GRPO (Group Relative Policy Optimization) fine-tunes the AR encoder using reward model feedback instead of supervised labels. This is the same algorithm used in DeepSeek-R1, adapted for text-to-image generation.

### How It Works

```
                    ┌─────────────────────────────────────────────┐
                    │              GRPO Training Loop              │
                    └─────────────────────────────────────────────┘

  Phase 1 — Rollout                Phase 2 — Score & Advantage         Phase 3 — Policy Update
  ─────────────────                ─────────────────────────────        ───────────────────────
  For each prompt q:               For each group:                     Re-compute log-probs
    Sample G images                  Score with reward model(s)          under updated π_θ
    from current π_θ                 Â_i = (r_i - μ) / σ              Clipped surrogate loss
    (AR encoder + DiT)               Compute ref log-probs              + KL penalty vs π_ref
                                     from frozen π_ref                  Backward + optimiser step
```

**Key design:**
- Only the AR encoder is trained (via LoRA); the DiT decoder is always frozen
- A frozen copy of the initial AR weights serves as the reference model π_ref for the KL penalty
- Within each group, the decoder seed is fixed — visual differences are solely attributable to the AR encoder
- Uses HuggingFace Accelerate for multi-GPU / mixed-precision support

### Quick Start (Single GPU)

```bash
python grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models clip_score \
    --group_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-6 \
    --output_dir ./outputs/grpo
```

### Multi-GPU Training (via Accelerate)

```bash
# Generate accelerate config (one-time)
accelerate config

# Launch on multiple GPUs
accelerate launch --num_processes 4 grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models clip_score \
    --group_size 4 \
    --num_epochs 3 \
    --mixed_precision bf16 \
    --output_dir ./outputs/grpo
```

Accelerate handles:
- Data parallelism across GPUs
- Mixed precision (bf16/fp16)
- Gradient accumulation and synchronization
- Distributed logging (TensorBoard / W&B)

### Reward Models

Reward models live under `finetune/rewards/`. Currently supported:

| Reward Model  | Type     | Notes                                                    |
| ------------- | -------- | -------------------------------------------------------- |
| `clip_score`  | Local    | CLIP text-image similarity. Fast, good baseline.         |
| `hpsv3`       | Service  | Human Preference Score v3. More accurate, needs isolation. |

#### Using CLIPScore (local)

```bash
python grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models clip_score \
    --reward_clip_model openai/clip-vit-large-patch14 \
    --output_dir ./outputs/grpo
```

#### Using HPSv3 (via remote service)

HPSv3 may conflict with your main `transformers` version. Run it as a separate service:

```bash
# Terminal 1: Start HPSv3 service (in isolated env)
python rewards/hpsv3_service.py --host 127.0.0.1 --port 8009 --device cuda

# Terminal 2: Run GRPO training (main env)
python grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models hpsv3 \
    --reward_service_url http://127.0.0.1:8009 \
    --group_size 4 \
    --num_epochs 3 \
    --output_dir ./outputs/grpo
```

#### Rollout-Only Mode (no training)

Use `grpo_rollout.py` to sample and score without updating the policy — useful for reward model debugging or data collection:

```bash
python grpo_rollout.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --group_size 4 \
    --reward_models clip_score \
    --reward_clip_model openai/clip-vit-large-patch14 \
    --reward_device cuda \
    --output_dir ./outputs/grpo
```

### Prompt JSONL Format

One JSON object per line. `metadata` is optional (defaults to 1024×1024):

```jsonl
{"prompt": "a cat sitting on a windowsill at sunset"}
{"prompt": "an astronaut riding a horse on mars", "metadata": {"height": 1024, "width": 1024}}
```

### GRPO Training Parameters

| Parameter                        | Default      | Description                                          |
| -------------------------------- | ------------ | ---------------------------------------------------- |
| `--group_size`                   | 4            | Samples per prompt (G). More = better advantages     |
| `--clip_eps`                     | 0.2          | PPO-style clipping range for importance ratio        |
| `--kl_coef`                      | 0.01         | KL penalty coefficient (β) vs frozen reference       |
| `--learning_rate`                | 5e-6         | Lower than SFT — RL updates are noisier              |
| `--num_inner_steps`              | 1            | PPO-style inner updates per rollout batch            |
| `--num_epochs`                   | 3            | Outer epochs over the prompt dataset                 |
| `--gradient_accumulation_steps`  | 1            | Accumulate before optimizer step                     |
| `--warmup_ratio`                 | 0.05         | Fraction of total steps for LR warmup                |
| `--max_grad_norm`                | 1.0          | Gradient clipping norm                               |
| `--lora_rank`                    | 32           | LoRA rank for AR encoder                             |
| `--mixed_precision`              | bf16         | Accelerate mixed precision: `no`, `fp16`, `bf16`     |
| `--skip_decode`                  | false        | Skip DiT+VAE decode (token-level rewards only)       |
| `--reward_models`                | clip_score   | Comma-separated: `clip_score`, `hpsv3`               |
| `--resume_lora_path`             | —            | Resume training from an existing LoRA checkpoint      |
| `--report_to`                    | tensorboard  | Experiment tracking: `tensorboard`, `wandb`, `all`, `none` |
| `--save_steps`                   | 50           | Save checkpoint every N optimizer steps              |

### Using Trained GRPO Checkpoints

GRPO outputs standard LoRA checkpoints — load them the same way as SFT:

```python
from diffusers import GlmImagePipeline
from transformers import GlmImageForConditionalGeneration
from peft import PeftModel
import torch

pipe = GlmImagePipeline.from_pretrained("/path/to/GLM-Image", torch_dtype=torch.bfloat16)
base_model = GlmImageForConditionalGeneration.from_pretrained(
    "/path/to/GLM-Image/vision_language_encoder", torch_dtype=torch.bfloat16
)
peft_model = PeftModel.from_pretrained(base_model, "./outputs/grpo/final")
merged = peft_model.merge_and_unload()
pipe.vision_language_encoder = merged
pipe = pipe.to("cuda")

image = pipe("a beautiful sunset over mountains").images[0]
image.save("grpo_output.png")
```

### GRPO Training Log Example

```
============================================================
GRPO Training for GLM-Image (Accelerate)
============================================================
  Num processes: 4
  Device: cuda:0
  Mixed precision: bf16
  Prompts: 500
  Group size (G): 4
  Epochs: 3
  Prompts per batch: 1
  Inner steps: 1
  Gradient accumulation: 1
  Estimated total updates: 6000
  Learning rate: 5e-06
  Clip eps: 0.2
  KL coef (beta): 0.01
  Reward models: clip_score
============================================================

Epoch 1/3
============================================================

Phase 1: Rolling out samples...
Rollout complete: 500 groups in 1823.5s

Phase 2: Computing reference log-probs & advantages...
Reference log-probs computed in 412.3s

Phase 3: Updating policy...
  step=1 | loss=0.0312 (policy=0.0298, kl=0.0014) | clip=0.00 | reward=0.2841 | adv=1.204 | lr=5.00e-06
  step=2 | loss=0.0287 (policy=0.0271, kl=0.0016) | clip=0.01 | reward=0.3012 | adv=0.832 | lr=5.00e-06
  ...

Epoch 1 summary:
  Total loss: 0.0295
  Policy loss: 0.0279
  KL loss: 0.0016
  Mean reward: 0.2934
  Approx KL: 0.0015
  Clip fraction: 0.82%
  Time: rollout=1823.5s, ref=412.3s, update=356.1s, total=2591.9s
```

---

## Using Trained LoRA

### Option 1: Compare Script

```bash
python compare.py \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/epoch-5 \
    --prompt "Your prompt here" \
    --save_dir ./comparisons
```

### Option 2: Merge LoRA Weights

```bash
python finetune_lora.py \
    --merge_lora \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/epoch-5 \
    --merged_output_path ./merged_model
```

### Option 3: Python API

```python
from diffusers import GlmImagePipeline
from transformers import GlmImageForConditionalGeneration
from peft import PeftModel
import torch

# Load pipeline
pipe = GlmImagePipeline.from_pretrained(
    "/path/to/GLM-Image",
    torch_dtype=torch.bfloat16,
)

# Load and merge LoRA
base_model = GlmImageForConditionalGeneration.from_pretrained(
    "/path/to/GLM-Image/vision_language_encoder",
    torch_dtype=torch.bfloat16,
)
peft_model = PeftModel.from_pretrained(base_model, "./outputs/glm-image-lora/epoch-5")
merged_model = peft_model.merge_and_unload()

# Replace encoder in pipeline
pipe.vision_language_encoder = merged_model
pipe = pipe.to("cuda")

# Generate
image = pipe("A cat sitting on a windowsill").images[0]
image.save("output.png")
```

## Troubleshooting

### Generation fails after loading LoRA

**Error**: `shape '[1, 1, 32, 32]' is invalid for input of size 1`

**Solution**: Use the correct loading method. LoRA was trained on `GlmImageForConditionalGeneration`, so load it the same way:

```python
# ✅ Correct
base_model = GlmImageForConditionalGeneration.from_pretrained(model_path)
peft_model = PeftModel.from_pretrained(base_model, lora_path)
merged = peft_model.merge_and_unload()
pipe.vision_language_encoder = merged

# ❌ Wrong (may cause shape mismatch)
pipe.vision_language_encoder = PeftModel.from_pretrained(pipe.vision_language_encoder, lora_path)
```

### Debug checkpoint issues

```bash
python debug_lora.py \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/epoch-5
```

## Hardware Requirements

| Configuration          | VRAM  | Batch Size |
| ---------------------- | ----- | ---------- |
| Single GPU (A100 80GB) | ~60GB | 4          |
| Single GPU (A100 40GB) | ~35GB | 2          |

Enable gradient checkpointing (default) to reduce memory usage.
