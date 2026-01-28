# GLM-Image LoRA Finetuning

This directory provides tools for finetuning GLM-Image using LoRA (Low-Rank Adaptation).

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

## Training Details

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

| File               | Description                        |
| ------------------ | ---------------------------------- |
| `finetune_lora.py` | Main training script               |
| `data_loader.py`   | Dataset loading utilities          |
| `compare.py`       | Compare base vs LoRA model outputs |
| `plot_loss.py`     | Visualize training loss curves     |
| `debug_lora.py`    | Debug LoRA checkpoint issues       |

## Training Parameters

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

## Example Training Log

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
