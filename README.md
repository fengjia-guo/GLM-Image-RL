# GLM-Image-RL

Reinforcement learning finetuning for [GLM-Image](https://github.com/zai-org/GLM-Image) using GRPO (Group Relative Policy Optimization, from [DeepSeek-R1](https://arxiv.org/abs/2501.12948)).

Only the 9B autoregressive (AR) encoder is trained via LoRA; the 7B diffusion decoder stays frozen. Within each prompt group the decoder seed is fixed, so visual differences come solely from the AR encoder.

> This repo also includes the SFT (supervised fine-tuning) scripts from the upstream [GLM-Image](https://github.com/zai-org/GLM-Image) repo. See the [upstream finetune docs](https://github.com/zai-org/GLM-Image/tree/main/finetune) for SFT usage.

## Setup

```bash
# Install transformers & diffusers from source
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git

# Other dependencies
pip install -r requirements.txt
```

## Project Structure

```
finetune/
├── grpo_trainer.py        # Full GRPO training loop (rollout → reward → policy update)
├── grpo_rollout.py        # Rollout + optional reward scoring (no training)
├── finetune_lora.py       # SFT training script (from upstream)
├── data_loader.py         # Dataset loading utilities
├── compare.py             # Compare base vs LoRA model outputs
├── plot_loss.py           # Training loss visualization
└── rewards/
    ├── factory.py         # Reward model registry
    ├── base.py            # Base reward class
    ├── clip_score_reward.py   # CLIP text-image similarity (local)
    ├── hpsv3_reward.py    # HPSv3 reward (local or remote)
    ├── hpsv3_service.py   # HPSv3 standalone HTTP service
    └── remote_reward.py   # Generic remote reward client
```

## GRPO Training

### Single GPU

```bash
cd finetune

python grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models clip_score \
    --group_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-6 \
    --output_dir ./outputs/grpo
```

### Multi-GPU (via Accelerate)

```bash
accelerate launch --num_processes 4 finetune/grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --reward_models clip_score \
    --group_size 4 \
    --num_epochs 3 \
    --mixed_precision bf16 \
    --output_dir ./outputs/grpo
```

### Rollout-Only (no training)

Sample and score without updating the policy — useful for reward debugging or data collection:

```bash
cd finetune

python grpo_rollout.py \
    --model_path /path/to/GLM-Image \
    --prompt_jsonl prompts.jsonl \
    --group_size 4 \
    --reward_models clip_score \
    --reward_clip_model openai/clip-vit-large-patch14 \
    --reward_device cuda \
    --output_dir ./outputs/grpo
```

### Prompt Format

One JSON object per line. `metadata` is optional (defaults to 1024×1024):

```jsonl
{"prompt": "a cat sitting on a windowsill at sunset"}
{"prompt": "an astronaut riding a horse on mars", "metadata": {"height": 1024, "width": 1024}}
```

### GRPO Parameters

| Parameter                        | Default      | Description                                          |
| -------------------------------- | ------------ | ---------------------------------------------------- |
| `--group_size`                   | 4            | Samples per prompt (G)                               |
| `--clip_eps`                     | 0.2          | PPO-style clipping range                             |
| `--kl_coef`                      | 0.01         | KL penalty coefficient (β) vs frozen reference       |
| `--learning_rate`                | 5e-6         | Lower than SFT — RL updates are noisier              |
| `--num_inner_steps`              | 1            | Inner updates per rollout batch                      |
| `--num_epochs`                   | 3            | Outer epochs over the prompt dataset                 |
| `--gradient_accumulation_steps`  | 1            | Accumulate before optimizer step                     |
| `--warmup_ratio`                 | 0.05         | Fraction of total steps for LR warmup                |
| `--max_grad_norm`                | 1.0          | Gradient clipping norm                               |
| `--lora_rank`                    | 32           | LoRA rank for AR encoder                             |
| `--mixed_precision`              | bf16         | `no`, `fp16`, `bf16`                                 |
| `--skip_decode`                  | false        | Skip DiT+VAE decode (token-level rewards only)       |
| `--reward_models`                | clip_score   | Comma-separated: `clip_score`, `hpsv3`               |
| `--resume_lora_path`             | —            | Resume from an existing LoRA checkpoint               |
| `--report_to`                    | tensorboard  | `tensorboard`, `wandb`, `all`, `none`                |
| `--save_steps`                   | 50           | Save checkpoint every N optimizer steps              |

## Remote Services

The diffusion decoder and reward scoring can each be offloaded to separate HTTP services. This is useful when the decoder or reward model can't coexist in the same process (e.g. VRAM limits, dependency conflicts).

### Remote Decoder Service

Pass `--decoder_service_url` to offload DiT + VAE decoding:

```bash
python grpo_trainer.py \
    --model_path /path/to/GLM-Image \
    --decoder_service_url http://127.0.0.1:8012 \
    ...
```

The service must implement a `POST /decode` endpoint:

**Request:**

```json
{
  "token_ids": [[1, 2, 3, ...]],
  "prompt_context": {
    "prompt": "a cat sitting on a windowsill",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 1.5,
    "num_inference_steps": 50,
    "seed": 42
  }
}
```

- `token_ids`: list of token id sequences (each sequence is a list of ints from the AR encoder)
- `prompt_context`: generation parameters

**Response:**

```json
{
  "images": ["<base64-encoded PNG>"]
}
```

### Remote Reward Service

There are two modes:

#### 1. Generic reward service (`--reward_service_url`)

When `--reward_service_url` is set, **all** reward models are routed through it via `POST /score`, regardless of `--reward_models`:

```bash
python grpo_trainer.py \
    --reward_models clip_score \
    --reward_service_url http://127.0.0.1:9000 \
    ...
```

**Request:**

```json
{
  "prompt": "a cat sitting on a windowsill",
  "response": "",
  "generated_images": ["<base64-encoded PNG>"]
}
```

**Response:**

```json
{
  "score": 0.284
}
```

#### 2. HPSv3 service (built-in)

For HPSv3 specifically, a standalone service is included to avoid dependency conflicts:

```bash
# Terminal 1: start HPSv3 service (in an isolated env with hpsv3 installed)
python finetune/rewards/hpsv3_service.py --host 127.0.0.1 --port 8009 --device cuda

# Terminal 2: training
python finetune/grpo_trainer.py \
    --reward_models hpsv3 \
    --reward_service_url http://127.0.0.1:8009 \
    ...
```

HPSv3 service API (`POST /score`):

**Request:**

```json
{
  "prompts": ["prompt 1", "prompt 2"],
  "images_base64": ["<base64 PNG>", "<base64 PNG>"]
}
```

**Response:**

```json
{
  "ok": true,
  "scores": [0.312, 0.287]
}
```

Health check: `GET /health` → `{"ok": true}`

## Reward Models

| Model        | Type   | Notes                                              |
| ------------ | ------ | -------------------------------------------------- |
| `clip_score` | Local  | CLIP text-image similarity. Fast, good baseline.   |
| `hpsv3`      | Either | Human Preference Score v3. Local or via service.   |

## Using Trained Checkpoints

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
pipe.vision_language_encoder = peft_model.merge_and_unload()
pipe = pipe.to("cuda")

image = pipe("a beautiful sunset over mountains").images[0]
image.save("output.png")
```

> **Note:** LoRA was trained on `GlmImageForConditionalGeneration` — always load it that way. Loading directly onto `pipe.vision_language_encoder` may cause shape mismatches.

## Acknowledgements

This project is built on top of [GLM-Image](https://github.com/zai-org/GLM-Image) by [ZhipuAI](https://github.com/zai-org). The SFT scripts, data loader, and model architecture originate from that repo. See the [GLM-Image technical blog](https://z.ai/blog/glm-image) and [model card](https://huggingface.co/zai-org/GLM-Image) for details on the base model.

## License

This project inherits the [Apache 2.0](LICENSE) license from [GLM-Image](https://github.com/zai-org/GLM-Image).
