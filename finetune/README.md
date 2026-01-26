# GLM-Image Finetuning Guide

This directory contains tools and examples for finetuning GLM-Image on custom datasets using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Overview

GLM-Image's architecture consists of two main components:

- **Autoregressive (AR) Generator**: A 9B-parameter model that generates compact visual token encodings (256 tokens → 1K-4K tokens)
- **Diffusion Decoder**: A 7B-parameter DiT-based decoder that transforms visual tokens into high-resolution images

**Finetuning Strategy**: We focus on adapting the **Autoregressive Generator** using LoRA (Low-Rank Adaptation), which efficiently modifies the model's behavior while keeping the majority of parameters frozen. The diffusion decoder remains unchanged, leveraging its pre-trained capabilities for high-quality image generation.

## Current Status

### ✅ Completed

- Project structure and directory setup
- Core training script (`finetune_lora.py`)
- Dataset utilities (`data_loader.py`)
- HQ dataset support (jackyhate/text-to-image-2M)
- LoRA integration via PEFT library
- Checkpoint saving and merging utilities

### 🚧 In Progress

- **Batch Support**: Homogeneous input batching implementation (PR under review in transformers)
- T2I training mode validation
- I2I training mode

### 📋 Planned Features

- [ ] DeepSpeed integration for distributed training
- [ ] LLaMA-Factory compatibility layer
- [ ] Comprehensive Jupyter notebook tutorial
- [ ] Model evaluation metrics
- [ ] Inference optimization

## Architecture Details

### Why Finetune the AR Component?

The autoregressive generator is responsible for:

1. **Semantic Understanding**: Interpreting text prompts and image conditions
2. **Compositional Layout**: Planning the spatial arrangement of visual elements
3. **Knowledge Integration**: Incorporating domain-specific knowledge and styles

By applying LoRA to the AR model's attention layers, we can:

- Adapt to new visual concepts and styles
- Improve text rendering for specific languages or fonts
- Specialize in domain-specific generation (e.g., medical images, design layouts)
- Enhance identity preservation for specific subjects

The diffusion decoder's high-frequency detail generation capabilities remain intact, ensuring consistent output quality.

### Training Flow Explained

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Target Image                                               │
│       ↓                                                     │
│  Vision Encoder → VQVAE.encode() → discrete tokens (0-16383)│
│                                           ↓                 │
│                                      [LABELS]               │
│                                           ↓                 │
│  Text Prompt → Tokenizer → input_ids → AR Model → logits   │
│                                           ↓                 │
│                           CrossEntropyLoss(logits, labels)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: We don't need the Diffusion Decoder during training because:

- The AR model learns to predict discrete "visual tokens" (like words in a visual language)
- The VQVAE converts images → tokens (training targets) and tokens → images (inference)
- The Diffusion Decoder is a fixed post-processor that refines the final output

**Vocabulary Design**:
| Token Range | Purpose |
|-------------|---------|
| 0-16383 | VQVAE codebook (discrete image tokens) |
| 16384 | `<image_start>` marker |
| 16385 | `<image_end>` / EOS |
| 16512+ | Text vocabulary |

## Supported Finetuning Modes

### Text-to-Image (T2I)

Finetune on paired text-image datasets to improve generation quality for specific domains, styles, or concepts.

**Use Cases**:

- Domain-specific generation (architecture, fashion, medical)
- Artistic style adaptation
- Enhanced text rendering for multilingual content
- Knowledge-intensive image generation

#### T2I Two-Stage Generation Architecture

In T2I mode, the AR model generates **two sets of image tokens**:

1. **Large Image Tokens** (`H×W`): Main target tokens used by Diffusion Decoder
2. **Small Image Tokens** (`H/2 × W/2`): Preview/reference tokens for guided generation

```
AR Generation Output:
┌──────────────────────────────────────────────────────────────┐
│ [Large Image Tokens: H×W] + [Small Image Tokens: H/2 × W/2] │
│         ↓                           ↓                        │
│   Upsample 2× → Diffusion      (preview reference)          │
│                   Decoder                                    │
└──────────────────────────────────────────────────────────────┘
```

#### Small Image Token Generation Methods

The small image tokens are derived from the large image. There are two approaches:

##### Method 1: Spatial Downsampling (Pooling)

```python
# Large tokens: (H, W) → Small tokens: (H/2, W/2)
small_tokens = F.avg_pool2d(large_tokens.float(), kernel_size=2, stride=2)
small_tokens = small_tokens.round().long()
```

| Pros                                      | Cons                                                    |
| ----------------------------------------- | ------------------------------------------------------- |
| ✅ Preserves local semantic relationships | ❌ Averaging may produce invalid token indices          |
| ✅ Smooth downsampling, less aliasing     | ❌ Blurs sharp token boundaries                         |
| ✅ Better for continuous feature spaces   | ❌ Tokens are discrete indices, averaging loses meaning |

##### Method 2: Direct Resize (Nearest Neighbor Interpolation)

```python
# Large tokens: (H, W) → Small tokens: (H/2, W/2)
large_2d = large_tokens.view(1, 1, H, W).float()
small_2d = F.interpolate(large_2d, size=(H//2, W//2), mode='nearest')
small_tokens = small_2d.view(-1).long()
```

| Pros                                            | Cons                                |
| ----------------------------------------------- | ----------------------------------- |
| ✅ Preserves exact token values (valid indices) | ❌ May introduce aliasing artifacts |
| ✅ Semantically consistent with VQVAE codebook  | ❌ Loses some spatial information   |
| ✅ Computationally efficient                    | ❌ Sharp boundaries, less smooth    |
| ✅ **Reversible**: Matches inference upsampling |                                     |

##### Recommendation: **Method 2 (Nearest Neighbor)**

For discrete tokens from VQVAE, **nearest neighbor interpolation is strongly preferred** because:

1. **Token Validity**: VQVAE tokens are discrete indices (0-16383). Averaging produces non-integer values that must be rounded, potentially mapping to semantically unrelated codebook entries.

2. **Consistency with Inference**: During inference, the pipeline uses `F.interpolate(mode='nearest')` to upsample small→large. Using the same method (in reverse) for training ensures alignment:

   ```python
   # Inference: small → large (upsample)
   large = F.interpolate(small, scale_factor=2, mode='nearest')

   # Training: large → small (downsample)
   small = F.interpolate(large, scale_factor=0.5, mode='nearest')  # ← should match
   ```

3. **Semantic Preservation**: Each VQVAE token represents a specific visual pattern from the codebook. Nearest neighbor sampling preserves these exact patterns, while pooling creates "hybrid" tokens that don't exist in the codebook.

```python
def create_small_image_tokens(large_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Create small image tokens from large image tokens via nearest neighbor downsampling.

    Args:
        large_tokens: (H*W,) tensor of token indices
        H, W: spatial dimensions of large image

    Returns:
        small_tokens: (H//2 * W//2,) tensor of token indices
    """
    large_2d = large_tokens.view(1, 1, H, W).float()
    small_2d = F.interpolate(large_2d, size=(H//2, W//2), mode='nearest')
    return small_2d.view(-1).long()
```

### Image-to-Image (I2I)

Finetune on image transformation tasks with text guidance using **HQ-Edit** dataset.

**Dataset**: [UCSC-VLAA/HQ-Edit](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit)

- 197K+ high-quality instruction-based image editing pairs
- Generated using GPT-4V and DALL-E 3
- Contains: source image, target image, edit instruction, descriptions

**Dataset Structure**:
| Field | Description |
|-------|-------------|
| `input_image` | Source image before editing |
| `output_image` | Target image after editing |
| `edit` | Editing instruction (e.g., "Change the sky to sunset") |
| `input` | Description of input image |
| `output` | Description of output image |
| `inverse_edit` | Reverse editing instruction |

**Use Cases**:

- Style transfer specialization
- Image editing and inpainting
- Subject-consistent generation
- Identity-preserving transformations

**Quick Usage**:

```python
from data_loader import DatasetConfig, HQEditDataset

config = DatasetConfig(
    task_type="i2i",
    resolution=1024,
    i2i_prompt_type="edit",  # or "output_description"
)
dataset = HQEditDataset(config)

for sample in dataset.iterate():
    print(f"Edit: {sample['edit_instruction']}")
    print(f"Source shape: {sample['source_image'].shape}")
    print(f"Target shape: {sample['target_image'].shape}")
    break
```

## Installation

```bash
# Install core dependencies
pip install -r requirement.txt

# Install transformers and diffusers from source (required for latest features)
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git

# Install PEFT for LoRA support
pip install peft
```

## LoRA Hyperparameter Guide

Choosing the right LoRA parameters is **critical for image generation quality**. Unlike text-based LLMs, image generation AR models require more capacity to learn visual patterns.

### Rank Selection

| Rank   | Trainable Params | Use Case                     | Notes                    |
| ------ | ---------------- | ---------------------------- | ------------------------ |
| 8      | ~8M              | Testing/debugging            | Too small for production |
| 16     | ~16M             | Light style tuning           | Minimal capacity         |
| **32** | ~32M             | **Recommended baseline**     | Good balance             |
| 64     | ~64M             | Strong style adaptation      | Better for domain shift  |
| 128    | ~130M            | Full fine-tuning alternative | Maximum expressiveness   |

**Recommendation**: Start with **rank=32** for image generation. Increase to 64-128 if:

- Training on very different visual domain (e.g., medical images, anime)
- Learning complex new styles or concepts
- Loss plateaus but quality still unsatisfactory

### Alpha Selection

The `alpha/rank` ratio controls the **effective learning rate** of LoRA:

| Alpha            | Effect                                | Stability       |
| ---------------- | ------------------------------------- | --------------- |
| `alpha = rank`   | Standard adaptation                   | Very stable     |
| `alpha = 2*rank` | **Recommended** - Stronger adaptation | Stable          |
| `alpha = 4*rank` | Aggressive adaptation                 | May be unstable |

**Formula**: `effective_lr = lr * (alpha / rank)`

With `alpha = 2*rank`, you get 2x the effective learning rate, allowing faster adaptation without destabilizing training.

### Target Module Selection

This is the **most important decision** for image generation finetuning.

#### Attention Layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)

| Aspect            | Impact                                                          |
| ----------------- | --------------------------------------------------------------- |
| **Function**      | Controls how tokens attend to each other                        |
| **Visual Effect** | Affects composition, spatial layout, object relationships       |
| **Best For**      | Compositional changes, spatial arrangement, multi-object scenes |
| **Parameters**    | ~4 matrices per layer                                           |

#### MLP Layers (`gate_up_proj`, `down_proj`)

| Aspect            | Impact                                                |
| ----------------- | ----------------------------------------------------- |
| **Function**      | Feature transformation, stores "knowledge"            |
| **Visual Effect** | Affects style, texture, color palette, fine details   |
| **Best For**      | **Style transfer, domain adaptation, visual quality** |
| **Parameters**    | ~3 matrices per layer (larger than attention!)        |

#### Comparison

| Target                 | Params (rank=32) | Best For            | Image Quality Impact |
| ---------------------- | ---------------- | ------------------- | -------------------- |
| Attention only         | ~32M             | Layout, composition | Medium               |
| MLP only               | ~48M             | Style, textures     | High                 |
| **Both (recommended)** | ~80M             | Full adaptation     | **Highest**          |

### Recommended Configurations

#### 🎯 Default (Balanced)

```bash
python finetune_lora.py \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_up_proj down_proj
```

- **Trainable params**: ~80M (0.9% of 9B model)
- **Use case**: General finetuning, style adaptation

#### 💾 Memory-Constrained

```bash
python finetune_lora.py \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj v_proj gate_up_proj down_proj
```

- **Trainable params**: ~32M
- **Use case**: Limited GPU memory, faster experimentation

#### 🎨 Style-Focused (Best for visual quality)

```bash
python finetune_lora.py \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_up_proj down_proj
```

- **Trainable params**: ~160M
- **Use case**: Strong style adaptation, domain transfer

#### 🔬 Maximum Capacity

```bash
python finetune_lora.py \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_up_proj down_proj
```

- **Trainable params**: ~320M (~3.5% of model)
- **Use case**: Near full finetuning, maximum adaptation

### Why MLP Matters for Image Generation

Research from Stable Diffusion LoRA training shows that **MLP layers are crucial for learning visual styles**:

1. **Style is stored in MLP**: Visual patterns, color palettes, and textures are encoded in MLP weights
2. **Attention handles structure**: Spatial relationships and composition are managed by attention
3. **Combined effect**: Training both gives the best results for most use cases

```
Text Prompt → [Attention: WHERE to place things]
           → [MLP: WHAT things look like]
           → Visual Tokens
```

### Quick Reference

| Scenario     | Rank   | Alpha  | Modules      | Params  |
| ------------ | ------ | ------ | ------------ | ------- |
| Quick test   | 8      | 16     | attn only    | 8M      |
| Light tuning | 16     | 32     | attn+mlp     | 40M     |
| **Standard** | **32** | **64** | **attn+mlp** | **80M** |
| Strong style | 64     | 128    | attn+mlp     | 160M    |
| Maximum      | 128    | 256    | attn+mlp     | 320M    |

## Batch Size & Training Efficiency

### Understanding Batch Size vs Gradient Accumulation

```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

| Method                    | GPU Utilization | Memory Usage | Speed           |
| ------------------------- | --------------- | ------------ | --------------- |
| batch_size=1, accum=4     | Low ❌          | Minimal      | Slow            |
| **batch_size=2, accum=2** | Medium ✅       | Moderate     | **Recommended** |
| batch_size=4, accum=1     | High ✅         | High         | Fast            |
| batch_size=8, accum=1     | Maximum ✅      | Very High    | Fastest         |

**Key Insight**: Same effective batch size, but **real batch_size > 1 trains faster** because:

- GPU parallel computation is better utilized
- Gradient accumulation only simulates large batch gradients
- Memory bandwidth is used more efficiently

### Recommended Configurations by GPU

| GPU           | VRAM  | batch_size | accum | Effective | Notes          |
| ------------- | ----- | ---------- | ----- | --------- | -------------- |
| RTX 3090/4090 | 24GB  | 1          | 4     | 4         | Limited VRAM   |
| A100 40GB     | 40GB  | 2          | 2     | 4         | Default        |
| **A100 80GB** | 80GB  | **4**      | **1** | **4**     | **Optimal**    |
| Multi-GPU     | 80GB+ | 4-8        | 1     | 4-8/GPU   | Use accelerate |

### Example Commands

```bash
# Default (A100 40GB or similar)
python finetune_lora.py \
    --batch_size 2 \
    --gradient_accumulation_steps 2

# High-memory GPU (A100 80GB)
python finetune_lora.py \
    --batch_size 4 \
    --gradient_accumulation_steps 1

# Low-memory GPU (RTX 3090/4090)
python finetune_lora.py \
    --batch_size 1 \
    --gradient_accumulation_steps 4

# Maximum throughput (multi-GPU with accelerate)
accelerate launch finetune_lora.py \
    --batch_size 4 \
    --gradient_accumulation_steps 1
```

### Memory Optimization Tips

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch_size first**, then increase gradient_accumulation
2. **Enable gradient checkpointing** (enabled by default): `--gradient_checkpointing`
3. **Use bf16 mixed precision** (default): `--mixed_precision bf16`
4. **Use 8-bit Adam**: `--use_8bit_adam` (requires bitsandbytes)

## Monitoring & Visualization

### Real-time Loss Plotting

Monitor training progress with live loss curves:

```bash
# Start live monitoring (auto-refresh every 5 seconds)
python plot_loss.py --log_dir ./outputs/glm-image-lora

# Adjust refresh interval
python plot_loss.py --log_dir ./outputs/glm-image-lora --refresh_interval 10

# View current state without refresh
python plot_loss.py --log_dir ./outputs/glm-image-lora --no_refresh

# Export plot to file
python plot_loss.py --log_dir ./outputs/glm-image-lora --export loss_curve.png

# Adjust smoothing (higher = smoother curve)
python plot_loss.py --log_dir ./outputs/glm-image-lora --smoothing 0.95
```

The plot shows:

- **Raw loss** (light blue): Actual loss values per step
- **Smoothed loss** (dark blue): Exponential moving average
- **Learning rate**: Current LR schedule progress

### Comparing Checkpoints

Use `compare.py` to evaluate training progress by comparing generated images:

```bash
# Compare base model vs single checkpoint
python compare.py \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/checkpoint-1000 \
    --prompt "A cat sitting on a windowsill" \
    --seed 42

# Compare multiple checkpoints
python compare.py \
    --model_path /path/to/GLM-Image \
    --lora_paths checkpoint-500 checkpoint-1000 checkpoint-2000 \
    --prompt "A beautiful sunset over the ocean"

# Auto-find all checkpoints in output directory
python compare.py \
    --model_path /path/to/GLM-Image \
    --output_dir ./outputs/glm-image-lora \
    --prompt "A futuristic cityscape"

# Compare with multiple prompts from file
python compare.py \
    --model_path /path/to/GLM-Image \
    --lora_path ./outputs/glm-image-lora/checkpoint-1000 \
    --prompt_file test_prompts.txt \
    --save_dir ./comparison_results
```

Example `test_prompts.txt`:

```
A cat sitting on a windowsill
A beautiful sunset over the ocean
A futuristic cityscape at night
Portrait of a woman with flowing hair
```

The comparison tool generates:

- **Side-by-side comparison image**: All models in one grid
- **Individual images**: Separate files for each model variant

## Quick Start

> **Note**: Full implementation is under active development. The following shows the intended usage pattern.

```python
# Basic T2I finetuning example
from finetune_lora import GlmImageLoraTrainer

trainer = GlmImageLoraTrainer(
    model_path="zai-org/GLM-Image",
    task_type="t2i",
    lora_rank=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
)

trainer.train(
    train_dataset="path/to/hq_dataset",
    output_dir="./outputs/glm-image-lora",
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-4,
)
```

## Dataset

### Recommended Dataset: text-to-image-2M

We use the [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset, a high-quality curated text-image pair dataset:

| Subset          | Size | Resolution | Description                                                   |
| --------------- | ---- | ---------- | ------------------------------------------------------------- |
| `data_1024_10K` | 10K  | 1024×1024  | High-quality images generated by Flux-dev with GPT-4o prompts |
| `data_512_2M`   | 2M   | 512×512    | Large-scale training data from multiple sources               |

**Quick usage:**

```python
from data_loader import DatasetConfig, TextToImage2MDataset

# Load 10K high-resolution dataset
# Data will be cached to finetune/data/ by default
config = DatasetConfig(
    subset="data_1024_10K",
    resolution=1024,
    streaming=False,  # Set to True for streaming mode
    cache_dir=None,  # Default: finetune/data/, or specify custom path
)
dataset = TextToImage2MDataset(config)

# Preview samples
for sample in dataset.iterate():
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Image shape: {sample['image'].shape}")
    break
```

**Cache Management:**

Downloaded datasets are cached in `finetune/data/` by default (not in system cache). This allows better control and cleanup:

```bash
# Check cache size
du -sh finetune/data/

# Clear cache to free disk space
rm -rf finetune/data/

# Specify custom cache location
python finetune_lora.py --cache_dir /path/to/custom/cache
```

### Custom Local Dataset Format

#### T2I Dataset Structure

```
dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── metadata.jsonl
```

**metadata.jsonl** format:

```jsonl
{"image": "images/image001.jpg", "prompt": "A detailed description of the image..."}
{"image": "images/image002.jpg", "prompt": "Another image description..."}
```

#### I2I Dataset Structure

```
dataset/
├── source_images/
│   ├── source001.jpg
│   └── ...
├── target_images/
│   ├── target001.jpg
│   └── ...
└── metadata.jsonl
```

**metadata.jsonl** format:

```jsonl
{
  "source": "source_images/source001.jpg",
  "target": "target_images/target001.jpg",
  "prompt": "Transform the snow forest to underground station..."
}
```

## LoRA Configuration

LoRA applies low-rank decomposition to the attention layers of the AR model:

```python
lora_config = {
    "r": 8,  # Rank of LoRA matrices (typically 4-16)
    "lora_alpha": 16,  # Scaling factor (usually 2*r)
    "target_modules": [
        "q_proj",  # Query projection in attention
        "k_proj",  # Key projection in attention
        "v_proj",  # Value projection in attention
        "o_proj",  # Output projection in attention
    ],
    "lora_dropout": 0.05,
    "bias": "none",
}
```

### Hyperparameter Guidelines

| Parameter       | Recommended Range | Description                                              |
| --------------- | ----------------- | -------------------------------------------------------- |
| `rank (r)`      | 4-16              | Higher ranks capture more complexity but increase memory |
| `lora_alpha`    | 2*r to 4*r        | Controls adaptation strength                             |
| `learning_rate` | 1e-5 to 1e-4      | Lower for subtle adaptations                             |
| `batch_size`    | 1-4 per GPU       | Limited by 80GB VRAM constraint                          |

## Advanced Features (Planned)

### DeepSpeed Integration

Support for ZeRO-2 and ZeRO-3 optimization stages to enable training on multi-GPU setups with reduced memory footprint.

```bash
# Example with DeepSpeed (planned)
deepspeed --num_gpus=8 finetune_lora.py \
    --deepspeed_config ds_config.json \
    --model_path zai-org/GLM-Image \
    --dataset path/to/hq_dataset
```

### LLaMA-Factory Compatibility

Integration with LLaMA-Factory ecosystem for streamlined finetuning workflows.

```yaml
# Example config for LLaMA-Factory (planned)
model_name_or_path: zai-org/GLM-Image
task_type: multimodal
finetuning_type: lora
lora_rank: 8
lora_target: q_proj,k_proj,v_proj,o_proj
```

## Hardware Requirements

| Setup              | VRAM  | Batch Size | Training Time (est.) |
| ------------------ | ----- | ---------- | -------------------- |
| Single H100 (80GB) | 80GB  | 1-2        | Baseline             |
| 2x H100 (80GB)     | 160GB | 4-8        | 2x faster            |
| 4x H100 (80GB)     | 320GB | 8-16       | 4x faster            |

**Note**: The AR model requires significant memory even with LoRA. Full model requires ~80GB for inference, LoRA finetuning requires similar amounts due to gradient computation.

## Development Roadmap

### Phase 1: Core Implementation (Current)

- [x] Project structure
- [ ] Basic LoRA finetuning script
- [ ] Dataset loading utilities
- [ ] Training loop with mixed precision
- [ ] Checkpoint saving and resuming

### Phase 2: Enhanced Features

- [ ] Multi-GPU support with DeepSpeed
- [ ] Gradient checkpointing for memory efficiency
- [ ] Validation and evaluation metrics
- [ ] Tensorboard/WandB logging
- [ ] Model merging utilities

### Phase 3: Ecosystem Integration

- [ ] LLaMA-Factory adapter
- [ ] HuggingFace Hub integration
- [ ] Pre-configured training recipes
- [ ] Community dataset support
- [ ] Inference optimization tools

### Phase 4: Advanced Techniques

- [ ] Full parameter finetuning option
- [ ] Mixed-precision training optimization
- [ ] Custom LoRA rank scheduling
- [ ] Multi-task training support

## Contributing

This is an active development area. Contributions are welcome! Please ensure:

- Code follows the existing style and structure
- New features include documentation and examples
- Training scripts are tested on representative datasets

## References

- [PEFT Library](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [GLM-Image Model Card](https://huggingface.co/zai-org/GLM-Image)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## License

This project follows the same license as the main GLM-Image repository.

## Support

For questions and discussions:

- GitHub Issues: Report bugs and request features
- WeChat/Discord: Join the community (see main README)
- Technical Blog: Check [z.ai/blog](https://z.ai/blog/glm-image) for updates
