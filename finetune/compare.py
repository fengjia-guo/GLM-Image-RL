#!/usr/bin/env python3
"""
GLM-Image LoRA Comparison Tool

Compare image generation results between:
1. Base model (without LoRA)
2. LoRA-finetuned model (from checkpoint)

This helps evaluate training progress and quality improvements.

Usage:
    # Compare single prompt
    python compare.py \
        --model_path /path/to/GLM-Image \
        --lora_path ./outputs/glm-image-lora/checkpoint-1000 \
        --prompt "A cat sitting on a windowsill"

    # Compare multiple prompts from file
    python compare.py \
        --model_path /path/to/GLM-Image \
        --lora_path ./outputs/glm-image-lora/checkpoint-1000 \
        --prompt_file prompts.txt \
        --output_dir ./comparison_results

    # Compare multiple checkpoints
    python compare.py \
        --model_path /path/to/GLM-Image \
        --lora_paths checkpoint-500 checkpoint-1000 checkpoint-2000 \
        --prompt "A beautiful sunset over the ocean"
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_pipeline(
    model_path: str,
    lora_path: Optional[str] = None,
    device: str = "cuda",
    merge_lora: bool = True,
):
    """
    Load GLM-Image pipeline with optional LoRA weights.

    The LoRA was trained on GlmImageForConditionalGeneration (transformers format),
    so we need to load it the same way and then replace the component in the pipeline.

    Args:
        model_path: Path to base GLM-Image model
        lora_path: Optional path to LoRA checkpoint
        device: Device to load model on
        merge_lora: Whether to merge LoRA weights (recommended for inference)

    Returns:
        GlmImagePipeline with or without LoRA
    """
    from diffusers import GlmImagePipeline

    logging.info(f"Loading pipeline from {model_path}")

    pipe = GlmImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        logging.info(f"Loading LoRA weights from {lora_path}")
        from peft import PeftModel
        from transformers import GlmImageForConditionalGeneration

        # Check adapter files exist
        adapter_config = Path(lora_path) / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(f"No adapter_config.json found in {lora_path}")

        # Load adapter config to verify
        import json

        with open(adapter_config) as f:
            config = json.load(f)
        logging.info(
            f"LoRA config: r={config.get('r')}, alpha={config.get('lora_alpha')}, "
            f"target_modules={config.get('target_modules')}"
        )

        # CRITICAL: LoRA was trained on GlmImageForConditionalGeneration, NOT on
        # pipe.vision_language_encoder directly. The model structures differ:
        # - Training: GlmImageForConditionalGeneration loaded from vision_language_encoder/
        # - Pipeline: pipe.vision_language_encoder is the same model class
        #
        # But we need to load LoRA using the SAME loading path as training to ensure
        # weight names match correctly.

        # Detect model format and load base model the same way as training
        model_index_path = Path(model_path) / "model_index.json"
        if model_index_path.exists():
            # Diffusers format - load from vision_language_encoder subdirectory
            model_subdir = Path(model_path) / "vision_language_encoder"
            logging.info(f"Loading base model from {model_subdir} (diffusers format)")
        else:
            # Transformers format - load directly
            model_subdir = model_path
            logging.info(
                f"Loading base model from {model_subdir} (transformers format)"
            )

        # Load the model the same way as during training
        base_model = GlmImageForConditionalGeneration.from_pretrained(
            model_subdir,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        # Apply LoRA weights
        logging.info("Applying LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model, lora_path)

        if merge_lora:
            # Merge LoRA weights and unload for correct inference
            logging.info("Merging LoRA weights into base model...")
            merged_model = peft_model.merge_and_unload()

            # Replace the pipeline's vision_language_encoder with merged model
            pipe.vision_language_encoder = merged_model
        else:
            logging.info("Using LoRA in adapter mode (not merged)")
            pipe.vision_language_encoder = peft_model

    pipe = pipe.to(device)

    return pipe


def generate_image(
    pipe,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: Optional[int] = None,
) -> Optional[Image.Image]:
    """Generate a single image. Returns None if generation fails."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
    except RuntimeError as e:
        logging.error(f"Generation failed: {e}")
        logging.error("This may indicate the LoRA training has destabilized the model.")
        logging.error(
            "Try: 1) Using an earlier checkpoint, 2) Training with smaller learning rate, "
            "3) Using lower LoRA rank"
        )
        return None


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    prompt: str,
    max_label_width: int = 200,
) -> Image.Image:
    """
    Create a side-by-side comparison grid with labels.

    Args:
        images: List of images to compare
        labels: Labels for each image (e.g., "Base", "LoRA-1000")
        prompt: The prompt used for generation
        max_label_width: Maximum width for label text

    Returns:
        Combined comparison image
    """
    if not images:
        raise ValueError("No images provided")

    n_images = len(images)
    img_width, img_height = images[0].size

    # Layout parameters
    padding = 20
    label_height = 40
    prompt_height = 60

    # Calculate grid dimensions
    total_width = n_images * img_width + (n_images + 1) * padding
    total_height = img_height + label_height + prompt_height + 3 * padding

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception as e:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception as e:
            font = ImageFont.load_default()

    # Draw prompt at top
    prompt_text = f"Prompt: {prompt}"
    if len(prompt_text) > 100:
        prompt_text = prompt_text[:97] + "..."
    draw.text((padding, padding), prompt_text, fill=(0, 0, 0), font=font)

    # Draw images and labels
    y_offset = prompt_height + padding
    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = padding + i * (img_width + padding)

        # Paste image
        canvas.paste(img, (x_offset, y_offset + label_height))

        # Draw label above image
        label_x = x_offset + img_width // 2
        draw.text((label_x, y_offset), label, fill=(0, 0, 0), font=font, anchor="mt")

    return canvas


def compare_single_prompt(
    model_path: str,
    lora_paths: List[str],
    prompt: str,
    output_path: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: int = 42,
    include_base: bool = True,
    device: str = "cuda",
):
    """
    Generate comparison images for a single prompt.

    Args:
        model_path: Path to base model
        lora_paths: List of LoRA checkpoint paths
        prompt: Text prompt
        output_path: Path to save comparison image
        include_base: Whether to include base model (no LoRA) in comparison
    """
    images = []
    labels = []

    # Generate with base model
    if include_base:
        logging.info("Generating with base model...")
        pipe = load_pipeline(model_path, lora_path=None, device=device)
        img = generate_image(
            pipe, prompt, height, width, num_inference_steps, guidance_scale, seed
        )
        images.append(img)
        labels.append("Base Model")

        # Clear memory
        del pipe
        torch.cuda.empty_cache()

    # Generate with each LoRA checkpoint
    for lora_path in lora_paths:
        checkpoint_name = Path(lora_path).name
        logging.info(f"Generating with {checkpoint_name}...")

        try:
            pipe = load_pipeline(model_path, lora_path=lora_path, device=device)
            img = generate_image(
                pipe, prompt, height, width, num_inference_steps, guidance_scale, seed
            )

            if img is not None:
                images.append(img)
                labels.append(checkpoint_name)
            else:
                logging.warning(f"Skipping {checkpoint_name} due to generation failure")
                # Create a placeholder image
                placeholder = Image.new("RGB", (width, height), color=(200, 200, 200))
                draw = ImageDraw.Draw(placeholder)
                draw.text(
                    (width // 2, height // 2),
                    f"{checkpoint_name}\nFailed",
                    fill=(100, 100, 100),
                    anchor="mm",
                )
                images.append(placeholder)
                labels.append(f"{checkpoint_name} (FAILED)")
        except Exception as e:
            logging.error(f"Error loading {checkpoint_name}: {e}")
            continue
        finally:
            # Clear memory
            if "pipe" in locals():
                del pipe
            torch.cuda.empty_cache()

    # Create comparison grid
    logging.info("Creating comparison grid...")
    comparison = create_comparison_grid(images, labels, prompt)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    comparison.save(output_path)
    logging.info(f"Saved comparison to {output_path}")

    # Also save individual images
    base_path = Path(output_path)
    for img, label in zip(images, labels):
        individual_path = (
            base_path.parent
            / f"{base_path.stem}_{label.replace(' ', '_')}{base_path.suffix}"
        )
        img.save(individual_path)
        logging.info(f"Saved {individual_path}")

    return comparison


def compare_from_prompts_file(
    model_path: str,
    lora_paths: List[str],
    prompts_file: str,
    output_dir: str,
    **kwargs,
):
    """
    Generate comparisons for multiple prompts from a file.

    Args:
        prompts_file: Path to file with one prompt per line
        output_dir: Directory to save comparison images
    """
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Processing prompt {i + 1}/{len(prompts)}")
        logging.info(f"Prompt: {prompt[:50]}...")

        output_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
        compare_single_prompt(
            model_path=model_path,
            lora_paths=lora_paths,
            prompt=prompt,
            output_path=output_path,
            **kwargs,
        )

    logging.info(f"\nAll comparisons saved to {output_dir}")


def find_checkpoints(output_dir: str) -> List[str]:
    """Find all checkpoint directories in output directory."""
    checkpoints = []
    output_path = Path(output_dir)

    if not output_path.exists():
        return checkpoints

    for item in output_path.iterdir():
        if item.is_dir() and (
            item.name.startswith("checkpoint-") or item.name.startswith("epoch-")
        ):
            # Check if it contains LoRA weights
            if (item / "adapter_config.json").exists():
                checkpoints.append(str(item))

    # Sort by step number
    def get_step(path):
        name = Path(path).name
        if name.startswith("checkpoint-"):
            return int(name.split("-")[1])
        elif name.startswith("epoch-"):
            return int(name.split("-")[1]) * 1000000  # Epochs come after steps
        return 0

    checkpoints.sort(key=get_step)
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Compare GLM-Image generation with and without LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to base GLM-Image model"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to single LoRA checkpoint"
    )
    parser.add_argument(
        "--lora_paths",
        nargs="+",
        default=None,
        help="Paths to multiple LoRA checkpoints for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="LoRA output directory to auto-find checkpoints",
    )

    # Prompts
    parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt to generate"
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None, help="File with prompts (one per line)"
    )

    # Generation settings
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible generation"
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./comparisons",
        help="Directory to save comparison images",
    )
    parser.add_argument(
        "--no_base", action="store_true", help="Don't include base model in comparison"
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Validate inputs
    if not args.prompt and not args.prompt_file:
        parser.error("Either --prompt or --prompt_file is required")

    # Collect LoRA paths
    lora_paths = []
    if args.lora_path:
        lora_paths.append(args.lora_path)
    if args.lora_paths:
        lora_paths.extend(args.lora_paths)
    if args.output_dir:
        found = find_checkpoints(args.output_dir)
        logging.info(f"Found {len(found)} checkpoints in {args.output_dir}")
        lora_paths.extend(found)

    if not lora_paths:
        logging.warning(
            "No LoRA paths specified. Will only generate base model output."
        )

    # Generate comparisons
    kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "include_base": not args.no_base,
        "device": args.device,
    }

    if args.prompt_file:
        compare_from_prompts_file(
            model_path=args.model_path,
            lora_paths=lora_paths,
            prompts_file=args.prompt_file,
            output_dir=args.save_dir,
            **kwargs,
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.save_dir, f"comparison_{timestamp}.png")
        compare_single_prompt(
            model_path=args.model_path,
            lora_paths=lora_paths,
            prompt=args.prompt,
            output_path=output_path,
            **kwargs,
        )


if __name__ == "__main__":
    main()
