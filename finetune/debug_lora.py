#!/usr/bin/env python3
"""
Debug script for GLM-Image LoRA checkpoints.

This script helps diagnose issues with LoRA training and inference.

Usage:
    python debug_lora.py --model_path /path/to/GLM-Image --lora_path ./outputs/checkpoint-1000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def check_checkpoint_files(lora_path: str):
    """Check if checkpoint directory contains valid files."""
    logging.info(f"\n{'=' * 60}")
    logging.info("1. Checking checkpoint files...")
    logging.info(f"{'=' * 60}")

    path = Path(lora_path)

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",  # or .bin
    ]

    optional_files = [
        "training_state.json",
        "tokenizer_config.json",
    ]

    # Check required files
    for f in required_files:
        file_path = path / f
        alt_path = path / f.replace(".safetensors", ".bin")

        if file_path.exists():
            size = file_path.stat().st_size / 1024 / 1024
            logging.info(f"  ✓ {f} ({size:.2f} MB)")
        elif alt_path.exists():
            size = alt_path.stat().st_size / 1024 / 1024
            logging.info(f"  ✓ {alt_path.name} ({size:.2f} MB)")
        else:
            logging.error(f"  ✗ {f} NOT FOUND")
            return False

    # Check optional files
    for f in optional_files:
        file_path = path / f
        if file_path.exists():
            logging.info(f"  ○ {f} (optional)")

    return True


def analyze_adapter_config(lora_path: str):
    """Analyze the adapter configuration."""
    logging.info(f"\n{'=' * 60}")
    logging.info("2. Analyzing adapter configuration...")
    logging.info(f"{'=' * 60}")

    config_path = Path(lora_path) / "adapter_config.json"

    with open(config_path) as f:
        config = json.load(f)

    logging.info(f"  LoRA rank (r): {config.get('r')}")
    logging.info(f"  LoRA alpha: {config.get('lora_alpha')}")
    logging.info(f"  LoRA dropout: {config.get('lora_dropout')}")
    logging.info(f"  Target modules: {config.get('target_modules')}")
    logging.info(f"  Task type: {config.get('task_type')}")
    logging.info(f"  Bias: {config.get('bias')}")

    # Calculate effective scaling
    r = config.get("r", 8)
    alpha = config.get("lora_alpha", 16)
    scaling = alpha / r
    logging.info(f"  Effective scaling (alpha/r): {scaling}")

    if scaling > 4:
        logging.warning(f"  ⚠ High scaling ({scaling}) may cause instability")

    return config


def load_and_check_weights(lora_path: str):
    """Load and analyze LoRA weights."""
    logging.info(f"\n{'=' * 60}")
    logging.info("3. Analyzing LoRA weights...")
    logging.info(f"{'=' * 60}")

    # Find the weights file
    path = Path(lora_path)
    weights_file = path / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = path / "adapter_model.bin"

    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(weights_file))
    else:
        weights = torch.load(weights_file, map_location="cpu")

    logging.info(f"  Number of LoRA parameters: {len(weights)}")

    # Analyze weight statistics
    total_params = 0
    weight_stats = {}

    for name, tensor in weights.items():
        total_params += tensor.numel()

        # Get module name
        parts = name.split(".")
        module_type = "unknown"
        for part in parts:
            if part in ["lora_A", "lora_B"]:
                module_type = (
                    parts[parts.index(part) - 1] if parts.index(part) > 0 else "unknown"
                )
                break

        if module_type not in weight_stats:
            weight_stats[module_type] = {"count": 0, "params": 0, "norms": []}

        weight_stats[module_type]["count"] += 1
        weight_stats[module_type]["params"] += tensor.numel()
        weight_stats[module_type]["norms"].append(tensor.float().norm().item())

    logging.info(f"  Total trainable parameters: {total_params:,}")

    logging.info("\n  Weight statistics by module:")
    for module, stats in weight_stats.items():
        avg_norm = sum(stats["norms"]) / len(stats["norms"]) if stats["norms"] else 0
        max_norm = max(stats["norms"]) if stats["norms"] else 0
        logging.info(
            f"    {module}: {stats['params']:,} params, avg_norm={avg_norm:.4f}, max_norm={max_norm:.4f}"
        )

        if max_norm > 100:
            logging.warning(
                f"    ⚠ High weight norm in {module} - may indicate training issues"
            )

    return weights


def test_model_loading(model_path: str, lora_path: str):
    """Test loading the model with LoRA."""
    logging.info(f"\n{'=' * 60}")
    logging.info("4. Testing model loading...")
    logging.info(f"{'=' * 60}")

    try:
        from peft import PeftModel
        from transformers import AutoProcessor, GlmImageForConditionalGeneration

        # Detect model format
        model_index = Path(model_path) / "model_index.json"
        if model_index.exists():
            model_subdir = Path(model_path) / "vision_language_encoder"
            logging.info(f"  Loading from diffusers format: {model_subdir}")
        else:
            model_subdir = model_path
            logging.info(f"  Loading from transformers format: {model_subdir}")

        # Load base model
        logging.info("  Loading base model...")
        model = GlmImageForConditionalGeneration.from_pretrained(
            model_subdir,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        logging.info(f"  ✓ Base model loaded: {model.__class__.__name__}")

        # Load LoRA
        logging.info("  Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(model, lora_path)
        logging.info(f"  ✓ PEFT model loaded: {peft_model.__class__.__name__}")

        # Check adapter modules
        logging.info("\n  Adapter modules found:")
        adapter_modules = []
        for name, module in peft_model.named_modules():
            if "lora" in name.lower():
                adapter_modules.append(name)

        # Group by layer
        layer_counts = {}
        for name in adapter_modules:
            # Extract layer number
            import re

            match = re.search(r"layers\.(\d+)", name)
            if match:
                layer = int(match.group(1))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        if layer_counts:
            logging.info(f"    LoRA applied to {len(layer_counts)} layers")
            logging.info(
                f"    Layer range: {min(layer_counts.keys())} - {max(layer_counts.keys())}"
            )

        # Test merge
        logging.info("\n  Testing merge_and_unload...")
        merged_model = peft_model.merge_and_unload()
        logging.info(f"  ✓ Merged model: {merged_model.__class__.__name__}")

        del model, peft_model, merged_model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logging.error(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_generation(
    model_path: str, lora_path: str = None, device: str = "cuda"
):
    """Test simple token generation (not full image)."""
    logging.info(f"\n{'=' * 60}")
    logging.info("5. Testing token generation...")
    logging.info(f"{'=' * 60}")

    try:
        from peft import PeftModel
        from transformers import AutoProcessor, GlmImageForConditionalGeneration

        # Detect model format
        model_index = Path(model_path) / "model_index.json"
        if model_index.exists():
            model_subdir = Path(model_path) / "vision_language_encoder"
            processor_subdir = Path(model_path) / "processor"
        else:
            model_subdir = model_path
            processor_subdir = model_path

        # Load model
        model = GlmImageForConditionalGeneration.from_pretrained(
            model_subdir,
            torch_dtype=torch.bfloat16,
        )

        processor = AutoProcessor.from_pretrained(processor_subdir)

        if lora_path:
            logging.info(f"  Applying LoRA from {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()

        model = model.to(device)
        model.eval()

        # Prepare simple input
        prompt = "A cat"
        inputs = processor(
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate a few tokens
        logging.info(f"  Generating tokens for prompt: '{prompt}'")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Just generate a few tokens
                do_sample=False,
            )

        generated_tokens = outputs[0].tolist()
        logging.info(f"  ✓ Generated {len(generated_tokens)} tokens")

        # Check for image tokens
        image_start_id = getattr(model.config, "image_start_token_id", 16384)
        image_end_id = getattr(model.config, "image_end_token_id", 16385)

        has_image_start = image_start_id in generated_tokens
        has_image_end = image_end_id in generated_tokens

        logging.info(
            f"  Image start token ({image_start_id}) present: {has_image_start}"
        )
        logging.info(f"  Image end token ({image_end_id}) present: {has_image_end}")

        # Count image tokens
        image_tokens = [t for t in generated_tokens if 0 <= t < 16384]
        logging.info(f"  Image tokens (0-16383) generated: {len(image_tokens)}")

        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logging.error(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug GLM-Image LoRA checkpoint")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to base GLM-Image model"
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip_generation", action="store_true", help="Skip generation test (faster)"
    )

    args = parser.parse_args()

    logging.info("GLM-Image LoRA Checkpoint Debugger")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"LoRA path: {args.lora_path}")

    results = {}

    # Check 1: Files
    results["files"] = check_checkpoint_files(args.lora_path)
    if not results["files"]:
        logging.error("Checkpoint files missing. Aborting.")
        return

    # Check 2: Config
    config = analyze_adapter_config(args.lora_path)
    results["config"] = config is not None

    # Check 3: Weights
    weights = load_and_check_weights(args.lora_path)
    results["weights"] = weights is not None

    # Check 4: Model loading
    results["loading"] = test_model_loading(args.model_path, args.lora_path)

    # Check 5: Generation test
    if not args.skip_generation:
        logging.info("\n  Testing base model generation...")
        results["base_gen"] = test_simple_generation(
            args.model_path, lora_path=None, device=args.device
        )

        logging.info("\n  Testing LoRA model generation...")
        results["lora_gen"] = test_simple_generation(
            args.model_path, lora_path=args.lora_path, device=args.device
        )

    # Summary
    logging.info(f"\n{'=' * 60}")
    logging.info("SUMMARY")
    logging.info(f"{'=' * 60}")

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"  {check}: {status}")

    if all(results.values()):
        logging.info("\n✓ All checks passed!")
        logging.info("If inference still fails, the issue may be:")
        logging.info("  1. Training destabilized the model (try earlier checkpoint)")
        logging.info("  2. Learning rate too high (try 1e-5 instead of 1e-4)")
        logging.info("  3. LoRA rank too high (try rank=8 or 16)")
    else:
        logging.error("\n✗ Some checks failed. See above for details.")


if __name__ == "__main__":
    main()
