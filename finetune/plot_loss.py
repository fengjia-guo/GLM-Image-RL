#!/usr/bin/env python3
"""
GLM-Image Training Loss Visualization

Plot training loss curves from loss_history.json generated during training.

Usage:
    # Plot from existing logs
    python plot_loss.py --log_dir ./outputs/glm-image-lora

    # Export plot to file
    python plot_loss.py --log_dir ./outputs/glm-image-lora --export loss_curve.png

    # Live monitoring during training
    python plot_loss.py --log_dir ./outputs/glm-image-lora --live
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def read_loss_history(log_dir: str) -> List[Dict]:
    """Read training metrics from loss_history.json."""
    loss_file = Path(log_dir) / "loss_history.json"

    if not loss_file.exists():
        logging.warning(f"No loss_history.json found in {log_dir}")
        return []

    try:
        with open(loss_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error reading {loss_file}: {e}")
        return []


def smooth_curve(values: List[float], weight: float = 0.9) -> List[float]:
    """Apply exponential moving average smoothing."""
    if not values:
        return []

    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(
    log_dir: str,
    output_path: Optional[str] = None,
    smoothing: float = 0.9,
    figsize: Tuple[int, int] = (12, 5),
    live: bool = False,
    refresh_interval: float = 10.0,
):
    """
    Plot training loss curves.

    Args:
        log_dir: Path to training output directory
        output_path: Path to save plot (optional)
        smoothing: Smoothing factor for loss curve (0-1)
        figsize: Figure size
        live: Enable live monitoring mode
        refresh_interval: Refresh interval in seconds for live mode
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("Matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    if live:
        plt.ion()
        logging.info(f"Live monitoring {log_dir} (refresh every {refresh_interval}s)")
        logging.info("Press Ctrl+C to stop")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("GLM-Image LoRA Training Progress", fontsize=14)

    try:
        while True:
            history = read_loss_history(log_dir)

            if not history:
                if live:
                    logging.info("Waiting for training data...")
                    time.sleep(refresh_interval)
                    continue
                else:
                    logging.error(f"No training data found in {log_dir}")
                    return

            # Extract data
            steps = [h["step"] for h in history]
            losses = [h["loss"] for h in history]
            lrs = [h["lr"] for h in history]

            # Clear and redraw
            axes[0].clear()
            axes[1].clear()

            # Plot loss
            smoothed = smooth_curve(losses, smoothing)
            axes[0].plot(steps, losses, "b-", alpha=0.3, label="Raw Loss")
            axes[0].plot(
                steps, smoothed, "b-", linewidth=2, label=f"Smoothed (α={smoothing})"
            )
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right")

            # Add statistics
            min_loss = min(losses)
            min_step = steps[losses.index(min_loss)]
            final_loss = losses[-1]
            final_smooth = smoothed[-1]

            stats_text = f"Current: {final_loss:.4f}\nSmoothed: {final_smooth:.4f}\nMin: {min_loss:.4f} (step {min_step})"
            axes[0].text(
                0.02,
                0.05,
                stats_text,
                transform=axes[0].transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # Plot learning rate
            axes[1].plot(steps, lrs, "g-", linewidth=2)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_title("Learning Rate Schedule")
            axes[1].grid(True, alpha=0.3)

            current_lr = lrs[-1]
            axes[1].text(
                0.98,
                0.98,
                f"Current LR: {current_lr:.2e}",
                transform=axes[1].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logging.info(f"Saved plot to {output_path}")
                if not live:
                    break

            if live:
                plt.pause(refresh_interval)
            else:
                plt.show()
                break

    except KeyboardInterrupt:
        logging.info("\nStopped monitoring")
    finally:
        plt.ioff()


def main():
    parser = argparse.ArgumentParser(
        description="Plot GLM-Image training loss curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./outputs/glm-image-lora",
        help="Training output directory",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="Smoothing factor for loss curve (0-1)",
    )
    parser.add_argument("--export", type=str, default=None, help="Export plot to file")
    parser.add_argument(
        "--live", action="store_true", help="Enable live monitoring mode"
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="Refresh interval in seconds for live mode",
    )

    args = parser.parse_args()

    plot_loss(
        log_dir=args.log_dir,
        output_path=args.export,
        smoothing=args.smoothing,
        live=args.live,
        refresh_interval=args.refresh,
    )


if __name__ == "__main__":
    main()
