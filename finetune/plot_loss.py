#!/usr/bin/env python3
"""
GLM-Image Training Loss Visualization

This script provides real-time loss curve visualization during training.
It monitors the TensorBoard logs and plots loss curves dynamically.

Features:
- Real-time loss curve plotting
- Smoothed loss with moving average
- Learning rate visualization
- Auto-refresh from TensorBoard logs
- Export to PNG/PDF

Usage:
    # Monitor training in real-time
    python plot_loss.py --log_dir ./outputs/glm-image-lora
    
    # Plot from existing logs (no refresh)
    python plot_loss.py --log_dir ./outputs/glm-image-lora --no_refresh
    
    # Export plot
    python plot_loss.py --log_dir ./outputs/glm-image-lora --export loss_curve.png
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def read_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read training metrics from TensorBoard event files.
    
    Returns:
        Dict mapping metric names to list of (step, value) tuples
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logging.error("TensorBoard not installed. Run: pip install tensorboard")
        sys.exit(1)
    
    # Find event files
    log_path = Path(log_dir)
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    
    if not event_files:
        logging.warning(f"No TensorBoard event files found in {log_dir}")
        return {}
    
    metrics = {}
    
    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            for tag in ea.Tags().get("scalars", []):
                if tag not in metrics:
                    metrics[tag] = []
                
                for event in ea.Scalars(tag):
                    metrics[tag].append((event.step, event.value))
        except Exception as e:
            logging.warning(f"Error reading {event_file}: {e}")
    
    # Sort by step
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])
    
    return metrics


def read_loss_from_log_file(log_file: str) -> List[Tuple[int, float]]:
    """
    Parse loss values from training log file as fallback.
    
    Looks for patterns like:
    - loss=1.5888
    - loss: 1.5888
    """
    import re
    
    losses = []
    step = 0
    
    loss_pattern = re.compile(r'loss[=:]\s*([0-9.]+)')
    step_pattern = re.compile(r'(\d+)/\d+.*loss')
    
    with open(log_file, 'r') as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                loss = float(match.group(1))
                
                # Try to extract step
                step_match = step_pattern.search(line)
                if step_match:
                    step = int(step_match.group(1))
                else:
                    step += 1
                
                losses.append((step, loss))
    
    return losses


def smooth_curve(values: List[float], weight: float = 0.9) -> List[float]:
    """
    Apply exponential moving average smoothing.
    
    Args:
        values: Raw values
        weight: Smoothing weight (higher = smoother)
    
    Returns:
        Smoothed values
    """
    smoothed = []
    last = values[0] if values else 0
    
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed


class LivePlotter:
    """Real-time training loss plotter."""
    
    def __init__(
        self,
        log_dir: str,
        refresh_interval: float = 5.0,
        smoothing: float = 0.9,
        figsize: Tuple[int, int] = (12, 6),
    ):
        self.log_dir = log_dir
        self.refresh_interval = refresh_interval
        self.smoothing = smoothing
        self.figsize = figsize
        
        # Import matplotlib
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use interactive backend
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            logging.error("Matplotlib not installed. Run: pip install matplotlib")
            sys.exit(1)
        
        self.fig = None
        self.axes = None
        self.lines = {}
        
    def setup_figure(self):
        """Initialize the figure and axes."""
        self.plt.ion()  # Enable interactive mode
        self.fig, self.axes = self.plt.subplots(1, 2, figsize=self.figsize)
        self.fig.suptitle("GLM-Image LoRA Training Progress", fontsize=14)
        
        # Loss plot
        self.axes[0].set_xlabel("Step")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].set_title("Training Loss")
        self.axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        self.axes[1].set_xlabel("Step")
        self.axes[1].set_ylabel("Learning Rate")
        self.axes[1].set_title("Learning Rate Schedule")
        self.axes[1].grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
    def update(self, metrics: Dict[str, List[Tuple[int, float]]]):
        """Update plots with new data."""
        if self.fig is None:
            self.setup_figure()
        
        # Clear axes
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plot loss
        loss_data = metrics.get("train/loss", [])
        avg_loss_data = metrics.get("train/avg_loss", [])
        
        if loss_data:
            steps, losses = zip(*loss_data)
            smoothed = smooth_curve(list(losses), self.smoothing)
            
            # Raw loss (light)
            self.axes[0].plot(steps, losses, 'b-', alpha=0.3, label='Raw Loss')
            # Smoothed loss (dark)
            self.axes[0].plot(steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (α={self.smoothing})')
            
            # Add current loss text
            current_loss = losses[-1]
            current_smooth = smoothed[-1]
            self.axes[0].axhline(y=current_smooth, color='r', linestyle='--', alpha=0.5)
            self.axes[0].text(
                0.98, 0.98,
                f'Current: {current_loss:.4f}\nSmoothed: {current_smooth:.4f}',
                transform=self.axes[0].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )
        
        self.axes[0].set_xlabel("Step")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].set_title("Training Loss")
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].legend(loc='upper right')
        
        # Plot learning rate
        lr_data = metrics.get("train/learning_rate", [])
        if lr_data:
            steps, lrs = zip(*lr_data)
            self.axes[1].plot(steps, lrs, 'g-', linewidth=2)
            
            current_lr = lrs[-1]
            self.axes[1].text(
                0.98, 0.98,
                f'Current LR: {current_lr:.2e}',
                transform=self.axes[1].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            )
        
        self.axes[1].set_xlabel("Step")
        self.axes[1].set_ylabel("Learning Rate")
        self.axes[1].set_title("Learning Rate Schedule")
        self.axes[1].grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def run(self, no_refresh: bool = False):
        """Main loop for live plotting."""
        logging.info(f"Monitoring {self.log_dir}")
        logging.info(f"Refresh interval: {self.refresh_interval}s")
        logging.info("Press Ctrl+C to stop")
        
        try:
            while True:
                metrics = read_tensorboard_logs(self.log_dir)
                
                if metrics:
                    self.update(metrics)
                    total_steps = len(metrics.get("train/loss", []))
                    logging.info(f"Updated plot with {total_steps} data points")
                else:
                    logging.info("Waiting for training data...")
                
                if no_refresh:
                    self.plt.ioff()
                    self.plt.show()
                    break
                
                self.plt.pause(self.refresh_interval)
                
        except KeyboardInterrupt:
            logging.info("\nStopped monitoring")
        finally:
            self.plt.ioff()
            
    def export(self, output_path: str):
        """Export current plot to file."""
        metrics = read_tensorboard_logs(self.log_dir)
        if metrics:
            self.update(metrics)
            self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved plot to {output_path}")
        else:
            logging.error("No data to export")


def plot_static(
    log_dir: str,
    output_path: Optional[str] = None,
    smoothing: float = 0.9,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Create a static plot from training logs.
    
    Args:
        log_dir: Path to training output directory
        output_path: Path to save plot (optional)
        smoothing: Smoothing factor for loss curve
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    metrics = read_tensorboard_logs(log_dir)
    
    if not metrics:
        logging.error(f"No metrics found in {log_dir}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("GLM-Image LoRA Training Progress", fontsize=14)
    
    # Plot loss
    loss_data = metrics.get("train/loss", [])
    if loss_data:
        steps, losses = zip(*loss_data)
        smoothed = smooth_curve(list(losses), smoothing)
        
        axes[0].plot(steps, losses, 'b-', alpha=0.3, label='Raw Loss')
        axes[0].plot(steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (α={smoothing})')
        axes[0].legend()
        
        # Statistics
        min_loss = min(losses)
        min_step = steps[losses.index(min_loss)]
        final_loss = losses[-1]
        
        stats_text = f'Min Loss: {min_loss:.4f} (step {min_step})\nFinal Loss: {final_loss:.4f}'
        axes[0].text(
            0.98, 0.98, stats_text,
            transform=axes[0].transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate
    lr_data = metrics.get("train/learning_rate", [])
    if lr_data:
        steps, lrs = zip(*lr_data)
        axes[1].plot(steps, lrs, 'g-', linewidth=2)
    
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot GLM-Image training loss curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--log_dir", type=str, default="./outputs/glm-image-lora",
                        help="Training output directory with TensorBoard logs")
    parser.add_argument("--refresh_interval", type=float, default=5.0,
                        help="Refresh interval in seconds for live plotting")
    parser.add_argument("--smoothing", type=float, default=0.9,
                        help="Smoothing factor for loss curve (0-1)")
    parser.add_argument("--no_refresh", action="store_true",
                        help="Don't refresh, just show current state")
    parser.add_argument("--export", type=str, default=None,
                        help="Export plot to file and exit")
    
    args = parser.parse_args()
    
    if args.export:
        plot_static(args.log_dir, args.export, args.smoothing)
    else:
        plotter = LivePlotter(
            log_dir=args.log_dir,
            refresh_interval=args.refresh_interval,
            smoothing=args.smoothing,
        )
        plotter.run(no_refresh=args.no_refresh)


if __name__ == "__main__":
    main()
