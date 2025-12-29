"""
Utilities: Configuration loading, benchmarking, and helpers
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
import yaml
import torch
import torch.nn as nn


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def format_params(n: int) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cuda",
    use_fp16: bool = False,
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        n_warmup: Number of warmup iterations
        n_runs: Number of timed iterations
        device: Device to run on
        use_fp16: Whether to use FP16 inference
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    # Create input
    x = torch.randn(*input_shape, device=device)
    
    if use_fp16:
        model = model.half()
        x = x.half()
        
    # Warmup
    for _ in range(n_warmup):
        _ = model(x)
        
    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - start
    
    # Calculate statistics
    avg_time_ms = (elapsed / n_runs) * 1000
    fps = n_runs / elapsed
    
    return {
        "avg_time_ms": avg_time_ms,
        "fps": fps,
        "total_time_s": elapsed,
        "n_runs": n_runs,
        "input_shape": input_shape,
        "device": device,
        "fp16": use_fp16,
    }


def benchmark_video(
    model: nn.Module,
    n_frames: int = 16,
    image_size: int = 224,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cuda",
    use_fp16: bool = False,
) -> Dict[str, float]:
    """
    Benchmark video inference (frame-by-frame with temporal).
    
    Args:
        model: PyTorch model with temporal module
        n_frames: Number of frames per video
        image_size: Image resolution
        batch_size: Batch size
        n_warmup: Number of warmup videos
        n_runs: Number of timed videos
        device: Device to run on
        use_fp16: Whether to use FP16
        
    Returns:
        Dictionary with video benchmark results
    """
    model = model.to(device)
    model.eval()
    
    # Create video input
    video = torch.randn(batch_size, n_frames, 3, image_size, image_size, device=device)
    
    if use_fp16:
        model = model.half()
        video = video.half()
        
    # Warmup
    for _ in range(n_warmup):
        model.reset_temporal_memory(batch_size)
        _ = model(video)
        
    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        model.reset_temporal_memory(batch_size)
        _ = model(video)
        
    if device == "cuda":
        torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - start
    
    # Statistics
    videos_per_sec = n_runs / elapsed
    frames_per_sec = (n_runs * n_frames) / elapsed
    
    return {
        "videos_per_sec": videos_per_sec,
        "fps": frames_per_sec,
        "avg_video_time_ms": (elapsed / n_runs) * 1000,
        "n_frames": n_frames,
        "n_runs": n_runs,
    }


def get_device(preferred: str = "cuda") -> str:
    """Get best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks running average of a metric."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
