#!/usr/bin/env python3
"""
Benchmark Script: Test model performance on your hardware

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --config configs/default.yaml
    python scripts/benchmark.py --include-temporal --num-frames 16
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src import (
    create_model,
    load_config,
    count_parameters,
    format_params,
    benchmark_model,
    benchmark_video,
    get_device,
)


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    width = 60
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def run_benchmark(args):
    """Run full benchmark suite."""
    
    print_header("Lean Bio-Engine Benchmark", "█")
    
    # Load config
    if args.config:
        print(f"Loading config: {args.config}")
        config = load_config(args.config)
    else:
        # Default config
        config = {
            "backbone": {"name": "dinov2_vits14", "pretrained": True},
            "lora": {
                "enabled": True,
                "rank": 16,
                "alpha": 32,
                "target_modules": ["qkv"],
                "dropout": 0.05,
            },
            "head": {"n_joints": 24, "hidden_dim": 512},
            "temporal": {
                "enabled": args.include_temporal,
                "module_type": args.temporal_type,
                "window_size": 8,
                "perceiver_latents": 32,
            },
        }
    
    # Override temporal if specified
    if args.include_temporal:
        config["temporal"]["enabled"] = True
        
    # Get device
    device = get_device(args.device)
    print(f"Device: {device}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Create model
    print_header("Creating Model")
    
    print("Building model...")
    model = create_model(config, device=device)
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total:     {format_params(params['total'])}")
    print(f"  Trainable: {format_params(params['trainable'])} ({params['trainable_pct']:.1f}%)")
    print(f"  Frozen:    {format_params(params['frozen'])}")
    
    # Memory usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(torch.randn(1, 3, 224, 224, device=device))
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {peak_mem:.2f} GB")
    
    # Benchmark single image
    print_header("Image Inference Benchmark")
    
    for batch_size in [1, 4, 8]:
        for use_fp16 in [False, True]:
            if device == "cpu" and use_fp16:
                continue  # Skip FP16 on CPU
                
            dtype_str = "FP16" if use_fp16 else "FP32"
            
            try:
                results = benchmark_model(
                    model,
                    input_shape=(batch_size, 3, 224, 224),
                    n_warmup=args.warmup,
                    n_runs=args.runs,
                    device=device,
                    use_fp16=use_fp16,
                )
                
                print(f"Batch={batch_size:2d}, {dtype_str}: "
                      f"{results['fps']:7.1f} FPS, "
                      f"{results['avg_time_ms']:6.2f} ms/batch")
                      
            except Exception as e:
                print(f"Batch={batch_size:2d}, {dtype_str}: Failed ({e})")
                
    # Benchmark video if temporal enabled
    if config["temporal"]["enabled"]:
        print_header("Video Inference Benchmark")
        
        for n_frames in [8, 16, 32]:
            if n_frames > args.num_frames:
                continue
                
            try:
                results = benchmark_video(
                    model,
                    n_frames=n_frames,
                    batch_size=1,
                    n_warmup=3,
                    n_runs=10,
                    device=device,
                )
                
                print(f"Frames={n_frames:2d}: "
                      f"{results['fps']:6.1f} FPS, "
                      f"{results['avg_video_time_ms']:6.1f} ms/video")
                      
            except Exception as e:
                print(f"Frames={n_frames:2d}: Failed ({e})")
    
    # Mobile estimate
    print_header("Mobile Deployment Estimate")

    single_results = benchmark_model(
        model,
        input_shape=(1, 3, 224, 224),
        device=device,
        use_fp16=True if device == "cuda" else False,
    )

    # Rough estimate based on typical GPU -> Mobile ratios
    # RTX 4070 ~= 10x iPhone 15 NPU for this workload
    if device == "cuda":
        mobile_factor = 10.0 if "4070" in torch.cuda.get_device_name(0) else 8.0
    else:
        mobile_factor = 8.0
    estimated_mobile_fps = single_results['fps'] / mobile_factor
    
    print(f"Estimated mobile FPS (after CoreML/int8): {estimated_mobile_fps:.1f}")
    
    if estimated_mobile_fps >= 30:
        print("✅ Target 30 FPS: ACHIEVABLE")
    elif estimated_mobile_fps >= 15:
        print("⚠️  Target 30 FPS: May need optimization")
    else:
        print("❌ Target 30 FPS: Need lighter model (ViT-Tiny)")
        
    # Summary
    print_header("Summary")
    
    print(f"Model: DINOv2 ViT-Small + LoRA (r={config['lora']['rank']})")
    print(f"Trainable params: {format_params(params['trainable'])}")
    print(f"Best FPS (GPU): {single_results['fps']:.1f}")
    print(f"Estimated mobile FPS: {estimated_mobile_fps:.1f}")
    
    if config["temporal"]["enabled"]:
        print(f"Temporal: {config['temporal']['module_type']}")
    else:
        print("Temporal: disabled")
        
    print("\n✅ Benchmark complete!")
    

def main():
    parser = argparse.ArgumentParser(description="Benchmark Lean Bio-Engine")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to benchmark on",
    )
    parser.add_argument(
        "--include-temporal", "-t",
        action="store_true",
        help="Include temporal module in benchmark",
    )
    parser.add_argument(
        "--temporal-type",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "spatial_perceiver"],
        help="Type of temporal module",
    )
    parser.add_argument(
        "--num-frames", "-f",
        type=int,
        default=16,
        help="Number of frames for video benchmark",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
