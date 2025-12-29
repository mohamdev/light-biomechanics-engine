#!/usr/bin/env python3
"""
Demo Script: Test model with sample inputs

Usage:
    python scripts/demo.py
    python scripts/demo.py --image path/to/image.jpg
    python scripts/demo.py --video path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src import create_model, load_config, get_device


def demo_random_input(args):
    """Demo with random input tensor."""
    print("=" * 50)
    print(" Lean Bio-Engine Demo (Random Input)")
    print("=" * 50)
    
    device = get_device(args.device)
    print(f"\nDevice: {device}")
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "backbone": {"name": "dinov2_vits14"},
            "lora": {"enabled": True, "rank": 16, "alpha": 32},
            "head": {"n_joints": 24},
            "temporal": {"enabled": False},
        }
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device=device)
    model.eval()
    
    # Create random input
    print("\nRunning inference on random input...")
    x = torch.randn(1, 3, 224, 224, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    joints = output["joints"]
    features = output["features"]
    
    print(f"\nOutput shapes:")
    print(f"  Joints:   {joints.shape}")  # [1, 24, 3]
    print(f"  Features: {features.shape}")  # [1, 384]
    
    print(f"\nJoint predictions (first 5):")
    for i in range(5):
        j = joints[0, i]
        print(f"  Joint {i:2d}: x={j[0]:+.3f}, y={j[1]:+.3f}, z={j[2]:+.3f}")
    
    print("\n✅ Demo complete!")


def demo_image(args):
    """Demo with real image input."""
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        print("Please install: pip install pillow torchvision")
        return
        
    print("=" * 50)
    print(f" Lean Bio-Engine Demo (Image: {args.image})")
    print("=" * 50)
    
    device = get_device(args.device)
    
    # Load and preprocess image
    print("\nLoading image...")
    img = Image.open(args.image).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    x = transform(img).unsqueeze(0).to(device)
    
    # Create model
    print("Creating model...")
    config = {
        "backbone": {"name": "dinov2_vits14"},
        "lora": {"enabled": True, "rank": 16},
        "head": {"n_joints": 24},
        "temporal": {"enabled": False},
    }
    model = create_model(config, device=device)
    model.eval()
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(x)
    
    joints = output["joints"][0].cpu().numpy()
    
    print(f"\nPredicted {len(joints)} joints:")
    for i, j in enumerate(joints):
        print(f"  Joint {i:2d}: ({j[0]:+.3f}, {j[1]:+.3f}, {j[2]:+.3f})")
    
    print("\n✅ Done!")


def demo_video(args):
    """Demo with video input (frame-by-frame)."""
    try:
        import cv2
    except ImportError:
        print("Please install: pip install opencv-python")
        return
        
    from torchvision import transforms
    
    print("=" * 50)
    print(f" Lean Bio-Engine Demo (Video: {args.video})")
    print("=" * 50)
    
    device = get_device(args.device)
    
    # Create model with temporal
    print("\nCreating model with temporal module...")
    config = {
        "backbone": {"name": "dinov2_vits14"},
        "lora": {"enabled": True, "rank": 16},
        "head": {"n_joints": 24},
        "temporal": {"enabled": True, "module_type": "sliding_window"},
    }
    model = create_model(config, device=device)
    model.eval()
    model.reset_temporal_memory(batch_size=1)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Process frames
    frame_count = 0
    max_frames = min(args.max_frames, total_frames)
    
    print(f"\nProcessing {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        x = transform(frame).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(x, use_temporal=True)
        
        joints = output["joints"][0].cpu().numpy()
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count:4d}: Joint 0 = ({joints[0, 0]:+.3f}, {joints[0, 1]:+.3f}, {joints[0, 2]:+.3f})")
            
        frame_count += 1
        
    cap.release()
    
    print(f"\nProcessed {frame_count} frames")
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(description="Demo Lean Bio-Engine")
    
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
        help="Device to run on",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to input image",
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to process from video",
    )
    
    args = parser.parse_args()
    
    if args.image:
        demo_image(args)
    elif args.video:
        demo_video(args)
    else:
        demo_random_input(args)


if __name__ == "__main__":
    main()
