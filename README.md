# Light Biomechanics-Engine 🏃‍♂️

A lightweight biomechanics inference pipeline designed for mobile deployment, based on:
- **DINOv2 ViT-Small** backbone for robust geometric features
- **LoRA adaptation** for efficient fine-tuning (Universal Subspace approach)
- **EdgeTAM-inspired** temporal module for video processing

## Quick Start

### 1. Create Environment

```bash
# Using conda (recommended)
conda create -n bioengine python=3.10 -y
conda activate bioengine

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
# Dependencies
pip install peft transformers pyyaml omegaconf numpy tqdm einops pytest black mypy matplotlib opencv-python timm

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


```

### 3. Run Benchmark

```bash
# Quick benchmark (default settings)
python scripts/benchmark.py

# Full benchmark with video simulation
python scripts/benchmark.py --include-temporal --num-frames 16
```

## Project Structure

```
lean-bio-engine/
├── src/
│   ├── __init__.py
│   ├── backbone.py      # DINOv2 ViT-Small backbone
│   ├── heads.py         # Biomechanics prediction heads
│   ├── temporal.py      # EdgeTAM-inspired temporal module
│   ├── model.py         # Full pipeline assembly
│   └── utils.py         # Utilities and helpers
├── configs/
│   └── default.yaml     # Default configuration
├── scripts/
│   ├── benchmark.py     # Performance benchmarking
│   └── demo.py          # Interactive demo
├── tests/
│   └── test_model.py    # Unit tests
├── requirements.txt
└── README.md
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
backbone:
  name: "dinov2_vits14"
  freeze: false  # Set true to freeze backbone

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["qkv"]

head:
  n_joints: 24
  hidden_dim: 512
  dropout: 0.1

temporal:
  enabled: false  # Enable for video
  window_size: 8
  perceiver_latents: 32
```