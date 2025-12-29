"""
Lean Bio-Engine: Lightweight Biomechanics Inference Pipeline

A mobile-optimized architecture combining:
- DINOv2 ViT-Small backbone
- LoRA adaptation for efficient fine-tuning
- EdgeTAM-inspired temporal module
"""

from .backbone import create_backbone, BackboneWrapper
from .heads import BioHead, create_head
from .temporal import SlidingWindowMemory, SpatialPerceiver, create_temporal_module
from .model import LeanBioEngine, create_model
from .utils import (
    load_config,
    count_parameters,
    format_params,
    benchmark_model,
    benchmark_video,
    get_device,
)

__version__ = "0.1.0"

__all__ = [
    # Backbone
    "create_backbone",
    "BackboneWrapper",
    # Heads
    "BioHead",
    "create_head",
    # Temporal
    "SlidingWindowMemory",
    "SpatialPerceiver",
    "create_temporal_module",
    # Full model
    "LeanBioEngine",
    "create_model",
    # Utils
    "load_config",
    "count_parameters",
    "format_params",
    "benchmark_model",
    "benchmark_video",
    "get_device",
]
