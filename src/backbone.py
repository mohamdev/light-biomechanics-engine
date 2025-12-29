"""
Backbone Module: DINOv2 ViT-Small with optional LoRA adaptation

The backbone provides robust geometric features suitable for biomechanics tasks.
Based on the "Efficient Track Anything" finding that vanilla ViT is sufficient
for mobile deployment while maintaining strong performance.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class BackboneWrapper(nn.Module):
    """
    Wrapper around DINOv2 backbone that handles:
    - Feature extraction (CLS token + patch tokens)
    - Optional LoRA adaptation
    - Consistent output format
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        with_registers: bool = False,
    ):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.with_registers = with_registers
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning structured features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary with:
                - cls_token: [B, D] CLS token features
                - patch_tokens: [B, N, D] Patch token features
                - all_tokens: [B, N+1, D] All tokens concatenated
        """
        # DINOv2 forward returns different formats depending on call method
        # Using forward_features for intermediate features
        features = self.model.forward_features(x)
        
        # Handle different output formats
        if isinstance(features, dict):
            # Some versions return a dict
            tokens = features.get("x", features.get("x_prenorm", None))
            if tokens is None:
                tokens = list(features.values())[0]
        else:
            tokens = features
            
        # Split CLS and patch tokens
        cls_token = tokens[:, 0, :]  # [B, D]
        
        # Handle register tokens if present (DINOv2 with registers)
        if self.with_registers:
            # Registers are typically at positions 1:5
            patch_tokens = tokens[:, 5:, :]  # Skip CLS + 4 registers
        else:
            patch_tokens = tokens[:, 1:, :]  # Skip only CLS
            
        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "all_tokens": tokens,
        }
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.feature_dim


def create_backbone(
    name: str = "dinov2_vits14",
    pretrained: bool = True,
    freeze: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> BackboneWrapper:
    """
    Create a DINOv2 backbone with optional LoRA adaptation.
    
    Args:
        name: Model name (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
        pretrained: Load pretrained weights
        freeze: Freeze backbone weights (ignored if LoRA enabled)
        lora_config: LoRA configuration dict (None to disable)
        device: Target device
        
    Returns:
        BackboneWrapper instance
    """
    # Model dimension mapping
    DIM_MAP = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        # With registers
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
    }
    
    with_registers = "reg" in name
    feature_dim = DIM_MAP.get(name, 384)
    
    # Load model from torch hub
    print(f"Loading backbone: {name}")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        name,
        pretrained=pretrained,
    )
    
    # Apply LoRA if configured
    if lora_config is not None and lora_config.get("enabled", True):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft package required for LoRA. Install with: pip install peft"
            )
        
        print("Applying LoRA adaptation...")
        peft_config = LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", ["qkv"]),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    elif freeze:
        print("Freezing backbone weights")
        for param in model.parameters():
            param.requires_grad = False
            
    # Wrap and move to device
    wrapper = BackboneWrapper(
        model=model,
        feature_dim=feature_dim,
        with_registers=with_registers,
    )
    wrapper.to(device)
    
    return wrapper


def get_available_backbones() -> List[str]:
    """Return list of available backbone names."""
    return [
        "dinov2_vits14",    # ViT-Small (recommended for mobile)
        "dinov2_vitb14",    # ViT-Base
        "dinov2_vitl14",    # ViT-Large
        "dinov2_vitg14",    # ViT-Giant
        "dinov2_vits14_reg",  # With registers (better for dense tasks)
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
    ]
