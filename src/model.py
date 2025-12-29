"""
Model Assembly: Complete Lean Bio-Engine pipeline

Combines backbone, temporal module, and prediction head into
a unified model ready for training or inference.
"""

from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

from .backbone import create_backbone, BackboneWrapper
from .heads import create_head, BioHead
from .temporal import create_temporal_module


class LeanBioEngine(nn.Module):
    """
    Complete biomechanics inference pipeline.
    
    Architecture:
        Input Image → Backbone (DINOv2 + LoRA) → Temporal → Head → Joints/MHR
    
    For single images, temporal module is bypassed.
    For video, temporal module maintains memory across frames.
    """
    
    def __init__(
        self,
        backbone: BackboneWrapper,
        head: nn.Module,
        temporal: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.head = head
        self.temporal = temporal
        self.has_temporal = temporal is not None and not isinstance(temporal, nn.Identity)
        
    def forward(
        self,
        x: torch.Tensor,
        use_temporal: bool = True,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W] for video
            use_temporal: Whether to use temporal module (if available)
            update_memory: Whether to update temporal memory
            
        Returns:
            Dictionary with:
                - joints: [B, N_JOINTS, 3] joint predictions
                - features: [B, D] extracted features
        """
        is_video = x.dim() == 5
        
        if is_video:
            return self._forward_video(x, update_memory)
        else:
            return self._forward_image(x, use_temporal, update_memory)
            
    def _forward_image(
        self,
        x: torch.Tensor,
        use_temporal: bool,
        update_memory: bool,
    ) -> Dict[str, torch.Tensor]:
        """Process single image."""
        # Extract features
        features = self.backbone(x)
        cls_features = features["cls_token"]  # [B, D]
        
        # Apply temporal module if available and requested
        if self.has_temporal and use_temporal:
            cls_features = self.temporal(cls_features, update=update_memory)
            
        # Predict joints
        joints = self.head(cls_features)
        
        return {
            "joints": joints,
            "features": cls_features,
            "patch_features": features["patch_tokens"],
        }
        
    def _forward_video(
        self,
        x: torch.Tensor,
        update_memory: bool,
    ) -> Dict[str, torch.Tensor]:
        """Process video sequence."""
        B, T, C, H, W = x.shape
        
        # Reset temporal memory for new sequence
        if self.has_temporal:
            self.temporal.reset_memory(B)
            
        all_joints = []
        all_features = []
        
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            output = self._forward_image(frame, use_temporal=True, update_memory=update_memory)
            all_joints.append(output["joints"])
            all_features.append(output["features"])
            
        return {
            "joints": torch.stack(all_joints, dim=1),  # [B, T, N_JOINTS, 3]
            "features": torch.stack(all_features, dim=1),  # [B, T, D]
        }
        
    def reset_temporal_memory(self, batch_size: int = 1):
        """Reset temporal memory for new video sequence."""
        if self.has_temporal:
            self.temporal.reset_memory(batch_size)
            
    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameters (useful for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> LeanBioEngine:
    """
    Create complete model from configuration.
    
    Args:
        config: Full configuration dictionary
        device: Target device
        
    Returns:
        LeanBioEngine instance
    """
    if config is None:
        config = {}
        
    # Get sub-configs
    backbone_cfg = config.get("backbone", {})
    lora_cfg = config.get("lora", {})
    head_cfg = config.get("head", {})
    temporal_cfg = config.get("temporal", {})
    
    # Create backbone
    backbone = create_backbone(
        name=backbone_cfg.get("name", "dinov2_vits14"),
        pretrained=backbone_cfg.get("pretrained", True),
        freeze=backbone_cfg.get("freeze", False),
        lora_config=lora_cfg if lora_cfg.get("enabled", True) else None,
        device=device,
    )
    
    feature_dim = backbone.get_feature_dim()
    
    # Create head
    head = create_head(feature_dim, head_cfg)
    head.to(device)
    
    # Create temporal module
    temporal = None
    if temporal_cfg.get("enabled", False):
        temporal = create_temporal_module(feature_dim, temporal_cfg)
        temporal.to(device)
        
    # Assemble model
    model = LeanBioEngine(
        backbone=backbone,
        head=head,
        temporal=temporal,
    )
    
    return model


def load_model(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> LeanBioEngine:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration (if None, loaded from checkpoint)
        device: Target device
        
    Returns:
        LeanBioEngine instance with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config from checkpoint if not provided
    if config is None:
        config = checkpoint.get("config", {})
        
    # Create model
    model = create_model(config, device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model
