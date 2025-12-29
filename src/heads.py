"""
Prediction Heads: Lightweight decoders for biomechanics outputs

Inspired by SAM's decoder architecture - simple but effective.
Outputs can be 3D joint positions or MHR parameters.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn


class BioHead(nn.Module):
    """
    Biomechanics prediction head for joint/MHR estimation.
    
    Architecture inspired by lightweight SAM decoders:
    - Takes CLS token or pooled features
    - MLP with dropout for regularization
    - Outputs joint coordinates or MHR parameters
    """
    
    def __init__(
        self,
        in_dim: int,
        n_joints: int = 24,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        output_type: Literal["joints_3d", "mhr_params"] = "joints_3d",
    ):
        """
        Args:
            in_dim: Input feature dimension (from backbone)
            n_joints: Number of body joints (24 for SMPL)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            output_type: Type of output (joints_3d or mhr_params)
        """
        super().__init__()
        
        self.n_joints = n_joints
        self.output_type = output_type
        
        # Calculate output dimension
        if output_type == "joints_3d":
            out_dim = n_joints * 3  # X, Y, Z per joint
        else:  # mhr_params
            # MHR typically includes: shape (10), pose (72), translation (3)
            out_dim = 10 + 72 + 3
            
        self.out_dim = out_dim
        
        # MLP layers
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # GELU often works better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Initialize output layer with small weights for stable training
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.01)
        nn.init.zeros_(self.layers[-1].bias)
        
    def forward(
        self,
        features: torch.Tensor,
        use_cls: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Either CLS token [B, D] or all tokens [B, N, D]
            use_cls: If features has 3 dims, whether to use CLS (else avg pool)
            
        Returns:
            Predictions [B, out_dim] or reshaped based on output_type
        """
        # Handle different input shapes
        if features.dim() == 3:
            if use_cls:
                x = features[:, 0, :]  # Use CLS token
            else:
                x = features.mean(dim=1)  # Global average pooling
        else:
            x = features  # Already [B, D]
            
        # Forward through MLP
        out = self.layers(x)
        
        # Optionally reshape output
        if self.output_type == "joints_3d":
            out = out.view(-1, self.n_joints, 3)
            
        return out


class MultiScaleBioHead(nn.Module):
    """
    Multi-scale prediction head using features from multiple layers.
    
    Useful when you want to combine low-level and high-level features
    for more accurate predictions.
    """
    
    def __init__(
        self,
        in_dim: int,
        n_joints: int = 24,
        hidden_dim: int = 512,
        n_scales: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_joints = n_joints
        self.n_scales = n_scales
        
        # Feature fusion
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.fusion = nn.Linear(in_dim * n_scales, hidden_dim)
        
        # Prediction head
        self.head = BioHead(
            in_dim=hidden_dim,
            n_joints=n_joints,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of [B, D] features from different layers
            
        Returns:
            Joint predictions [B, n_joints, 3]
        """
        # Weighted combination
        weights = torch.softmax(self.scale_weights, dim=0)
        weighted = [w * f for w, f in zip(weights, multi_scale_features)]
        
        # Concatenate and fuse
        fused = torch.cat(weighted, dim=-1)
        fused = self.fusion(fused)
        
        return self.head(fused)


def create_head(
    in_dim: int,
    config: Optional[dict] = None,
) -> nn.Module:
    """
    Factory function to create prediction head.
    
    Args:
        in_dim: Input feature dimension
        config: Head configuration dict
        
    Returns:
        Head module
    """
    if config is None:
        config = {}
        
    return BioHead(
        in_dim=in_dim,
        n_joints=config.get("n_joints", 24),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.1),
        output_type=config.get("output_type", "joints_3d"),
    )
