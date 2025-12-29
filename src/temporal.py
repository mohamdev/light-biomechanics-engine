"""
Temporal Module: Memory and attention for video processing

Two implementations:
1. SlidingWindowMemory: Simple baseline for quick testing
2. SpatialPerceiver: EdgeTAM-inspired 2D spatial perceiver (production)

The key insight from EdgeTAM is that the memory attention bottleneck
can be solved by using a perceiver-like architecture that compresses
memory while preserving spatial structure.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SlidingWindowMemory(nn.Module):
    """
    Simple sliding window memory for temporal context.
    
    Good for prototyping and testing. For production, use SpatialPerceiver.
    """
    
    def __init__(
        self,
        feature_dim: int,
        window_size: int = 8,
        use_attention: bool = True,
    ):
        """
        Args:
            feature_dim: Feature dimension from backbone
            window_size: Number of frames to keep in memory
            use_attention: Use attention for aggregation (else avg pool)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.use_attention = use_attention
        
        if use_attention:
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.out_proj = nn.Linear(feature_dim, feature_dim)
            
        # Memory buffer (registered but not a parameter)
        self.register_buffer(
            "memory",
            torch.zeros(1, window_size, feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_mask",
            torch.zeros(1, window_size, dtype=torch.bool),
            persistent=False,
        )
        self.memory_ptr = 0
        
    def reset_memory(self, batch_size: int = 1):
        """Reset memory for new video."""
        device = self.memory.device
        self.memory = torch.zeros(
            batch_size, self.window_size, self.feature_dim, device=device
        )
        self.memory_mask = torch.zeros(
            batch_size, self.window_size, dtype=torch.bool, device=device
        )
        self.memory_ptr = 0
        
    def update_memory(self, features: torch.Tensor):
        """
        Add features to memory.
        
        Args:
            features: [B, D] current frame features
        """
        B = features.shape[0]
        
        # Ensure memory has correct batch size
        if self.memory.shape[0] != B:
            self.reset_memory(B)
            
        # Update at current pointer
        self.memory[:, self.memory_ptr] = features
        self.memory_mask[:, self.memory_ptr] = True
        
        # Advance pointer (circular buffer)
        self.memory_ptr = (self.memory_ptr + 1) % self.window_size
        
    def forward(
        self,
        current_features: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Aggregate current features with memory.
        
        Args:
            current_features: [B, D] current frame features
            update: Whether to update memory after aggregation
            
        Returns:
            [B, D] temporally-aware features
        """
        B, D = current_features.shape
        
        # Ensure memory initialized
        if self.memory.shape[0] != B:
            self.reset_memory(B)
            
        if self.use_attention:
            # Query from current, keys/values from memory
            Q = self.query_proj(current_features).unsqueeze(1)  # [B, 1, D]
            K = self.key_proj(self.memory)  # [B, W, D]
            V = self.value_proj(self.memory)  # [B, W, D]
            
            # Scaled dot-product attention
            scale = D ** -0.5
            attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, 1, W]
            
            # Mask invalid memory slots
            mask = ~self.memory_mask.unsqueeze(1)  # [B, 1, W]
            attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = attn.masked_fill(mask, 0.0)  # Zero out for numerical stability
            
            # Aggregate
            context = torch.matmul(attn, V).squeeze(1)  # [B, D]
            
            # Combine with current (residual connection)
            out = self.out_proj(current_features + context)
        else:
            # Simple average over valid memory
            valid_count = self.memory_mask.sum(dim=1, keepdim=True).clamp(min=1)
            memory_sum = (self.memory * self.memory_mask.unsqueeze(-1)).sum(dim=1)
            context = memory_sum / valid_count
            out = current_features + context
            
        # Update memory
        if update:
            self.update_memory(current_features)
            
        return out


class SpatialPerceiver(nn.Module):
    """
    2D Spatial Perceiver for memory compression (EdgeTAM-inspired).
    
    Key idea: Instead of storing full-resolution memory tokens,
    compress into fixed-size latent tokens while preserving spatial structure.
    This enables constant memory complexity regardless of video length.
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_latents: int = 32,
        n_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Feature dimension from backbone
            n_latents: Number of latent tokens (memory compression)
            n_heads: Number of attention heads
            window_size: Number of frames before memory compression
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_latents = n_latents
        self.n_heads = n_heads
        self.window_size = window_size
        head_dim = feature_dim // n_heads
        
        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(1, n_latents, feature_dim) * 0.02)
        
        # Cross-attention: latents attend to memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention among latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN for latent processing
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Short-term memory buffer
        self.register_buffer(
            "short_memory",
            torch.zeros(1, window_size, feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "short_mask",
            torch.zeros(1, window_size, dtype=torch.bool),
            persistent=False,
        )
        self.short_ptr = 0
        
        # Long-term compressed memory (latents after compression)
        self.register_buffer(
            "long_memory",
            torch.zeros(1, n_latents, feature_dim),
            persistent=False,
        )
        self.has_long_memory = False
        
    def reset_memory(self, batch_size: int = 1):
        """Reset all memory for new video."""
        device = self.short_memory.device
        
        self.short_memory = torch.zeros(
            batch_size, self.window_size, self.feature_dim, device=device
        )
        self.short_mask = torch.zeros(
            batch_size, self.window_size, dtype=torch.bool, device=device
        )
        self.short_ptr = 0
        
        self.long_memory = torch.zeros(
            batch_size, self.n_latents, self.feature_dim, device=device
        )
        self.has_long_memory = False
        
    def _compress_to_latents(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Compress memory tokens into fixed latent representation.
        
        Args:
            memory: [B, T, D] memory tokens
            
        Returns:
            [B, n_latents, D] compressed latents
        """
        B = memory.shape[0]
        
        # Initialize latents for this batch
        latents = repeat(self.latents, "1 n d -> b n d", b=B)
        
        # Cross-attention: latents query memory
        latents = self.norm1(latents)
        latents = latents + self.cross_attn(latents, memory, memory)[0]
        
        # Self-attention among latents
        latents = self.norm2(latents)
        latents = latents + self.self_attn(latents, latents, latents)[0]
        
        # FFN
        latents = self.norm3(latents)
        latents = latents + self.ffn(latents)
        
        return latents
        
    def forward(
        self,
        current_features: torch.Tensor,
        patch_features: Optional[torch.Tensor] = None,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Process current frame with temporal memory.
        
        Args:
            current_features: [B, D] current frame CLS features
            patch_features: [B, N, D] current frame patch features (optional)
            update: Whether to update memory
            
        Returns:
            [B, D] temporally-aware features
        """
        B, D = current_features.shape
        
        # Ensure memory initialized
        if self.short_memory.shape[0] != B:
            self.reset_memory(B)
            
        # Gather all memory for attention
        if self.has_long_memory:
            # Combine long-term compressed memory with short-term
            valid_short = self.short_memory * self.short_mask.unsqueeze(-1)
            all_memory = torch.cat([self.long_memory, valid_short], dim=1)
        else:
            all_memory = self.short_memory * self.short_mask.unsqueeze(-1)
            
        # Current attends to memory
        query = current_features.unsqueeze(1)  # [B, 1, D]
        context, _ = self.cross_attn(query, all_memory, all_memory)
        out = self.out_proj(current_features + context.squeeze(1))
        
        # Update memory
        if update:
            self.short_memory[:, self.short_ptr] = current_features
            self.short_mask[:, self.short_ptr] = True
            self.short_ptr += 1
            
            # Compress when short memory is full
            if self.short_ptr >= self.window_size:
                new_latents = self._compress_to_latents(self.short_memory)
                
                if self.has_long_memory:
                    # Merge with existing long memory
                    combined = torch.cat([self.long_memory, new_latents], dim=1)
                    self.long_memory = self._compress_to_latents(combined)
                else:
                    self.long_memory = new_latents
                    self.has_long_memory = True
                    
                # Reset short memory
                self.short_memory.zero_()
                self.short_mask.zero_()
                self.short_ptr = 0
                
        return out


def create_temporal_module(
    feature_dim: int,
    config: Optional[dict] = None,
) -> nn.Module:
    """
    Factory function to create temporal module.
    
    Args:
        feature_dim: Feature dimension from backbone
        config: Temporal module configuration
        
    Returns:
        Temporal module instance
    """
    if config is None:
        config = {}
        
    if not config.get("enabled", True):
        return nn.Identity()
        
    module_type = config.get("module_type", "sliding_window")
    
    if module_type == "sliding_window":
        return SlidingWindowMemory(
            feature_dim=feature_dim,
            window_size=config.get("window_size", 8),
            use_attention=True,
        )
    elif module_type == "spatial_perceiver":
        return SpatialPerceiver(
            feature_dim=feature_dim,
            n_latents=config.get("perceiver_latents", 32),
            n_heads=config.get("perceiver_heads", 4),
            window_size=config.get("window_size", 8),
        )
    else:
        raise ValueError(f"Unknown temporal module type: {module_type}")
