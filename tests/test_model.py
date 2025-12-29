"""
Unit Tests for Lean Bio-Engine

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn


class TestHeads:
    """Test prediction heads."""
    
    def test_bio_head_shapes(self):
        """Test BioHead output shapes."""
        from src.heads import BioHead
        
        head = BioHead(in_dim=384, n_joints=24)
        
        # Test with CLS token input
        x = torch.randn(2, 384)
        out = head(x)
        assert out.shape == (2, 24, 3)
        
        # Test with all tokens input
        x = torch.randn(2, 197, 384)  # 196 patches + 1 CLS
        out = head(x, use_cls=True)
        assert out.shape == (2, 24, 3)
        
    def test_bio_head_pooling(self):
        """Test BioHead with average pooling."""
        from src.heads import BioHead
        
        head = BioHead(in_dim=384, n_joints=24)
        
        x = torch.randn(2, 197, 384)
        out = head(x, use_cls=False)  # Use avg pooling
        assert out.shape == (2, 24, 3)


class TestTemporal:
    """Test temporal modules."""
    
    def test_sliding_window_shapes(self):
        """Test SlidingWindowMemory shapes."""
        from src.temporal import SlidingWindowMemory
        
        temporal = SlidingWindowMemory(feature_dim=384, window_size=8)
        
        # Process sequence of frames
        for _ in range(10):
            x = torch.randn(2, 384)
            out = temporal(x, update=True)
            assert out.shape == (2, 384)
            
    def test_sliding_window_memory_reset(self):
        """Test memory reset."""
        from src.temporal import SlidingWindowMemory
        
        temporal = SlidingWindowMemory(feature_dim=384, window_size=4)
        
        # Fill memory
        for _ in range(4):
            temporal(torch.randn(1, 384), update=True)
            
        # Check memory is filled
        assert temporal.memory_mask.sum() == 4
        
        # Reset
        temporal.reset_memory(batch_size=1)
        assert temporal.memory_mask.sum() == 0
        
    def test_spatial_perceiver_shapes(self):
        """Test SpatialPerceiver shapes."""
        from src.temporal import SpatialPerceiver
        
        perceiver = SpatialPerceiver(
            feature_dim=384,
            n_latents=32,
            n_heads=4,
            window_size=4,
        )
        
        # Process frames
        for _ in range(10):
            x = torch.randn(2, 384)
            out = perceiver(x, update=True)
            assert out.shape == (2, 384)
            
    def test_spatial_perceiver_compression(self):
        """Test that perceiver compresses memory."""
        from src.temporal import SpatialPerceiver
        
        perceiver = SpatialPerceiver(
            feature_dim=384,
            n_latents=16,
            window_size=4,
        )
        
        # Process enough frames to trigger compression
        for _ in range(8):  # 2x window size
            perceiver(torch.randn(1, 384), update=True)
            
        # Should have long memory now
        assert perceiver.has_long_memory


class TestCreateFunctions:
    """Test factory functions."""
    
    def test_create_head(self):
        """Test head creation."""
        from src.heads import create_head
        
        head = create_head(
            in_dim=384,
            config={"n_joints": 17, "hidden_dim": 256},
        )
        
        assert isinstance(head, nn.Module)
        
        x = torch.randn(1, 384)
        out = head(x)
        assert out.shape == (1, 17, 3)
        
    def test_create_temporal(self):
        """Test temporal module creation."""
        from src.temporal import create_temporal_module
        
        # Sliding window
        module = create_temporal_module(
            feature_dim=384,
            config={"enabled": True, "module_type": "sliding_window"},
        )
        assert not isinstance(module, nn.Identity)
        
        # Disabled
        module = create_temporal_module(
            feature_dim=384,
            config={"enabled": False},
        )
        assert isinstance(module, nn.Identity)


class TestUtils:
    """Test utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        from src.utils import count_parameters
        
        model = nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        counts = count_parameters(model)
        
        assert counts["total"] == 55
        assert counts["trainable"] == 55
        assert counts["frozen"] == 0
        
    def test_count_frozen_parameters(self):
        """Test frozen parameter counting."""
        from src.utils import count_parameters
        
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
            
        counts = count_parameters(model)
        
        assert counts["trainable"] == 0
        assert counts["frozen"] == 55
        
    def test_format_params(self):
        """Test parameter formatting."""
        from src.utils import format_params
        
        assert format_params(100) == "100"
        assert format_params(1500) == "1.50K"
        assert format_params(1500000) == "1.50M"
        assert format_params(1500000000) == "1.50B"


class TestIntegration:
    """Integration tests (require network for model download)."""
    
    @pytest.mark.slow
    def test_backbone_wrapper(self):
        """Test backbone wrapper with real DINOv2."""
        from src.backbone import create_backbone
        
        backbone = create_backbone(
            name="dinov2_vits14",
            pretrained=True,
            lora_config=None,
            device="cpu",
        )
        
        x = torch.randn(1, 3, 224, 224)
        features = backbone(x)
        
        assert "cls_token" in features
        assert "patch_tokens" in features
        assert features["cls_token"].shape == (1, 384)
        
    @pytest.mark.slow
    def test_full_model_cpu(self):
        """Test full model on CPU."""
        from src.model import create_model
        
        config = {
            "backbone": {"name": "dinov2_vits14"},
            "lora": {"enabled": False},  # Skip LoRA for faster test
            "head": {"n_joints": 24},
            "temporal": {"enabled": False},
        }
        
        model = create_model(config, device="cpu")
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output["joints"].shape == (1, 24, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
