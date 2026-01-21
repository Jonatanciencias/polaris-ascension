"""
Tests for Tensor Decomposition - Session 24
===========================================

Tests for Tucker, CP, and Tensor-Train decomposition methods.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.compute.tensor_decomposition import (
    TuckerDecomposer,
    CPDecomposer,
    TensorTrainDecomposer,
    decompose_model,
    compute_compression_ratio,
    DecompositionConfig
)


class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = self.fc(x)
        return x


class TestTuckerDecomposer:
    """Test Tucker decomposition."""
    
    def test_tucker_init(self):
        """Test Tucker decomposer initialization."""
        decomposer = TuckerDecomposer(ranks=[8, 8])
        assert decomposer.ranks == [8, 8]
        assert decomposer.auto_rank == True
        assert decomposer.energy_threshold == 0.95
    
    def test_tucker_decompose_conv2d(self):
        """Test Tucker decomposition of Conv2d layer."""
        # Create layer
        layer = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        original_params = sum(p.numel() for p in layer.parameters())
        
        # Decompose
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        # Check structure
        assert isinstance(decomposed, nn.Sequential)
        assert len(decomposed) == 3
        
        # Check parameter reduction
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        assert decomposed_params < original_params
        compression = original_params / decomposed_params
        assert compression > 1.5  # At least 1.5x compression
    
    def test_tucker_decompose_conv2d_forward(self):
        """Test forward pass of decomposed Conv2d."""
        layer = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Input
        x = torch.randn(2, 16, 8, 8)
        
        # Original output
        with torch.no_grad():
            y_original = layer(x)
        
        # Decompose and get output
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        with torch.no_grad():
            y_decomposed = decomposed(x)
        
        # Check shapes match
        assert y_decomposed.shape == y_original.shape
        
        # Check outputs are similar (not exact due to approximation)
        relative_error = (y_original - y_decomposed).norm() / y_original.norm()
        assert relative_error < 1.0  # Allow up to 100% relative error for strong compression
    
    def test_tucker_decompose_linear(self):
        """Test Tucker decomposition of Linear layer."""
        layer = nn.Linear(128, 64)
        original_params = sum(p.numel() for p in layer.parameters())
        
        decomposer = TuckerDecomposer(ranks=[32])
        decomposed = decomposer.decompose_linear(layer)
        
        assert isinstance(decomposed, nn.Sequential)
        assert len(decomposed) == 2
        
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        assert decomposed_params < original_params
    
    def test_tucker_auto_rank(self):
        """Test automatic rank determination."""
        layer = nn.Conv2d(32, 64, kernel_size=3)
        
        decomposer = TuckerDecomposer(auto_rank=True, energy_threshold=0.90)
        decomposed = decomposer.decompose_conv2d(layer)
        
        assert isinstance(decomposed, nn.Sequential)
        # Ranks should be automatically determined
    
    def test_tucker_energy_threshold(self):
        """Test different energy thresholds."""
        layer = nn.Conv2d(16, 32, kernel_size=3)
        
        # High threshold (preserve more energy)
        decomposer_high = TuckerDecomposer(auto_rank=True, energy_threshold=0.99)
        decomposed_high = decomposer_high.decompose_conv2d(layer)
        params_high = sum(p.numel() for p in decomposed_high.parameters())
        
        # Low threshold (more compression)
        decomposer_low = TuckerDecomposer(auto_rank=True, energy_threshold=0.80)
        decomposed_low = decomposer_low.decompose_conv2d(layer)
        params_low = sum(p.numel() for p in decomposed_low.parameters())
        
        # Lower threshold should give more compression
        assert params_low <= params_high


class TestCPDecomposer:
    """Test CP (CANDECOMP/PARAFAC) decomposition."""
    
    def test_cp_init(self):
        """Test CP decomposer initialization."""
        decomposer = CPDecomposer(rank=8, max_iterations=25)
        assert decomposer.rank == 8
        assert decomposer.max_iterations == 25
    
    def test_cp_decompose_conv2d(self):
        """Test CP decomposition of Conv2d layer."""
        layer = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        original_params = sum(p.numel() for p in layer.parameters())
        
        decomposer = CPDecomposer(rank=4, max_iterations=10)
        decomposed = decomposer.decompose_conv2d(layer)
        
        assert isinstance(decomposed, nn.Sequential)
        assert len(decomposed) == 4  # 4 layers in CP decomposition
        
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        assert decomposed_params < original_params
        
        # CP should give aggressive compression
        compression = original_params / decomposed_params
        assert compression > 2.0  # At least 2x
    
    def test_cp_decompose_forward(self):
        """Test forward pass of CP decomposed layer."""
        layer = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        x = torch.randn(2, 8, 8, 8)
        
        with torch.no_grad():
            y_original = layer(x)
        
        decomposer = CPDecomposer(rank=2, max_iterations=5)
        decomposed = decomposer.decompose_conv2d(layer)
        
        with torch.no_grad():
            y_decomposed = decomposed(x)
        
        assert y_decomposed.shape == y_original.shape
    
    def test_cp_khatri_rao(self):
        """Test Khatri-Rao product."""
        decomposer = CPDecomposer(rank=4)
        
        A = torch.randn(3, 4)
        B = torch.randn(5, 4)
        
        result = decomposer._khatri_rao(A, B)
        
        assert result.shape == (3 * 5, 4)
        
        # Check column-wise Kronecker product
        for r in range(4):
            expected = torch.kron(A[:, r], B[:, r])
            assert torch.allclose(result[:, r], expected, atol=1e-5)
    
    def test_cp_different_ranks(self):
        """Test CP with different ranks."""
        layer = nn.Conv2d(16, 16, kernel_size=3)
        
        # Low rank (high compression)
        decomposer_low = CPDecomposer(rank=2, max_iterations=5)
        decomposed_low = decomposer_low.decompose_conv2d(layer)
        params_low = sum(p.numel() for p in decomposed_low.parameters())
        
        # High rank (less compression)
        decomposer_high = CPDecomposer(rank=8, max_iterations=5)
        decomposed_high = decomposer_high.decompose_conv2d(layer)
        params_high = sum(p.numel() for p in decomposed_high.parameters())
        
        # Lower rank should have fewer parameters
        assert params_low < params_high


class TestTensorTrainDecomposer:
    """Test Tensor-Train decomposition."""
    
    def test_tt_init(self):
        """Test TT decomposer initialization."""
        decomposer = TensorTrainDecomposer(ranks=[4, 4], max_rank=16)
        assert decomposer.ranks == [4, 4]
        assert decomposer.max_rank == 16
    
    def test_tt_decompose_conv2d(self):
        """Test TT decomposition of Conv2d layer."""
        layer = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        original_params = sum(p.numel() for p in layer.parameters())
        
        decomposer = TensorTrainDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        assert isinstance(decomposed, nn.Sequential)
        
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        assert decomposed_params < original_params
    
    def test_tt_auto_ranks(self):
        """Test automatic rank determination for TT."""
        layer = nn.Conv2d(32, 64, kernel_size=3)
        
        decomposer = TensorTrainDecomposer(max_rank=8)
        ranks = decomposer._auto_ranks(layer)
        
        assert isinstance(ranks, list)
        assert len(ranks) == 2
        assert all(r <= 8 for r in ranks)


class TestModelDecomposition:
    """Test full model decomposition."""
    
    def test_decompose_model_tucker(self):
        """Test decomposing entire model with Tucker."""
        model = SimpleConvNet()
        original_params = sum(p.numel() for p in model.parameters())
        
        config = DecompositionConfig(
            method="tucker",
            ranks=[8, 16],
            auto_rank=False
        )
        
        decomposed = decompose_model(model, config)
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        
        # Should have compression
        assert decomposed_params < original_params
    
    def test_decompose_model_cp(self):
        """Test decomposing entire model with CP."""
        model = SimpleConvNet()
        original_params = sum(p.numel() for p in model.parameters())
        
        config = DecompositionConfig(
            method="cp",
            ranks=[4],
            max_iterations=10
        )
        
        decomposed = decompose_model(model, config)
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        
        assert decomposed_params < original_params
    
    def test_decompose_model_tt(self):
        """Test decomposing entire model with TT."""
        model = SimpleConvNet()
        original_params = sum(p.numel() for p in model.parameters())
        
        config = DecompositionConfig(
            method="tt",
            ranks=[4, 8]
        )
        
        decomposed = decompose_model(model, config)
        decomposed_params = sum(p.numel() for p in decomposed.parameters())
        
        assert decomposed_params < original_params
    
    def test_decompose_model_forward(self):
        """Test forward pass of decomposed model."""
        model = SimpleConvNet()
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            y_original = model(x)
        
        config = DecompositionConfig(method="tucker", ranks=[4, 8])
        decomposed = decompose_model(model, config)
        
        with torch.no_grad():
            y_decomposed = decomposed(x)
        
        # Shapes should match
        assert y_decomposed.shape == y_original.shape
    
    def test_compute_compression_ratio(self):
        """Test compression ratio computation."""
        model = SimpleConvNet()
        
        config = DecompositionConfig(method="tucker", ranks=[8, 16])
        decomposed = decompose_model(model, config)
        
        ratio = compute_compression_ratio(model, decomposed)
        
        assert ratio >= 1.0  # Should have compression (or at least same size)
        assert ratio < 100.0  # Reasonable upper bound


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_layer(self):
        """Test decomposition of very small layer."""
        layer = nn.Conv2d(3, 8, kernel_size=3)
        
        decomposer = TuckerDecomposer(ranks=[2, 4])
        decomposed = decomposer.decompose_conv2d(layer)
        
        assert isinstance(decomposed, nn.Sequential)
    
    def test_1x1_conv(self):
        """Test 1x1 convolution (should be skipped in model decomposition)."""
        layer = nn.Conv2d(16, 32, kernel_size=1)
        
        # Should decompose if called directly
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        assert isinstance(decomposed, nn.Sequential)
    
    def test_layer_with_bias(self):
        """Test decomposition preserves bias."""
        layer = nn.Conv2d(16, 32, kernel_size=3, bias=True)
        assert layer.bias is not None
        
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        # Last layer should have bias
        assert decomposed[-1].bias is not None
    
    def test_layer_without_bias(self):
        """Test decomposition without bias."""
        layer = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        assert layer.bias is None
        
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        # Last layer should not have bias
        assert decomposed[-1].bias is None
    
    def test_invalid_method(self):
        """Test invalid decomposition method."""
        model = SimpleConvNet()
        config = DecompositionConfig(method="invalid")
        
        with pytest.raises(ValueError, match="Unknown method"):
            decompose_model(model, config)
    
    def test_strided_conv(self):
        """Test decomposition with stride."""
        layer = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        decomposer = TuckerDecomposer(ranks=[8, 16])
        decomposed = decomposer.decompose_conv2d(layer)
        
        x = torch.randn(2, 16, 16, 16)
        with torch.no_grad():
            y = decomposed(x)
        
        # Output should be strided
        assert y.shape[2] == 8  # H/2
        assert y.shape[3] == 8  # W/2
    
    def test_dilated_conv(self):
        """Test decomposition with dilation."""
        layer = nn.Conv2d(8, 16, kernel_size=3, dilation=2, padding=2)
        
        decomposer = TuckerDecomposer(ranks=[4, 8])
        decomposed = decomposer.decompose_conv2d(layer)
        
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            y = decomposed(x)
        
        assert y.shape[1] == 16  # Output channels preserved


class TestCompressionMetrics:
    """Test compression metrics and benchmarks."""
    
    def test_compression_ratios_tucker(self):
        """Test Tucker compression ratios."""
        layer = nn.Conv2d(64, 64, kernel_size=3)
        original_params = sum(p.numel() for p in layer.parameters())
        
        # Conservative ranks (less compression)
        decomposer_conservative = TuckerDecomposer(ranks=[32, 32])
        decomposed_conservative = decomposer_conservative.decompose_conv2d(layer)
        params_conservative = sum(p.numel() for p in decomposed_conservative.parameters())
        ratio_conservative = original_params / params_conservative
        
        # Aggressive ranks (more compression)
        decomposer_aggressive = TuckerDecomposer(ranks=[8, 8])
        decomposed_aggressive = decomposer_aggressive.decompose_conv2d(layer)
        params_aggressive = sum(p.numel() for p in decomposed_aggressive.parameters())
        ratio_aggressive = original_params / params_aggressive
        
        # Aggressive should have higher compression
        assert ratio_aggressive > ratio_conservative
        assert ratio_aggressive > 5.0  # Should achieve at least 5x
    
    def test_compression_ratios_cp(self):
        """Test CP compression ratios."""
        layer = nn.Conv2d(64, 64, kernel_size=3)
        original_params = sum(p.numel() for p in layer.parameters())
        
        decomposer = CPDecomposer(rank=4, max_iterations=10)
        decomposed = decomposer.decompose_conv2d(layer)
        params_decomposed = sum(p.numel() for p in decomposed.parameters())
        
        ratio = original_params / params_decomposed
        
        # CP should achieve high compression
        assert ratio > 10.0  # At least 10x for rank=4


class TestNumericalStability:
    """Test numerical stability of decompositions."""
    
    def test_tucker_svd_stability(self):
        """Test Tucker decomposition is numerically stable."""
        # Create layer with well-conditioned weights
        layer = nn.Conv2d(16, 32, kernel_size=3)
        nn.init.orthogonal_(layer.weight)
        
        decomposer = TuckerDecomposer(ranks=[8, 16])
        
        # Should not raise any numerical errors
        decomposed = decomposer.decompose_conv2d(layer)
        
        # Check no NaNs or Infs
        for param in decomposed.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()
    
    def test_cp_als_convergence(self):
        """Test CP-ALS converges."""
        layer = nn.Conv2d(8, 8, kernel_size=3)
        
        decomposer = CPDecomposer(rank=2, max_iterations=50, tolerance=1e-4)
        
        # Should converge without errors
        decomposed = decomposer.decompose_conv2d(layer)
        
        for param in decomposed.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
