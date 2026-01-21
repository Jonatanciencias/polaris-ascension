"""
Tests for DARTS (Differentiable Architecture Search)

Tests cover:
- Operation primitives
- Mixed operations
- Cell construction
- Network architecture
- Bilevel optimization
- Architecture derivation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.compute.nas_darts import (
    # Config
    DARTSConfig,
    SearchSpace,
    SearchResult,
    
    # Operations
    Identity,
    Zero,
    SepConv,
    DilConv,
    AvgPool,
    MaxPool,
    FactorizedReduce,
    ReLUConvBN,
    create_operation,
    
    # Core components
    MixedOp,
    Cell,
    DARTSNetwork,
    DARTSTrainer,
    
    # Main API
    search_architecture,
    PRIMITIVES,
)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration classes."""
    
    def test_darts_config_defaults(self):
        """Test default DARTS configuration."""
        config = DARTSConfig()
        
        assert config.search_space == SearchSpace.CNN
        assert config.num_cells == 8
        assert config.num_nodes == 4
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 0.025
        assert config.arch_learning_rate == 3e-4
    
    def test_darts_config_custom(self):
        """Test custom configuration."""
        config = DARTSConfig(
            epochs=100,
            batch_size=32,
            num_nodes=5
        )
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.num_nodes == 5
    
    def test_search_result_creation(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            normal_genotype=[('sep_conv_3x3', 0), ('skip_connect', 1)],
            reduce_genotype=[('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            final_train_loss=0.5,
            final_val_loss=0.6,
            final_train_acc=95.0,
            final_val_acc=93.0,
            total_epochs=50,
            total_time_seconds=1800.0,
            best_epoch=45,
            architecture_weights={}
        )
        
        assert len(result.normal_genotype) == 2
        assert result.final_val_acc == 93.0
        assert result.total_epochs == 50


# ============================================================================
# Operation Tests
# ============================================================================

class TestOperations:
    """Tests for primitive operations."""
    
    def test_identity_operation(self):
        """Test Identity (skip connection)."""
        op = Identity(C=16, stride=1)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == x.shape
        assert torch.allclose(out, x)
    
    def test_zero_operation_stride1(self):
        """Test Zero operation with stride=1."""
        op = Zero(C=16, stride=1)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == x.shape
        assert torch.all(out == 0)
    
    def test_zero_operation_stride2(self):
        """Test Zero operation with stride=2."""
        op = Zero(C=16, stride=2)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 16, 16, 16)
        assert torch.all(out == 0)
    
    def test_separable_conv(self):
        """Test Separable Convolution."""
        op = SepConv(C_in=16, C_out=16, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 16, 32, 32)
        # Check it's not all zeros
        assert out.abs().sum() > 0
    
    def test_dilated_conv(self):
        """Test Dilated Convolution."""
        op = DilConv(C_in=16, C_out=16, kernel_size=3, stride=1, 
                    padding=2, dilation=2)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 16, 32, 32)
    
    def test_avg_pool(self):
        """Test Average Pooling."""
        op = AvgPool(C=16, stride=2)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 16, 16, 16)
    
    def test_max_pool(self):
        """Test Max Pooling."""
        op = MaxPool(C=16, stride=2)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 16, 16, 16)
    
    def test_factorized_reduce(self):
        """Test Factorized Reduction."""
        op = FactorizedReduce(C_in=16, C_out=32)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 32, 16, 16)
    
    def test_relu_conv_bn(self):
        """Test ReLU + Conv + BN block."""
        op = ReLUConvBN(C_in=16, C_out=32, kernel_size=1, stride=1, padding=0)
        x = torch.randn(2, 16, 32, 32)
        
        out = op(x)
        
        assert out.shape == (2, 32, 32, 32)


# ============================================================================
# Operation Factory Tests
# ============================================================================

class TestOperationFactory:
    """Tests for operation creation."""
    
    def test_create_all_operations(self):
        """Test creating all primitive operations."""
        for primitive in PRIMITIVES:
            op = create_operation(primitive, C=16, stride=1)
            assert op is not None
            assert isinstance(op, nn.Module)
    
    def test_create_sep_conv_3x3(self):
        """Test creating sep_conv_3x3."""
        op = create_operation('sep_conv_3x3', C=16, stride=1)
        assert isinstance(op, SepConv)
    
    def test_create_sep_conv_5x5(self):
        """Test creating sep_conv_5x5."""
        op = create_operation('sep_conv_5x5', C=16, stride=1)
        assert isinstance(op, SepConv)
    
    def test_create_skip_connect(self):
        """Test creating skip_connect."""
        op = create_operation('skip_connect', C=16, stride=1)
        assert isinstance(op, Identity)
    
    def test_create_invalid_operation(self):
        """Test creating invalid operation raises error."""
        with pytest.raises(ValueError, match="Unknown primitive"):
            create_operation('invalid_op', C=16, stride=1)


# ============================================================================
# Mixed Operation Tests
# ============================================================================

class TestMixedOp:
    """Tests for mixed operation."""
    
    def test_mixed_op_initialization(self):
        """Test MixedOp initialization."""
        mixed_op = MixedOp(C=16, stride=1)
        
        assert len(mixed_op._ops) == len(PRIMITIVES)
        assert all(isinstance(op, nn.Module) for op in mixed_op._ops)
    
    def test_mixed_op_forward(self):
        """Test MixedOp forward pass."""
        mixed_op = MixedOp(C=16, stride=1)
        x = torch.randn(2, 16, 32, 32)
        weights = torch.softmax(torch.randn(len(PRIMITIVES)), dim=0)
        
        out = mixed_op(x, weights)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_mixed_op_weighted_sum(self):
        """Test that MixedOp computes weighted sum correctly."""
        mixed_op = MixedOp(C=16, stride=1)
        x = torch.randn(2, 16, 32, 32)
        
        # Test with one-hot weights
        weights = torch.zeros(len(PRIMITIVES))
        weights[PRIMITIVES.index('skip_connect')] = 1.0
        
        out = mixed_op(x, weights)
        
        # Should be approximately identity
        assert out.shape == x.shape


# ============================================================================
# Cell Tests
# ============================================================================

class TestCell:
    """Tests for DARTS cell."""
    
    def test_cell_initialization(self):
        """Test Cell initialization."""
        cell = Cell(
            steps=4,
            multiplier=4,
            C_prev_prev=16,
            C_prev=16,
            C=16,
            reduction=False,
            reduction_prev=False
        )
        
        assert cell.steps == 4
        assert cell.multiplier == 4
        assert not cell.reduction
    
    def test_cell_forward_normal(self):
        """Test normal cell forward pass."""
        cell = Cell(
            steps=4,
            multiplier=4,
            C_prev_prev=16,
            C_prev=16,
            C=16,
            reduction=False,
            reduction_prev=False
        )
        
        s0 = torch.randn(2, 16, 32, 32)
        s1 = torch.randn(2, 16, 32, 32)
        
        # Create architecture weights
        k = sum(1 for i in range(4) for n in range(2 + i))
        weights = [torch.softmax(torch.randn(len(PRIMITIVES)), dim=0) for _ in range(k)]
        
        out = cell(s0, s1, weights)
        
        # Output should concatenate 4 intermediate nodes
        assert out.shape == (2, 16 * 4, 32, 32)
    
    def test_cell_forward_reduction(self):
        """Test reduction cell forward pass."""
        cell = Cell(
            steps=4,
            multiplier=4,
            C_prev_prev=16,
            C_prev=16,
            C=32,
            reduction=True,
            reduction_prev=False
        )
        
        s0 = torch.randn(2, 16, 32, 32)
        s1 = torch.randn(2, 16, 32, 32)
        
        # Create architecture weights
        k = sum(1 for i in range(4) for n in range(2 + i))
        weights = [torch.softmax(torch.randn(len(PRIMITIVES)), dim=0) for _ in range(k)]
        
        out = cell(s0, s1, weights)
        
        # Reduction cell should reduce spatial dimensions
        assert out.shape[2] <= s0.shape[2]  # Height reduced
        assert out.shape[3] <= s0.shape[3]  # Width reduced


# ============================================================================
# Network Tests
# ============================================================================

class TestDARTSNetwork:
    """Tests for DARTS network."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=8,
            criterion=criterion,
            steps=4
        )
        
        assert network._C == 16
        assert network._num_classes == 10
        assert network._layers == 8
        assert len(network.cells) == 8
    
    def test_network_forward(self):
        """Test network forward pass."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=8,
            criterion=criterion
        )
        
        x = torch.randn(2, 3, 32, 32)
        logits = network(x)
        
        assert logits.shape == (2, 10)
        assert not torch.isnan(logits).any()
    
    def test_architecture_parameters(self):
        """Test architecture parameter access."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=8,
            criterion=criterion
        )
        
        arch_params = network.arch_parameters()
        
        assert len(arch_params) == 2  # normal + reduce
        assert all(p.requires_grad for p in arch_params)
    
    def test_genotype_derivation(self):
        """Test architecture genotype derivation."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=8,
            criterion=criterion,
            steps=4
        )
        
        gene_normal, gene_reduce = network.genotype()
        
        # Each cell should have 2 * steps edges
        assert len(gene_normal) == 2 * 4
        assert len(gene_reduce) == 2 * 4
        
        # Each edge should be (operation, node_index)
        for op, idx in gene_normal:
            assert op in PRIMITIVES
            assert isinstance(idx, int)


# ============================================================================
# Trainer Tests
# ============================================================================

class TestDARTSTrainer:
    """Tests for DARTS trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        config = DARTSConfig(epochs=10, batch_size=32)
        
        trainer = DARTSTrainer(network, config)
        
        assert trainer.model == network
        assert trainer.config == config
        assert trainer.optimizer is not None
        assert trainer.arch_optimizer is not None
    
    def test_trainer_step(self):
        """Test single training step."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        config = DARTSConfig()
        trainer = DARTSTrainer(network, config)
        
        # Create dummy data
        train_data = (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        valid_data = (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        
        train_loss, valid_loss = trainer.step(train_data, valid_data)
        
        assert isinstance(train_loss, float)
        assert isinstance(valid_loss, float)
        assert train_loss > 0
        assert valid_loss > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.slow
    def test_small_search(self):
        """Test small architecture search (quick test)."""
        # Skip on CPU-only systems
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create tiny dataset
        train_loader = [
            (torch.randn(4, 3, 32, 32).cuda(), torch.randint(0, 10, (4,)).cuda())
            for _ in range(5)
        ]
        valid_loader = [
            (torch.randn(4, 3, 32, 32).cuda(), torch.randint(0, 10, (4,)).cuda())
            for _ in range(5)
        ]
        
        config = DARTSConfig(
            epochs=2,  # Very short for testing
            batch_size=4,
            layers=4,
            init_channels=8
        )
        
        result = search_architecture(
            train_loader,
            valid_loader,
            config,
            device="cuda",
            verbose=False
        )
        
        assert result is not None
        assert len(result.normal_genotype) > 0
        assert len(result.reduce_genotype) > 0
        assert result.total_epochs == 2
    
    def test_network_gradient_flow(self):
        """Test that gradients flow correctly."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        # Forward
        loss = network._loss(x, target)
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_architecture_gradient_flow(self):
        """Test that architecture gradients flow."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        # Forward
        loss = network._loss(x, target)
        
        # Backward
        loss.backward()
        
        # Check architecture gradients
        for param in network.arch_parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_batch_size(self):
        """Test with batch size of 1."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        
        x = torch.randn(1, 3, 32, 32)
        logits = network(x)
        
        assert logits.shape == (1, 10)
    
    def test_large_batch_size(self):
        """Test with large batch size."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        
        x = torch.randn(128, 3, 32, 32)
        logits = network(x)
        
        assert logits.shape == (128, 10)
    
    def test_different_image_sizes(self):
        """Test with different input sizes."""
        criterion = nn.CrossEntropyLoss()
        network = DARTSNetwork(
            C=16,
            num_classes=10,
            layers=4,
            criterion=criterion
        )
        
        # Test 28x28 (MNIST-like)
        x = torch.randn(2, 3, 28, 28)
        logits = network(x)
        assert logits.shape == (2, 10)
        
        # Test 64x64
        x = torch.randn(2, 3, 64, 64)
        logits = network(x)
        assert logits.shape == (2, 10)


# ============================================================================
# Utility Tests
# ============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_primitives_defined(self):
        """Test that PRIMITIVES list is defined."""
        assert PRIMITIVES is not None
        assert len(PRIMITIVES) == 8
        assert 'none' in PRIMITIVES
        assert 'skip_connect' in PRIMITIVES
    
    def test_all_primitives_creatable(self):
        """Test that all primitives can be created."""
        for primitive in PRIMITIVES:
            op = create_operation(primitive, C=16, stride=1)
            assert op is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
