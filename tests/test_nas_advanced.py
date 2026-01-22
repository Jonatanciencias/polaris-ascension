"""
Tests for Advanced NAS Features (Session 28)

Comprehensive test suite for:
1. Progressive architecture refinement
2. Multi-branch search spaces
3. Automated mixed precision

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from src.compute.nas_advanced import (
    ProgressiveNAS,
    MixedPrecisionNAS,
    MultiBranchOperation,
    PrecisionLevel,
    SearchStage,
    ProgressiveConfig,
    MultiBranchConfig,
    MixedPrecisionConfig,
    create_progressive_nas,
    create_mixed_precision_nas
)
from src.compute.nas_darts import DARTSConfig


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def simple_dataloader():
    """Create simple test dataloader"""
    class SimpleDataset:
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
            self.batch_size = 16
            
        def __len__(self):
            return self.num_samples // self.batch_size
        
        def __iter__(self):
            for _ in range(len(self)):
                inputs = torch.randn(self.batch_size, 3, 32, 32)
                targets = torch.randint(0, 10, (self.batch_size,))
                yield inputs, targets
    
    dataset = SimpleDataset()
    dataset.batch_size = 16
    return dataset


@pytest.fixture
def simple_model():
    """Create simple test model"""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return TestModel()


# ============================================================================
# Test Configurations
# ============================================================================

class TestConfigurations:
    """Test configuration classes"""
    
    def test_progressive_config_creation(self):
        """Test ProgressiveConfig creation"""
        config = ProgressiveConfig()
        
        assert config.coarse_epochs == 10
        assert config.medium_epochs == 20
        assert config.fine_epochs == 30
        assert config.coarse_keep_ratio == 0.5
        assert config.medium_keep_ratio == 0.3
        assert config.prune_operations is True
    
    def test_progressive_config_custom(self):
        """Test custom ProgressiveConfig"""
        config = ProgressiveConfig(
            coarse_epochs=5,
            medium_epochs=15,
            fine_epochs=25,
            coarse_keep_ratio=0.6,
            patience=10
        )
        
        assert config.coarse_epochs == 5
        assert config.medium_epochs == 15
        assert config.fine_epochs == 25
        assert config.coarse_keep_ratio == 0.6
        assert config.patience == 10
    
    def test_multi_branch_config(self):
        """Test MultiBranchConfig"""
        config = MultiBranchConfig()
        
        assert config.max_branches == 3
        assert 'conv' in config.branch_types
        assert 'attention' in config.branch_types
        assert config.use_gating is True
        assert config.allow_skip_connections is True
    
    def test_mixed_precision_config(self):
        """Test MixedPrecisionConfig"""
        config = MixedPrecisionConfig()
        
        assert PrecisionLevel.FP32 in config.available_precisions
        assert PrecisionLevel.FP16 in config.available_precisions
        assert PrecisionLevel.INT8 in config.available_precisions
        assert config.sensitivity_threshold == 0.01
        assert config.preserve_first_last is True
        assert config.fp16_beneficial is False  # RX 580 default


# ============================================================================
# Test PrecisionLevel Enum
# ============================================================================

class TestPrecisionLevel:
    """Test PrecisionLevel enum"""
    
    def test_precision_bits(self):
        """Test bit-width property"""
        assert PrecisionLevel.FP32.bits == 32
        assert PrecisionLevel.FP16.bits == 16
        assert PrecisionLevel.INT8.bits == 8
        assert PrecisionLevel.INT4.bits == 4
    
    def test_precision_values(self):
        """Test enum values"""
        assert PrecisionLevel.FP32.value == "fp32"
        assert PrecisionLevel.FP16.value == "fp16"
        assert PrecisionLevel.INT8.value == "int8"
        assert PrecisionLevel.INT4.value == "int4"


# ============================================================================
# Test MultiBranchOperation
# ============================================================================

class TestMultiBranchOperation:
    """Test multi-branch operations"""
    
    def test_multi_branch_creation(self):
        """Test MultiBranchOperation creation"""
        op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv', 'attention', 'identity'],
            use_gating=True
        )
        
        assert op.num_branches == 3
        assert len(op.branches) == 3
        assert op.use_gating is True
    
    def test_multi_branch_forward(self):
        """Test forward pass"""
        op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv', 'identity'],
            use_gating=True
        )
        
        x = torch.randn(4, 32, 16, 16)
        output = op(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_multi_branch_no_gating(self):
        """Test without learnable gating"""
        op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv', 'identity'],
            use_gating=False
        )
        
        # Gate weights should be fixed
        assert not op.gate_weights.requires_grad
        
        x = torch.randn(4, 32, 16, 16)
        output = op(x)
        
        assert output.shape == x.shape
    
    def test_dominant_branch(self):
        """Test get_dominant_branch"""
        op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv', 'attention', 'identity'],
            use_gating=True
        )
        
        branch_type, weight = op.get_dominant_branch()
        
        assert branch_type in ['conv', 'attention', 'identity']
        assert 0 <= weight <= 1.0
    
    def test_branch_types(self):
        """Test different branch types"""
        # Conv branch
        op_conv = MultiBranchOperation(
            C=32, stride=1, branch_types=['conv'], use_gating=False
        )
        x = torch.randn(2, 32, 8, 8)
        out = op_conv(x)
        assert out.shape == x.shape
        
        # Attention branch
        op_att = MultiBranchOperation(
            C=32, stride=1, branch_types=['attention'], use_gating=False
        )
        out = op_att(x)
        assert out.shape == x.shape
        
        # Identity branch
        op_id = MultiBranchOperation(
            C=32, stride=1, branch_types=['identity'], use_gating=False
        )
        out = op_id(x)
        assert out.shape == x.shape


# ============================================================================
# Test ProgressiveNAS
# ============================================================================

class TestProgressiveNAS:
    """Test progressive architecture search"""
    
    def test_progressive_nas_creation(self):
        """Test ProgressiveNAS initialization"""
        darts_config = DARTSConfig(num_cells=4, num_nodes=3)
        darts_config.num_classes = 10
        progressive_config = ProgressiveConfig(
            coarse_epochs=2,
            medium_epochs=3,
            fine_epochs=4
        )
        
        nas = ProgressiveNAS(darts_config, progressive_config)
        
        assert nas.darts_config == darts_config
        assert nas.progressive_config == progressive_config
        assert len(nas.search_history) == 3  # 3 stages
    
    def test_create_candidate_model(self):
        """Test candidate model creation"""
        darts_config = DARTSConfig(num_cells=4, num_nodes=3)
        darts_config.num_classes = 10
        progressive_config = ProgressiveConfig()
        
        nas = ProgressiveNAS(darts_config, progressive_config)
        model = nas._create_candidate_model()
        
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_evaluate(self, simple_model, simple_dataloader):
        """Test model evaluation"""
        darts_config = DARTSConfig(num_cells=4, num_nodes=3)
        darts_config.num_classes = 10
        progressive_config = ProgressiveConfig()
        
        nas = ProgressiveNAS(darts_config, progressive_config)
        
        accuracy = nas._evaluate(simple_model, simple_dataloader)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_coarse_search(self, simple_dataloader):
        """Test coarse search stage"""
        darts_config = DARTSConfig(num_cells=4, num_nodes=3)
        darts_config.num_classes = 10
        progressive_config = ProgressiveConfig(
            coarse_epochs=1,  # Quick test
            coarse_keep_ratio=0.5
        )
        
        nas = ProgressiveNAS(darts_config, progressive_config)
        
        candidates = nas._coarse_search(simple_dataloader, simple_dataloader)
        
        assert len(candidates) > 0
        assert len(candidates) <= 5  # Max candidates
        assert all('accuracy' in c for c in candidates)
        assert all('stage' in c for c in candidates)
        
        # Check sorting (descending accuracy)
        accuracies = [c['accuracy'] for c in candidates]
        assert accuracies == sorted(accuracies, reverse=True)
    
    def test_search_history_tracking(self, simple_dataloader):
        """Test that search history is tracked"""
        darts_config = DARTSConfig(num_cells=4, num_nodes=3)
        darts_config.num_classes = 10
        progressive_config = ProgressiveConfig(coarse_epochs=1)
        
        nas = ProgressiveNAS(darts_config, progressive_config)
        
        # Run coarse search
        nas._coarse_search(simple_dataloader, simple_dataloader)
        
        # Check history
        assert SearchStage.COARSE in nas.search_history
        assert len(nas.search_history[SearchStage.COARSE]) > 0


# ============================================================================
# Test MixedPrecisionNAS
# ============================================================================

class TestMixedPrecisionNAS:
    """Test mixed precision NAS"""
    
    def test_mixed_precision_creation(self):
        """Test MixedPrecisionNAS initialization"""
        config = MixedPrecisionConfig()
        nas = MixedPrecisionNAS(config)
        
        assert nas.config == config
        assert len(nas.layer_sensitivity) == 0
        assert len(nas.precision_map) == 0
    
    def test_simulate_quantization(self, simple_model):
        """Test quantization simulation"""
        config = MixedPrecisionConfig()
        nas = MixedPrecisionNAS(config)
        
        # Get a weight tensor
        weight = simple_model.conv1.weight.data
        
        # Simulate INT8 quantization
        quantized = nas._simulate_quantization(weight, PrecisionLevel.INT8)
        
        assert quantized.shape == weight.shape
        assert not torch.equal(quantized, weight)  # Should be different
        
        # FP32 should return unchanged
        fp32 = nas._simulate_quantization(weight, PrecisionLevel.FP32)
        assert torch.equal(fp32, weight)
    
    def test_evaluate_accuracy(self, simple_model, simple_dataloader):
        """Test accuracy evaluation"""
        config = MixedPrecisionConfig(sensitivity_samples=50)
        nas = MixedPrecisionNAS(config)
        
        accuracy = nas._evaluate_accuracy(simple_model, simple_dataloader)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_measure_sensitivities(self, simple_model, simple_dataloader):
        """Test sensitivity measurement"""
        config = MixedPrecisionConfig(sensitivity_samples=50)
        nas = MixedPrecisionNAS(config)
        
        baseline_acc = 0.5  # Mock baseline
        
        nas._measure_sensitivities(simple_model, simple_dataloader, baseline_acc)
        
        # Should have sensitivity for Conv2d and Linear layers
        assert len(nas.layer_sensitivity) > 0
        
        # Check that we measured conv and fc layers
        layer_names = list(nas.layer_sensitivity.keys())
        assert any('conv' in name for name in layer_names)
        assert any('fc' in name for name in layer_names)
    
    def test_assign_precisions(self, simple_model):
        """Test precision assignment"""
        config = MixedPrecisionConfig(
            preserve_first_last=True,
            sensitivity_threshold=0.01
        )
        nas = MixedPrecisionNAS(config)
        
        # Mock sensitivity data
        nas.layer_sensitivity = {
            'conv1': 0.05,  # High sensitivity
            'conv2': 0.005,  # Low sensitivity
            'fc': 0.02  # Medium sensitivity
        }
        
        nas._assign_precisions(simple_model)
        
        assert len(nas.precision_map) == 3
        
        # conv1 should be FP32 (first layer or high sensitivity)
        assert nas.precision_map['conv1'] == PrecisionLevel.FP32
        
        # conv2 should be INT8 (low sensitivity)
        assert nas.precision_map['conv2'] == PrecisionLevel.INT8
    
    def test_count_precisions(self):
        """Test precision counting"""
        config = MixedPrecisionConfig()
        nas = MixedPrecisionNAS(config)
        
        # Set up precision map
        nas.precision_map = {
            'layer1': PrecisionLevel.FP32,
            'layer2': PrecisionLevel.FP32,
            'layer3': PrecisionLevel.INT8,
            'layer4': PrecisionLevel.FP16
        }
        
        counts = nas._count_precisions()
        
        assert counts['fp32'] == 2
        assert counts['int8'] == 1
        assert counts['fp16'] == 1
    
    def test_analyze_and_assign(self, simple_model, simple_dataloader):
        """Test full analyze and assign pipeline"""
        config = MixedPrecisionConfig(sensitivity_samples=50)
        nas = MixedPrecisionNAS(config)
        
        precision_map = nas.analyze_and_assign(simple_model, simple_dataloader)
        
        assert isinstance(precision_map, dict)
        assert len(precision_map) > 0
        
        # All values should be PrecisionLevel
        for precision in precision_map.values():
            assert isinstance(precision, PrecisionLevel)
    
    def test_apply_precision_map(self, simple_model):
        """Test applying precision map"""
        config = MixedPrecisionConfig()
        nas = MixedPrecisionNAS(config)
        
        # Set up precision map
        nas.precision_map = {
            'conv1': PrecisionLevel.FP32,
            'conv2': PrecisionLevel.INT8,
            'fc': PrecisionLevel.FP32
        }
        
        # Apply (currently just logs, doesn't modify)
        model = nas.apply_precision_map(simple_model)
        
        assert model is not None


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_progressive_nas(self):
        """Test progressive NAS factory"""
        nas = create_progressive_nas(num_classes=100)
        
        assert isinstance(nas, ProgressiveNAS)
        assert nas.darts_config.num_classes == 100
        assert nas.progressive_config.coarse_epochs == 5
    
    def test_create_mixed_precision_nas(self):
        """Test mixed precision NAS factory"""
        # For RX 580 (no FP16 benefit)
        nas_rx580 = create_mixed_precision_nas(fp16_beneficial=False)
        
        assert isinstance(nas_rx580, MixedPrecisionNAS)
        assert nas_rx580.config.fp16_beneficial is False
        
        # For GPU with FP16 acceleration
        nas_fp16 = create_mixed_precision_nas(fp16_beneficial=True)
        assert nas_fp16.config.fp16_beneficial is True


# ============================================================================
# Test Integration
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_progressive_with_mixed_precision(self, simple_dataloader):
        """Test progressive NAS with mixed precision"""
        # Create progressive NAS
        progressive = create_progressive_nas(num_classes=10)
        
        # Create mixed precision NAS
        mixed_precision = create_mixed_precision_nas(fp16_beneficial=False)
        
        # Create a model
        model = progressive._create_candidate_model()
        
        # Analyze precision
        precision_map = mixed_precision.analyze_and_assign(
            model,
            simple_dataloader
        )
        
        assert len(precision_map) > 0
        
        # Apply precision
        model = mixed_precision.apply_precision_map(model)
        
        assert model is not None
    
    def test_multi_branch_in_search(self):
        """Test multi-branch operations in search"""
        # Create multi-branch operation
        branch_op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv', 'attention', 'identity'],
            use_gating=True
        )
        
        # Test forward
        x = torch.randn(4, 32, 16, 16)
        output = branch_op(x)
        
        assert output.shape == x.shape
        
        # Get dominant branch
        branch_type, weight = branch_op.get_dominant_branch()
        
        assert branch_type in ['conv', 'attention', 'identity']
        assert weight > 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_dataloader(self):
        """Test with empty dataloader"""
        class EmptyDataLoader:
            def __init__(self):
                self.batch_size = 16
            def __iter__(self):
                return iter([])
        
        empty_loader = EmptyDataLoader()
        
        config = MixedPrecisionConfig()
        nas = MixedPrecisionNAS(config)
        
        # Should handle gracefully
        model = nn.Linear(10, 10)
        accuracy = nas._evaluate_accuracy(model, empty_loader)
        
        assert accuracy == 0.0
    
    def test_single_branch(self):
        """Test multi-branch with single branch"""
        op = MultiBranchOperation(
            C=32,
            stride=1,
            branch_types=['conv'],
            use_gating=True
        )
        
        x = torch.randn(2, 32, 8, 8)
        output = op(x)
        
        assert output.shape == x.shape
        
        branch_type, weight = op.get_dominant_branch()
        assert branch_type == 'conv'
        assert weight == 1.0
    
    def test_zero_keep_ratio(self):
        """Test progressive with extreme keep ratios"""
        config = ProgressiveConfig(
            coarse_keep_ratio=0.01,  # Keep almost nothing
            medium_keep_ratio=0.01
        )
        
        assert config.coarse_keep_ratio == 0.01


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_multi_branch_forward_speed(self):
        """Test multi-branch forward pass speed"""
        import time
        
        op = MultiBranchOperation(
            C=64,
            stride=1,
            branch_types=['conv', 'attention', 'identity'],
            use_gating=True
        )
        
        x = torch.randn(16, 64, 32, 32)
        
        # Warmup
        for _ in range(10):
            _ = op(x)
        
        # Time
        start = time.time()
        for _ in range(100):
            _ = op(x)
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 10.0  # 10 seconds for 100 iterations
    
    def test_sensitivity_analysis_speed(self, simple_model, simple_dataloader):
        """Test sensitivity analysis speed"""
        import time
        
        config = MixedPrecisionConfig(sensitivity_samples=10)  # Small for speed
        nas = MixedPrecisionNAS(config)
        
        start = time.time()
        baseline_acc = nas._evaluate_accuracy(simple_model, simple_dataloader)
        nas._measure_sensitivities(simple_model, simple_dataloader, baseline_acc)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
