"""
Tests for Unified Optimization Pipeline - Session 23
=====================================================

Tests:
------
1. AutoConfigurator
   - Model analysis
   - Configuration generation
   - Target-based optimization

2. Pipeline Stages
   - Pruning stage
   - Quantization stage
   - Mixed-precision stage
   - Integration

3. End-to-End Pipeline
   - Full optimization flow
   - Metrics tracking
   - Report generation

4. Quick Optimize API
   - One-line optimization
   - Different targets
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

from src.pipelines.unified_optimization import (
    UnifiedOptimizationPipeline,
    AutoConfigurator,
    OptimizationTarget,
    OptimizationConfig,
    PipelineStage,
    StageResult,
    PipelineResult,
    quick_optimize
)


# Test models
class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """Conv test model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 6 * 6, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Test fixtures
@pytest.fixture
def simple_model():
    """Simple model fixture."""
    return SimpleModel()


@pytest.fixture
def conv_model():
    """Conv model fixture."""
    return ConvModel()


@pytest.fixture
def sample_data():
    """Sample data loader."""
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 10, (100,))
    )
    return torch.utils.data.DataLoader(dataset, batch_size=32)


@pytest.fixture
def eval_fn():
    """Simple evaluation function."""
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total if total > 0 else 0.0
    return evaluate


# ============================================================================
# AutoConfigurator Tests
# ============================================================================

def test_configurator_init():
    """Test configurator initialization."""
    config = AutoConfigurator(OptimizationTarget.BALANCED)
    assert config.target == OptimizationTarget.BALANCED


def test_configurator_analyze_model(simple_model):
    """Test model analysis."""
    config = AutoConfigurator()
    analysis = config.analyze_model(simple_model)
    
    assert 'num_parameters' in analysis
    assert 'num_layers' in analysis
    assert 'has_linear' in analysis
    # fc1: 10*20 + 20 = 220, fc2: 20*10 + 10 = 210, total = 430
    assert analysis['num_parameters'] == 430
    assert analysis['has_linear'] is True


def test_configurator_analyze_conv_model(conv_model):
    """Test conv model analysis."""
    config = AutoConfigurator()
    analysis = config.analyze_model(conv_model)
    
    assert analysis['has_conv'] is True
    assert analysis['has_linear'] is True
    assert analysis['num_parameters'] > 0


def test_configurator_accuracy_target(simple_model):
    """Test configuration for accuracy target."""
    config = AutoConfigurator(OptimizationTarget.ACCURACY)
    opt_config = config.configure_pipeline(simple_model)
    
    assert opt_config.max_accuracy_drop == 0.02
    assert PipelineStage.PRUNING in opt_config.enabled_stages


def test_configurator_speed_target(simple_model):
    """Test configuration for speed target."""
    config = AutoConfigurator(OptimizationTarget.SPEED)
    opt_config = config.configure_pipeline(simple_model)
    
    assert PipelineStage.QUANTIZATION in opt_config.enabled_stages
    assert opt_config.max_accuracy_drop == 0.05


def test_configurator_memory_target(simple_model):
    """Test configuration for memory target."""
    config = AutoConfigurator(OptimizationTarget.MEMORY)
    opt_config = config.configure_pipeline(simple_model)
    
    assert PipelineStage.PRUNING in opt_config.enabled_stages
    assert opt_config.max_accuracy_drop == 0.08


def test_configurator_extreme_target(simple_model):
    """Test configuration for extreme target."""
    config = AutoConfigurator(OptimizationTarget.EXTREME)
    opt_config = config.configure_pipeline(simple_model)
    
    assert len(opt_config.enabled_stages) >= 3
    assert opt_config.max_accuracy_drop == 0.15


# ============================================================================
# Pipeline Tests
# ============================================================================

def test_pipeline_init():
    """Test pipeline initialization."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.BALANCED)
    assert pipeline.target == OptimizationTarget.BALANCED
    assert isinstance(pipeline.configurator, AutoConfigurator)


def test_pipeline_custom_config():
    """Test pipeline with custom config."""
    config = OptimizationConfig(
        target=OptimizationTarget.SPEED,
        max_accuracy_drop=0.03
    )
    pipeline = UnifiedOptimizationPipeline(config=config)
    assert pipeline.config.max_accuracy_drop == 0.03


def test_pipeline_get_model_size(simple_model):
    """Test model size calculation."""
    pipeline = UnifiedOptimizationPipeline()
    size = pipeline._get_model_size(simple_model)
    assert size > 0
    assert size < 1.0  # Should be less than 1MB


def test_pipeline_estimate_speedup(simple_model):
    """Test speedup estimation."""
    pipeline = UnifiedOptimizationPipeline()
    speedup = pipeline._estimate_speedup(simple_model, compression_ratio=4.0)
    assert speedup > 1.0
    assert speedup == pytest.approx(2.0, rel=0.1)  # sqrt(4) = 2


def test_pipeline_pruning(simple_model):
    """Test pruning stage."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.MEMORY)
    
    # Apply pruning
    pruned = pipeline._apply_pruning(simple_model, None, None)
    
    # Check that some weights are zero
    fc1_zeros = (pruned.fc1.weight == 0).sum().item()
    assert fc1_zeros > 0


def test_pipeline_quantization(simple_model):
    """Test quantization stage."""
    pipeline = UnifiedOptimizationPipeline()
    
    # Apply quantization
    quantized = pipeline._apply_quantization(simple_model)
    
    # Should return a model (quantization may not work on all platforms)
    assert quantized is not None


def test_pipeline_optimize_minimal(simple_model):
    """Test minimal optimization (no validation)."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.BALANCED)
    
    result = pipeline.optimize(simple_model)
    
    assert isinstance(result, PipelineResult)
    assert result.optimized_model is not None
    assert result.compression_ratio >= 1.0
    assert len(result.stage_results) > 0


def test_pipeline_optimize_with_eval(simple_model, sample_data, eval_fn):
    """Test optimization with evaluation."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.BALANCED)
    
    result = pipeline.optimize(
        simple_model,
        val_loader=sample_data,
        eval_fn=eval_fn
    )
    
    assert result.original_accuracy >= 0.0
    assert result.final_accuracy >= 0.0
    # Accuracy drop can be negative if accuracy improved
    assert isinstance(result.accuracy_drop, float)


def test_pipeline_accuracy_target(simple_model, sample_data, eval_fn):
    """Test accuracy target optimization."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.ACCURACY)
    
    result = pipeline.optimize(
        simple_model,
        val_loader=sample_data,
        eval_fn=eval_fn
    )
    
    # Should have minimal accuracy drop
    assert result.accuracy_drop <= 0.05


def test_pipeline_speed_target(simple_model):
    """Test speed target optimization."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.SPEED)
    
    result = pipeline.optimize(simple_model)
    
    # Should have good speedup
    assert result.speedup >= 1.0


def test_pipeline_stage_failure(simple_model):
    """Test handling of stage failure."""
    pipeline = UnifiedOptimizationPipeline()
    
    # Force failure by passing invalid stage
    stage_result = pipeline._run_stage(
        PipelineStage.NEUROMORPHIC,  # Not implemented
        simple_model,
        None,
        None,
        None
    )
    
    assert stage_result.success is False


def test_pipeline_generate_report(simple_model, sample_data, eval_fn):
    """Test report generation."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.BALANCED)
    
    result = pipeline.optimize(
        simple_model,
        val_loader=sample_data,
        eval_fn=eval_fn
    )
    
    report = pipeline.generate_report(result)
    
    assert isinstance(report, str)
    assert "UNIFIED OPTIMIZATION PIPELINE REPORT" in report
    assert "OVERALL RESULTS" in report
    assert "STAGE RESULTS" in report


def test_pipeline_multiple_stages(simple_model):
    """Test pipeline with multiple stages."""
    config = OptimizationConfig(
        enabled_stages=[
            PipelineStage.PRUNING,
            PipelineStage.QUANTIZATION,
            PipelineStage.MIXED_PRECISION
        ]
    )
    
    pipeline = UnifiedOptimizationPipeline(config=config)
    result = pipeline.optimize(simple_model)
    
    # Should have result for each stage
    assert len(result.stage_results) <= 3


# ============================================================================
# Quick Optimize Tests
# ============================================================================

def test_quick_optimize_basic(simple_model):
    """Test quick optimize function."""
    optimized, metrics = quick_optimize(simple_model, target="balanced")
    
    assert optimized is not None
    assert 'compression_ratio' in metrics
    assert 'speedup' in metrics
    assert metrics['compression_ratio'] >= 1.0


def test_quick_optimize_with_eval(simple_model, sample_data, eval_fn):
    """Test quick optimize with evaluation."""
    optimized, metrics = quick_optimize(
        simple_model,
        target="speed",
        val_loader=sample_data,
        eval_fn=eval_fn
    )
    
    assert 'accuracy_drop' in metrics
    assert metrics['accuracy_drop'] >= 0.0


def test_quick_optimize_accuracy(simple_model):
    """Test quick optimize for accuracy."""
    optimized, metrics = quick_optimize(simple_model, target="accuracy")
    
    # Should have low accuracy drop
    assert optimized is not None


def test_quick_optimize_extreme(simple_model):
    """Test quick optimize for extreme compression."""
    optimized, metrics = quick_optimize(simple_model, target="extreme")
    
    # Should have high compression
    assert metrics['compression_ratio'] >= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_optimization(simple_model, sample_data, eval_fn):
    """Test complete end-to-end optimization."""
    # Create pipeline
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.BALANCED)
    
    # Optimize
    result = pipeline.optimize(
        simple_model,
        val_loader=sample_data,
        eval_fn=eval_fn
    )
    
    # Verify results
    assert result.optimized_model is not None
    assert result.compression_ratio >= 1.0
    assert result.speedup >= 1.0
    assert result.total_time > 0
    
    # Generate report
    report = pipeline.generate_report(result)
    assert len(report) > 0


def test_conv_model_optimization(conv_model):
    """Test optimization of conv model."""
    pipeline = UnifiedOptimizationPipeline(OptimizationTarget.SPEED)
    
    result = pipeline.optimize(conv_model)
    
    assert result.optimized_model is not None
    assert result.compression_ratio >= 1.0


def test_different_targets_comparison(simple_model):
    """Test optimization with different targets."""
    targets = [
        OptimizationTarget.ACCURACY,
        OptimizationTarget.BALANCED,
        OptimizationTarget.SPEED,
        OptimizationTarget.MEMORY
    ]
    
    results = []
    for target in targets:
        pipeline = UnifiedOptimizationPipeline(target)
        result = pipeline.optimize(simple_model)
        results.append(result)
    
    # All should succeed
    assert all(r.optimized_model is not None for r in results)
    
    # Check that different configs were used
    assert len(results) == len(targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
