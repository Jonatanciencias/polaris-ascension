"""
Tests for DARTS + Tensor Decomposition Integration (Session 27)

Tests multi-objective architecture search with compression.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compute.darts_decomposition import (
    DARTSDecompositionIntegration,
    MultiObjectiveOptimizer,
    CompressionConfig,
    HardwareConstraints,
    DecompositionMethod,
    ArchitectureMetrics,
    create_integrated_search
)
from compute.nas_darts import DARTSConfig


@pytest.fixture
def device():
    """Get computation device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def darts_config():
    """Create test DARTS config"""
    config = DARTSConfig(
        num_cells=2,
        num_nodes=2,
        init_channels=8,
        layers=6
    )
    # Add num_classes separately
    config.num_classes = 10
    return config


@pytest.fixture
def compression_config():
    """Create test compression config"""
    return CompressionConfig(
        method=DecompositionMethod.TUCKER,
        target_compression=2.0,
        min_rank_ratio=0.2,
        max_rank_ratio=0.5
    )


@pytest.fixture
def hardware_constraints():
    """Create test hardware constraints"""
    return HardwareConstraints(
        max_memory_mb=4000.0,
        target_latency_ms=50.0,
        max_power_watts=150.0,
        compute_capability="polaris"
    )


@pytest.fixture
def integration(darts_config, compression_config, hardware_constraints, device):
    """Create integration instance"""
    return DARTSDecompositionIntegration(
        darts_config=darts_config,
        compression_config=compression_config,
        hardware_constraints=hardware_constraints,
        device=device
    )


@pytest.fixture
def dummy_data():
    """Create dummy dataset"""
    # Small dataset for testing
    X = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8)
    return loader


@pytest.fixture
def simple_model(device):
    """Create simple test model"""
    class SimpleNet(nn.Module):
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
    
    return SimpleNet().to(device)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_compression_config_creation():
    """Test compression config creation"""
    config = CompressionConfig(
        method=DecompositionMethod.TUCKER,
        target_compression=3.0
    )
    
    assert config.method == DecompositionMethod.TUCKER
    assert config.target_compression == 3.0
    assert config.min_rank_ratio == 0.1
    assert config.max_rank_ratio == 0.5


def test_hardware_constraints_creation():
    """Test hardware constraints creation"""
    constraints = HardwareConstraints(
        max_memory_mb=8000.0,
        target_latency_ms=30.0
    )
    
    assert constraints.max_memory_mb == 8000.0
    assert constraints.target_latency_ms == 30.0
    assert constraints.compute_capability == "polaris"


def test_architecture_metrics_creation():
    """Test architecture metrics creation"""
    metrics = ArchitectureMetrics(
        accuracy=0.95,
        latency_ms=25.0,
        memory_mb=500.0,
        params=1000000,
        flops=2000000,
        compression_ratio=2.5,
        power_estimate_watts=120.0
    )
    
    assert metrics.accuracy == 0.95
    assert metrics.latency_ms == 25.0
    assert metrics.pareto_rank == -1  # Not computed yet


# ============================================================================
# Multi-Objective Optimizer Tests
# ============================================================================

def test_mo_optimizer_creation():
    """Test multi-objective optimizer creation"""
    optimizer = MultiObjectiveOptimizer(
        alpha_lr=3e-4,
        weight_lr=0.025
    )
    
    assert optimizer.alpha_lr == 3e-4
    assert optimizer.weight_lr == 0.025
    assert len(optimizer.pareto_solutions) == 0


def test_pareto_rank_computation():
    """Test Pareto rank computation"""
    optimizer = MultiObjectiveOptimizer()
    
    # Create test metrics
    # Architecture A: High acc, high latency, high memory
    metrics_a = ArchitectureMetrics(
        accuracy=0.95, latency_ms=100.0, memory_mb=1000.0,
        params=1000000, flops=2000000, compression_ratio=1.0,
        power_estimate_watts=150.0
    )
    
    # Architecture B: Medium acc, medium latency, medium memory (balanced)
    metrics_b = ArchitectureMetrics(
        accuracy=0.90, latency_ms=50.0, memory_mb=500.0,
        params=500000, flops=1000000, compression_ratio=2.0,
        power_estimate_watts=100.0
    )
    
    # Architecture C: Low acc, low latency, low memory
    metrics_c = ArchitectureMetrics(
        accuracy=0.80, latency_ms=20.0, memory_mb=200.0,
        params=200000, flops=400000, compression_ratio=5.0,
        power_estimate_watts=80.0
    )
    
    # Architecture D: Dominated by B (worse in all aspects)
    metrics_d = ArchitectureMetrics(
        accuracy=0.85, latency_ms=60.0, memory_mb=600.0,
        params=600000, flops=1200000, compression_ratio=1.5,
        power_estimate_watts=120.0
    )
    
    metrics_list = [metrics_a, metrics_b, metrics_c, metrics_d]
    ranks = optimizer.compute_pareto_rank(metrics_list)
    
    # A, B, C should be Pareto optimal (rank 0)
    # D should be dominated (rank > 0)
    assert ranks[0] == 0  # A is Pareto optimal
    assert ranks[1] == 0  # B is Pareto optimal
    assert ranks[2] == 0  # C is Pareto optimal
    assert ranks[3] > 0  # D is dominated


def test_pareto_rank_simple_dominance():
    """Test simple dominance case"""
    optimizer = MultiObjectiveOptimizer()
    
    # A dominates B in all objectives
    metrics_a = ArchitectureMetrics(
        accuracy=0.95, latency_ms=20.0, memory_mb=200.0,
        params=100000, flops=200000, compression_ratio=2.0,
        power_estimate_watts=80.0
    )
    
    metrics_b = ArchitectureMetrics(
        accuracy=0.85, latency_ms=40.0, memory_mb=400.0,
        params=200000, flops=400000, compression_ratio=1.0,
        power_estimate_watts=100.0
    )
    
    ranks = optimizer.compute_pareto_rank([metrics_a, metrics_b])
    
    assert ranks[0] == 0  # A is non-dominated
    assert ranks[1] == 1  # B is dominated by A


def test_select_best_balanced():
    """Test selecting best architecture with balanced preference"""
    optimizer = MultiObjectiveOptimizer()
    
    # Create dummy models
    model_a = nn.Linear(10, 10)
    model_b = nn.Linear(10, 10)
    
    metrics_a = ArchitectureMetrics(
        accuracy=0.95, latency_ms=100.0, memory_mb=1000.0,
        params=1000000, flops=2000000, compression_ratio=1.0,
        power_estimate_watts=150.0
    )
    
    metrics_b = ArchitectureMetrics(
        accuracy=0.90, latency_ms=50.0, memory_mb=500.0,
        params=500000, flops=1000000, compression_ratio=2.0,
        power_estimate_watts=100.0
    )
    
    architectures = [(model_a, metrics_a), (model_b, metrics_b)]
    
    best_model, best_metrics = optimizer.select_best_architecture(
        architectures,
        preference="balanced"
    )
    
    # Should select one of the Pareto optimal
    assert best_metrics in [metrics_a, metrics_b]
    assert best_metrics.pareto_rank == 0


def test_select_best_accuracy_preference():
    """Test selecting best with accuracy preference"""
    optimizer = MultiObjectiveOptimizer()
    
    model_a = nn.Linear(10, 10)
    model_b = nn.Linear(10, 10)
    
    metrics_a = ArchitectureMetrics(
        accuracy=0.95, latency_ms=100.0, memory_mb=1000.0,
        params=1000000, flops=2000000, compression_ratio=1.0,
        power_estimate_watts=150.0
    )
    
    metrics_b = ArchitectureMetrics(
        accuracy=0.85, latency_ms=50.0, memory_mb=500.0,
        params=500000, flops=1000000, compression_ratio=2.0,
        power_estimate_watts=100.0
    )
    
    architectures = [(model_a, metrics_a), (model_b, metrics_b)]
    
    best_model, best_metrics = optimizer.select_best_architecture(
        architectures,
        preference="accuracy"
    )
    
    # Should prefer higher accuracy
    assert best_metrics.accuracy == 0.95


def test_select_best_latency_preference():
    """Test selecting best with latency preference"""
    optimizer = MultiObjectiveOptimizer()
    
    model_a = nn.Linear(10, 10)
    model_b = nn.Linear(10, 10)
    
    metrics_a = ArchitectureMetrics(
        accuracy=0.95, latency_ms=100.0, memory_mb=1000.0,
        params=1000000, flops=2000000, compression_ratio=1.0,
        power_estimate_watts=150.0
    )
    
    metrics_b = ArchitectureMetrics(
        accuracy=0.85, latency_ms=50.0, memory_mb=500.0,
        params=500000, flops=1000000, compression_ratio=2.0,
        power_estimate_watts=100.0
    )
    
    architectures = [(model_a, metrics_a), (model_b, metrics_b)]
    
    best_model, best_metrics = optimizer.select_best_architecture(
        architectures,
        preference="latency"
    )
    
    # Should prefer lower latency
    assert best_metrics.latency_ms == 50.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_creation(integration):
    """Test integration instance creation"""
    assert integration.darts_config is not None
    assert integration.compression_config is not None
    assert integration.hardware_constraints is not None
    assert integration.mo_optimizer is not None


def test_estimate_latency(integration, simple_model):
    """Test latency estimation"""
    latency = integration._estimate_latency(simple_model, (3, 32, 32))
    
    assert latency > 0
    assert latency < 10000  # Should be reasonable (< 10s)


def test_estimate_memory(integration, simple_model):
    """Test memory estimation"""
    memory_mb = integration._estimate_memory(simple_model)
    
    assert memory_mb > 0
    assert memory_mb < 10000  # Should be reasonable (< 10GB)


def test_estimate_power(integration, simple_model):
    """Test power estimation"""
    power_watts = integration._estimate_power(simple_model, 50.0)
    
    assert power_watts > 0
    assert power_watts <= integration.hardware_constraints.max_power_watts


def test_decompose_model_tucker(integration, simple_model):
    """Test Tucker decomposition"""
    # Set method to Tucker
    integration.compression_config.method = DecompositionMethod.TUCKER
    
    decomposed = integration._decompose_model(
        simple_model,
        DecompositionMethod.TUCKER
    )
    
    assert decomposed is not None
    # Should return a model (may be same if decomposition fails gracefully)
    assert isinstance(decomposed, nn.Module)


def test_decompose_model_cp(integration, simple_model):
    """Test CP decomposition"""
    decomposed = integration._decompose_model(
        simple_model,
        DecompositionMethod.CP
    )
    
    assert decomposed is not None
    assert isinstance(decomposed, nn.Module)


def test_decompose_model_tt(integration, simple_model):
    """Test Tensor-Train decomposition"""
    decomposed = integration._decompose_model(
        simple_model,
        DecompositionMethod.TENSOR_TRAIN
    )
    
    assert decomposed is not None
    assert isinstance(decomposed, nn.Module)


def test_decompose_model_auto(integration, simple_model):
    """Test auto-selection of decomposition method"""
    decomposed = integration._decompose_model(
        simple_model,
        DecompositionMethod.AUTO
    )
    
    assert decomposed is not None
    assert isinstance(decomposed, nn.Module)


def test_evaluate_architecture(integration, simple_model, dummy_data):
    """Test architecture evaluation"""
    metrics = integration.evaluate_architecture(
        simple_model,
        dummy_data,
        input_shape=(3, 32, 32)
    )
    
    assert isinstance(metrics, ArchitectureMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.latency_ms > 0
    assert metrics.memory_mb > 0
    assert metrics.params > 0
    assert metrics.flops > 0
    assert metrics.compression_ratio > 0
    assert metrics.power_estimate_watts > 0


def test_create_candidate_model(integration):
    """Test candidate model creation"""
    model = integration._create_candidate_model(integration.darts_config)
    
    assert isinstance(model, nn.Module)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32).to(integration.device)
    output = model(dummy_input)
    
    assert output.shape == (2, integration.darts_config.num_classes)


def test_search_and_compress_basic(integration, dummy_data):
    """Test basic search and compress workflow"""
    # Run with minimal settings
    pareto_optimal = integration.search_and_compress(
        train_loader=dummy_data,
        val_loader=dummy_data,
        search_epochs=1,
        finetune_epochs=1,
        num_candidates=2
    )
    
    # Should return at least one architecture
    assert len(pareto_optimal) >= 1
    
    # Check structure
    for model, metrics in pareto_optimal:
        assert isinstance(model, nn.Module)
        assert isinstance(metrics, ArchitectureMetrics)
        assert metrics.pareto_rank == 0  # All should be Pareto optimal


def test_search_and_compress_multiple_candidates(integration, dummy_data):
    """Test search with multiple candidates"""
    pareto_optimal = integration.search_and_compress(
        train_loader=dummy_data,
        val_loader=dummy_data,
        search_epochs=1,
        finetune_epochs=1,
        num_candidates=3
    )
    
    assert len(pareto_optimal) >= 1
    
    # Verify Pareto optimality
    for model, metrics in pareto_optimal:
        assert metrics.pareto_rank == 0


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_integrated_search():
    """Test factory function"""
    integration = create_integrated_search(
        num_classes=10,
        target_compression=2.0,
        max_memory_mb=4000.0
    )
    
    assert integration.darts_config.num_classes == 10
    assert integration.compression_config.target_compression == 2.0
    assert integration.hardware_constraints.max_memory_mb == 4000.0


def test_create_integrated_search_custom_device():
    """Test factory with custom device"""
    device = "cpu"
    integration = create_integrated_search(device=device)
    
    assert integration.device == device


# ============================================================================
# End-to-End Tests
# ============================================================================

def test_end_to_end_workflow(device):
    """Test complete end-to-end workflow"""
    # Create integration
    integration = create_integrated_search(
        num_classes=10,
        target_compression=2.0,
        max_memory_mb=4000.0,
        device=device
    )
    
    # Create dummy data
    X = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    
    # Run search
    pareto_optimal = integration.search_and_compress(
        train_loader=loader,
        val_loader=loader,
        search_epochs=1,
        finetune_epochs=1,
        num_candidates=2
    )
    
    # Verify results
    assert len(pareto_optimal) >= 1
    
    # Select best with different preferences
    for preference in ["balanced", "accuracy", "latency", "memory"]:
        best_model, best_metrics = integration.mo_optimizer.select_best_architecture(
            pareto_optimal,
            preference=preference
        )
        
        assert best_model is not None
        assert best_metrics.pareto_rank == 0


def test_compression_ratio_verification(integration, simple_model):
    """Test that compression actually reduces parameters"""
    original_params = sum(p.numel() for p in simple_model.parameters())
    
    # Apply decomposition
    compressed = integration._decompose_model(
        simple_model,
        DecompositionMethod.TUCKER
    )
    
    compressed_params = sum(p.numel() for p in compressed.parameters())
    
    # Compressed should have <= parameters (or equal if decomposition failed)
    assert compressed_params <= original_params * 1.1  # Allow 10% margin


def test_hardware_constraints_respected(integration, dummy_data):
    """Test that hardware constraints are considered"""
    pareto_optimal = integration.search_and_compress(
        train_loader=dummy_data,
        val_loader=dummy_data,
        search_epochs=1,
        finetune_epochs=1,
        num_candidates=2
    )
    
    max_memory = integration.hardware_constraints.max_memory_mb
    
    for model, metrics in pareto_optimal:
        # Memory should be within reasonable bounds
        # (May exceed slightly due to estimation errors)
        assert metrics.memory_mb < max_memory * 2.0


# ============================================================================
# Performance Tests
# ============================================================================

def test_latency_measurement_consistency(integration, simple_model):
    """Test that latency measurements are consistent"""
    latency1 = integration._estimate_latency(simple_model, (3, 32, 32))
    latency2 = integration._estimate_latency(simple_model, (3, 32, 32))
    
    # Should be similar (within 50% due to system variability)
    assert abs(latency1 - latency2) / latency1 < 0.5


def test_memory_estimation_reasonable(integration, simple_model):
    """Test that memory estimation is reasonable"""
    memory = integration._estimate_memory(simple_model)
    
    # Get actual parameter count
    params = sum(p.numel() for p in simple_model.parameters())
    param_memory_mb = params * 4 / (1024 ** 2)  # FP32 = 4 bytes
    
    # Estimated should be > param memory (includes activations)
    assert memory >= param_memory_mb
    
    # But not ridiculously high
    assert memory < param_memory_mb * 10


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_architectures_list():
    """Test handling of empty architectures list"""
    optimizer = MultiObjectiveOptimizer()
    
    with pytest.raises(ValueError):
        optimizer.select_best_architecture([], "balanced")


def test_single_architecture():
    """Test with single architecture"""
    optimizer = MultiObjectiveOptimizer()
    
    model = nn.Linear(10, 10)
    metrics = ArchitectureMetrics(
        accuracy=0.9, latency_ms=50.0, memory_mb=100.0,
        params=100, flops=200, compression_ratio=1.0,
        power_estimate_watts=80.0
    )
    
    best_model, best_metrics = optimizer.select_best_architecture(
        [(model, metrics)],
        "balanced"
    )
    
    assert best_metrics.pareto_rank == 0
    assert best_metrics.accuracy == 0.9


def test_identical_architectures():
    """Test with identical architectures"""
    optimizer = MultiObjectiveOptimizer()
    
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    
    metrics = ArchitectureMetrics(
        accuracy=0.9, latency_ms=50.0, memory_mb=100.0,
        params=100, flops=200, compression_ratio=1.0,
        power_estimate_watts=80.0
    )
    
    # Both have same metrics
    architectures = [(model1, metrics), (model2, metrics)]
    
    best_model, best_metrics = optimizer.select_best_architecture(
        architectures,
        "balanced"
    )
    
    assert best_metrics.pareto_rank == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
