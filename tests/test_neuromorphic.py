"""
Tests for Neuromorphic Deployment
=================================

Validates functionality of:
- NeuromorphicDeployment
- Platform-specific exporters
- Power estimation
- Benchmarking
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.deployment.neuromorphic import (
    NeuromorphicDeployment,
    LoihiExporter,
    SpiNNakerExporter,
    NeuromorphicPlatform,
    PlatformConstraints,
    PowerEstimate,
    create_neuromorphic_deployer
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_snn():
    """Create simple SNN for testing."""
    snn = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return snn


@pytest.fixture
def test_data():
    """Create test input data."""
    return torch.randn(32, 784)


@pytest.fixture
def loihi_deployer():
    """Create Loihi deployer."""
    return NeuromorphicDeployment(platform='loihi2', optimization_level=2)


@pytest.fixture
def spinnaker_deployer():
    """Create SpiNNaker deployer."""
    return NeuromorphicDeployment(platform='spinnaker2', optimization_level=2)


# ============================================================================
# Platform Initialization Tests
# ============================================================================

def test_loihi_initialization():
    """Test Loihi platform initialization."""
    deployer = NeuromorphicDeployment(platform='loihi2')
    
    assert deployer.platform == NeuromorphicPlatform.LOIHI2
    assert deployer.constraints.max_neurons_per_core == 1024
    assert deployer.constraints.weight_bits == 8
    assert deployer.constraints.supports_stdp


def test_spinnaker_initialization():
    """Test SpiNNaker platform initialization."""
    deployer = NeuromorphicDeployment(platform='spinnaker2')
    
    assert deployer.platform == NeuromorphicPlatform.SPINNAKER2
    assert deployer.constraints.max_neurons_per_core == 256
    assert deployer.constraints.weight_bits == 16


def test_generic_platform():
    """Test generic platform fallback."""
    deployer = NeuromorphicDeployment(platform='generic')
    
    assert deployer.platform == NeuromorphicPlatform.GENERIC
    assert deployer.constraints is not None


# ============================================================================
# Optimization Tests
# ============================================================================

def test_basic_optimization(loihi_deployer, simple_snn):
    """Test basic SNN optimization."""
    optimized = loihi_deployer.optimize_for_platform(
        simple_snn,
        target_spike_rate=0.1
    )
    
    assert optimized is not None
    assert isinstance(optimized, nn.Module)


def test_weight_quantization(loihi_deployer, simple_snn):
    """Test weight quantization to platform bit-width."""
    original_weights = []
    for module in simple_snn.modules():
        if hasattr(module, 'weight'):
            original_weights.append(module.weight.clone())
    
    optimized = loihi_deployer.optimize_for_platform(simple_snn)
    
    # Check weights are quantized
    idx = 0
    for module in optimized.modules():
        if hasattr(module, 'weight'):
            # Weights should be different (quantized)
            assert not torch.allclose(module.weight, original_weights[idx])
            idx += 1


def test_spike_rate_tuning(loihi_deployer):
    """Test neuron parameter tuning for spike rate."""
    class SNNWithThreshold(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.threshold = 1.0
            self.leak = 0.8
        
        def forward(self, x):
            return self.linear(x)
    
    snn = SNNWithThreshold()
    original_threshold = snn.threshold
    
    optimized = loihi_deployer.optimize_for_platform(
        snn,
        target_spike_rate=0.05  # Low spike rate
    )
    
    # Threshold should be adjusted
    assert optimized.threshold != original_threshold


def test_optimization_levels(simple_snn):
    """Test different optimization levels."""
    # Level 0: no optimization
    deployer_0 = NeuromorphicDeployment(platform='loihi2', optimization_level=0)
    opt_0 = deployer_0.optimize_for_platform(simple_snn)
    
    # Level 2: aggressive optimization
    deployer_2 = NeuromorphicDeployment(platform='loihi2', optimization_level=2)
    opt_2 = deployer_2.optimize_for_platform(simple_snn)
    
    # Both should succeed
    assert opt_0 is not None
    assert opt_2 is not None


# ============================================================================
# Export Tests
# ============================================================================

def test_lava_export(loihi_deployer, simple_snn):
    """Test export to Lava format."""
    export_data = loihi_deployer.export_snn(simple_snn, output_format='lava')
    
    assert export_data['format'] == 'lava'
    assert 'layers' in export_data
    assert len(export_data['layers']) > 0
    assert 'metadata' in export_data


def test_pynn_export(spinnaker_deployer, simple_snn):
    """Test export to PyNN format."""
    export_data = spinnaker_deployer.export_snn(simple_snn, output_format='pynn')
    
    assert export_data['format'] == 'pynn'
    assert 'populations' in export_data
    assert len(export_data['populations']) > 0


def test_spike_json_export(loihi_deployer, simple_snn):
    """Test export to generic spike JSON."""
    export_data = loihi_deployer.export_snn(simple_snn, output_format='spike_json')
    
    assert export_data['format'] == 'spike_json'
    assert 'network' in export_data
    assert 'layers' in export_data['network']


def test_export_with_file(loihi_deployer, simple_snn, tmp_path):
    """Test export with file saving."""
    output_file = tmp_path / "export.json"
    
    export_data = loihi_deployer.export_snn(
        simple_snn,
        output_format='lava',
        output_path=str(output_file)
    )
    
    # Check file was created
    assert output_file.exists()
    
    # Check contents
    with open(output_file) as f:
        loaded_data = json.load(f)
    
    assert loaded_data['format'] == 'lava'


# ============================================================================
# Power Estimation Tests
# ============================================================================

def test_power_estimation(loihi_deployer, simple_snn):
    """Test power consumption estimation."""
    power = loihi_deployer.estimate_power(
        simple_snn,
        input_spike_rate=0.1
    )
    
    assert isinstance(power, PowerEstimate)
    assert power.cpu_power_watts > 0
    assert power.neuromorphic_power_watts > 0
    assert power.reduction_factor > 1.0


def test_power_scaling_with_spike_rate(loihi_deployer, simple_snn):
    """Test power scales with spike rate."""
    power_low = loihi_deployer.estimate_power(simple_snn, input_spike_rate=0.01)
    power_high = loihi_deployer.estimate_power(simple_snn, input_spike_rate=0.5)
    
    # Higher spike rate should consume more power
    assert power_high.neuromorphic_power_watts > power_low.neuromorphic_power_watts


def test_power_reduction_factor(loihi_deployer, simple_snn):
    """Test power reduction vs CPU/GPU."""
    power = loihi_deployer.estimate_power(simple_snn)
    
    # Should show significant reduction (>100x typical)
    assert power.reduction_factor > 50.0
    
    # Neuromorphic should be much lower than CPU
    assert power.neuromorphic_power_watts < power.cpu_power_watts / 50


def test_energy_per_spike(loihi_deployer, simple_snn):
    """Test energy per spike calculation."""
    power = loihi_deployer.estimate_power(simple_snn)
    
    # Should be in picojoule range (typical: 10-50pJ)
    assert 1e-12 < power.energy_per_spike_joules < 1e-9


# ============================================================================
# Benchmarking Tests
# ============================================================================

def test_benchmark_basic(loihi_deployer, simple_snn, test_data):
    """Test basic benchmarking."""
    results = loihi_deployer.benchmark(
        simple_snn,
        test_data,
        num_runs=10
    )
    
    assert 'latency_ms' in results
    assert 'throughput_inferences_per_sec' in results
    assert 'energy_per_inference_uj' in results
    assert results['latency_ms'] > 0
    assert results['throughput_inferences_per_sec'] > 0


def test_benchmark_consistency(loihi_deployer, simple_snn, test_data):
    """Test benchmark consistency across runs."""
    results1 = loihi_deployer.benchmark(simple_snn, test_data, num_runs=20)
    results2 = loihi_deployer.benchmark(simple_snn, test_data, num_runs=20)
    
    # Results should be similar (within 50% due to timing variance)
    ratio = results1['latency_ms'] / results2['latency_ms']
    assert 0.5 < ratio < 2.0


def test_throughput_calculation(loihi_deployer, simple_snn, test_data):
    """Test throughput calculation."""
    results = loihi_deployer.benchmark(simple_snn, test_data, num_runs=50)
    
    # Throughput should be inverse of latency
    expected_throughput = 1000.0 / results['latency_ms']
    
    # Should be close (within 10%)
    assert 0.9 * expected_throughput < results['throughput_inferences_per_sec'] < 1.1 * expected_throughput


# ============================================================================
# Platform-Specific Exporter Tests
# ============================================================================

def test_loihi_exporter(simple_snn):
    """Test LoihiExporter class."""
    exporter = LoihiExporter(optimization_level=2)
    
    assert exporter.platform == NeuromorphicPlatform.LOIHI2
    
    # Test Lava export
    lava_network = exporter.to_lava_network(simple_snn)
    assert lava_network['platform'] == 'loihi2'


def test_spinnaker_exporter(simple_snn):
    """Test SpiNNakerExporter class."""
    exporter = SpiNNakerExporter(optimization_level=2)
    
    assert exporter.platform == NeuromorphicPlatform.SPINNAKER2
    
    # Test PyNN export
    pynn_network = exporter.to_pynn_network(simple_snn)
    assert pynn_network['platform'] == 'spinnaker2'


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_factory_loihi():
    """Test factory function for Loihi."""
    deployer = create_neuromorphic_deployer('loihi2', optimization_level=1)
    
    assert isinstance(deployer, LoihiExporter)
    assert deployer.opt_level == 1


def test_factory_spinnaker():
    """Test factory function for SpiNNaker."""
    deployer = create_neuromorphic_deployer('spinnaker2', optimization_level=2)
    
    assert isinstance(deployer, SpiNNakerExporter)
    assert deployer.opt_level == 2


def test_factory_generic():
    """Test factory function for generic platform."""
    deployer = create_neuromorphic_deployer('generic')
    
    assert isinstance(deployer, NeuromorphicDeployment)
    assert deployer.platform == NeuromorphicPlatform.GENERIC


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_deployment(simple_snn, test_data, tmp_path):
    """Test complete deployment pipeline."""
    # 1. Create deployer
    deployer = NeuromorphicDeployment(platform='loihi2', optimization_level=2)
    
    # 2. Optimize
    optimized = deployer.optimize_for_platform(simple_snn, target_spike_rate=0.1)
    assert optimized is not None
    
    # 3. Export
    export_file = tmp_path / "network.json"
    export_data = deployer.export_snn(
        optimized,
        output_format='lava',
        output_path=str(export_file)
    )
    assert export_file.exists()
    
    # 4. Estimate power
    power = deployer.estimate_power(optimized, input_spike_rate=0.1)
    assert power.reduction_factor > 50.0
    
    # 5. Benchmark
    results = deployer.benchmark(optimized, test_data, num_runs=10)
    assert results['latency_ms'] > 0


def test_multi_platform_export(simple_snn):
    """Test exporting to multiple platforms."""
    platforms = ['loihi2', 'spinnaker2', 'generic']
    
    for platform in platforms:
        deployer = NeuromorphicDeployment(platform=platform)
        
        # Optimize
        optimized = deployer.optimize_for_platform(simple_snn)
        
        # Export
        if platform == 'loihi2':
            export_data = deployer.export_snn(optimized, output_format='lava')
        elif platform == 'spinnaker2':
            export_data = deployer.export_snn(optimized, output_format='pynn')
        else:
            export_data = deployer.export_snn(optimized, output_format='spike_json')
        
        assert export_data is not None


# ============================================================================
# Edge Cases Tests
# ============================================================================

def test_empty_model():
    """Test with empty model."""
    model = nn.Sequential()
    deployer = NeuromorphicDeployment(platform='loihi2')
    
    # Should handle gracefully
    optimized = deployer.optimize_for_platform(model)
    assert optimized is not None


def test_single_layer():
    """Test with single layer."""
    model = nn.Linear(10, 5)
    deployer = NeuromorphicDeployment(platform='loihi2')
    
    optimized = deployer.optimize_for_platform(model)
    export_data = deployer.export_snn(optimized, output_format='lava')
    
    assert len(export_data['layers']) >= 1


def test_large_network():
    """Test with large network."""
    layers = []
    for i in range(10):
        layers.extend([nn.Linear(100, 100), nn.ReLU()])
    model = nn.Sequential(*layers)
    
    deployer = NeuromorphicDeployment(platform='loihi2', optimization_level=1)
    
    # Should handle large networks
    optimized = deployer.optimize_for_platform(model)
    assert optimized is not None


def test_invalid_platform():
    """Test with invalid platform name."""
    with pytest.raises(ValueError):
        NeuromorphicDeployment(platform='invalid_platform')


# ============================================================================
# Performance Tests
# ============================================================================

def test_optimization_speed():
    """Test optimization completes in reasonable time."""
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    deployer = NeuromorphicDeployment(platform='loihi2', optimization_level=2)
    
    import time
    start = time.time()
    
    optimized = deployer.optimize_for_platform(model)
    
    elapsed = time.time() - start
    
    assert optimized is not None
    assert elapsed < 10.0  # Should complete in < 10s


def test_export_speed(simple_snn):
    """Test export completes quickly."""
    deployer = NeuromorphicDeployment(platform='loihi2')
    
    import time
    start = time.time()
    
    export_data = deployer.export_snn(simple_snn, output_format='lava')
    
    elapsed = time.time() - start
    
    assert export_data is not None
    assert elapsed < 5.0  # Should complete in < 5s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
