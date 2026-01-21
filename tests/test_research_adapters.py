"""
Tests for Research Integration Adapters - Session 20
====================================================

Tests for adapter classes that bridge new research modules with existing modules.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compute.research_adapters import (
    STDPAdapter,
    EvolutionaryPrunerAdapter,
    PINNQuantizationAdapter,
    SNNHybridAdapter,
    create_adapted_snn,
    create_adapted_pruner
)

from src.compute.snn_homeostasis import (
    HomeostaticSTDP,
    HomeostasisConfig,
    HomeostaticSpikingLayer
)

from src.compute.evolutionary_pruning import (
    EvolutionaryPruner,
    EvolutionaryConfig
)

from src.compute.physics_utils import (
    PINNNetwork,
    create_heat_pinn
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


@pytest.fixture
def homeostatic_stdp(device):
    """Create HomeostaticSTDP instance."""
    config = HomeostasisConfig()
    return HomeostaticSTDP(
        in_features=64,
        out_features=32,
        config=config,
        device=device
    )


@pytest.fixture
def homeostatic_layer(device):
    """Create HomeostaticSpikingLayer instance."""
    config = HomeostasisConfig(
        enable_synaptic_scaling=True,
        enable_intrinsic_plasticity=True,
        enable_sleep_cycles=True
    )
    return HomeostaticSpikingLayer(
        in_features=64,
        out_features=32,
        config=config,
        device=device
    )


# ============================================================================
# STDP Adapter Tests
# ============================================================================

class TestSTDPAdapter:
    """Test STDP adapter functionality."""
    
    def test_adapter_creation(self, homeostatic_stdp):
        """Test creating STDP adapter."""
        adapter = STDPAdapter(homeostatic_stdp)
        
        assert adapter.homeostatic_stdp is not None
        assert adapter.compatibility_mode == 'strict'
    
    def test_legacy_params_interface(self, homeostatic_stdp):
        """Test that adapter provides legacy STDPParams-like interface."""
        adapter = STDPAdapter(homeostatic_stdp)
        
        # Should have params attribute
        assert hasattr(adapter, 'params')
    
    def test_update_method(self, homeostatic_stdp, device):
        """Test update method with layer."""
        adapter = STDPAdapter(homeostatic_stdp)
        
        # Create test layer
        layer = nn.Linear(64, 32).to(device)
        
        # Create test spikes
        pre_spikes = (torch.rand(8, 64, device=device) > 0.9).float()
        post_spikes = (torch.rand(8, 32, device=device) > 0.9).float()
        
        # Get initial weights
        initial_weights = layer.weight.data.clone()
        
        # Apply update
        adapter.update(layer, pre_spikes, post_spikes, learning_rate=0.1)
        
        # Weights should have changed
        assert not torch.allclose(layer.weight.data, initial_weights)
    
    def test_from_layer_factory(self, device):
        """Test creating adapter from layer."""
        layer = nn.Linear(64, 32).to(device)
        
        adapter = STDPAdapter.from_layer(layer)
        
        assert adapter.homeostatic_stdp.in_features == 64
        assert adapter.homeostatic_stdp.out_features == 32
    
    def test_get_statistics(self, homeostatic_stdp):
        """Test getting statistics."""
        adapter = STDPAdapter(homeostatic_stdp)
        
        stats = adapter.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'A_plus' in stats
        assert 'A_minus' in stats
    
    def test_metaplasticity_state(self, homeostatic_stdp):
        """Test getting metaplasticity state."""
        adapter = STDPAdapter(homeostatic_stdp, compatibility_mode='enhanced')
        
        state = adapter.get_metaplasticity_state()
        
        assert 'A_plus' in state
        assert 'A_minus' in state
        assert 'post_activity_avg' in state
        assert 'synaptic_tags' in state


# ============================================================================
# Evolutionary Pruner Adapter Tests
# ============================================================================

class TestEvolutionaryPrunerAdapter:
    """Test evolutionary pruner adapter."""
    
    def test_adapter_requires_trained_pruner(self, simple_model, device):
        """Test that adapter requires trained pruner."""
        config = EvolutionaryConfig(
            population_size=3,
            generations=2
        )
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        # Should raise error before evolution
        with pytest.raises(ValueError, match="Must run evolution"):
            EvolutionaryPrunerAdapter(pruner)
    
    def test_adapter_with_trained_pruner(self, simple_model, device):
        """Test adapter with trained pruner."""
        config = EvolutionaryConfig(
            population_size=3,
            generations=1,
            initial_sparsity=0.3,
            target_sparsity=0.7
        )
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        # Manually set best_individual (simulate evolution)
        pruner.best_individual = {}
        for name, layer in pruner.prunable_layers.items():
            mask = torch.ones_like(layer.weight.data)
            pruner.best_individual[name] = mask
        
        # Now adapter should work
        adapter = EvolutionaryPrunerAdapter(pruner)
        
        assert adapter.pruner is not None
    
    def test_compression_stats(self, simple_model, device):
        """Test getting compression statistics."""
        config = EvolutionaryConfig(population_size=2, generations=1)
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        # Simulate pruned masks
        pruner.best_individual = {}
        for name, layer in pruner.prunable_layers.items():
            mask = (torch.rand_like(layer.weight.data) > 0.5).float()
            pruner.best_individual[name] = mask
        
        adapter = EvolutionaryPrunerAdapter(pruner)
        stats = adapter.get_compression_stats()
        
        assert 'per_layer' in stats
        assert 'overall' in stats
        assert stats['overall']['total_sparsity'] >= 0
        assert stats['overall']['total_sparsity'] <= 1


# ============================================================================
# PINN Quantization Adapter Tests
# ============================================================================

class TestPINNQuantizationAdapter:
    """Test PINN quantization adapter."""
    
    def test_adapter_creation(self, device):
        """Test creating PINN quantization adapter."""
        pinn = create_heat_pinn(
            input_dim=2,
            hidden_dims=[32, 32],
            diffusivity=0.01
        )
        
        adapter = PINNQuantizationAdapter(pinn)
        
        assert adapter.pinn is not None
        assert adapter.quantized_model is None
    
    def test_quantize_warning_without_quantizer(self, device):
        """Test quantization when quantizer not available."""
        pinn = create_heat_pinn(input_dim=2, hidden_dims=[16, 16])
        adapter = PINNQuantizationAdapter(pinn)
        
        # May raise ImportError if quantizer not available
        # This is expected behavior
        try:
            quantized = adapter.quantize(precision='int8')
            assert quantized is not None
        except ImportError:
            pytest.skip("AdaptiveQuantizer not available")


# ============================================================================
# SNN Hybrid Adapter Tests
# ============================================================================

class TestSNNHybridAdapter:
    """Test SNN hybrid adapter."""
    
    def test_adapter_creation(self, homeostatic_layer):
        """Test creating SNN hybrid adapter."""
        adapter = SNNHybridAdapter(homeostatic_layer)
        
        assert adapter.snn_layer is not None
    
    def test_forward_fallback(self, homeostatic_layer, device):
        """Test forward pass without hybrid scheduler (fallback)."""
        adapter = SNNHybridAdapter(homeostatic_layer, hybrid_scheduler=None)
        
        input_spikes = (torch.rand(4, 64, device=device) > 0.9).float()
        
        output = adapter.forward_hybrid(input_spikes)
        
        assert output.shape == (4, 32)
        assert ((output == 0) | (output == 1)).all()
    
    def test_partitioning_stats(self, homeostatic_layer):
        """Test getting partitioning statistics."""
        adapter = SNNHybridAdapter(homeostatic_layer)
        
        stats = adapter.get_partitioning_stats()
        
        assert 'spike_processing' in stats
        assert 'stdp_updates' in stats


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_adapted_snn_homeostatic(self):
        """Test creating adapted SNN with homeostasis."""
        snn = create_adapted_snn(
            in_features=64,
            out_features=32,
            use_homeostasis=True,
            use_hybrid=False
        )
        
        assert isinstance(snn, HomeostaticSpikingLayer)
    
    def test_create_adapted_snn_hybrid(self):
        """Test creating adapted SNN with hybrid scheduler."""
        snn = create_adapted_snn(
            in_features=64,
            out_features=32,
            use_homeostasis=True,
            use_hybrid=True
        )
        
        # May be HomeostaticSpikingLayer or SNNHybridAdapter depending on imports
        assert snn is not None
    
    def test_create_adapted_pruner_function(self, simple_model):
        """Test creating adapted pruner."""
        config = EvolutionaryConfig(
            population_size=2,
            generations=1
        )
        
        adapter = create_adapted_pruner(
            model=simple_model,
            config=config,
            export_format='auto'
        )
        
        assert isinstance(adapter, EvolutionaryPrunerAdapter)


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdapterIntegration:
    """Integration tests for multiple adapters."""
    
    def test_stdp_adapter_with_homeostatic_layer(self, homeostatic_layer, device):
        """Test STDP adapter working with homeostatic layer."""
        # Get STDP from layer
        stdp = homeostatic_layer.stdp
        
        # Create adapter
        adapter = STDPAdapter(stdp)
        
        # Create test layer
        layer = nn.Linear(64, 32).to(device)
        
        # Test update
        pre_spikes = (torch.rand(4, 64, device=device) > 0.9).float()
        post_spikes = (torch.rand(4, 32, device=device) > 0.9).float()
        
        adapter.update(layer, pre_spikes, post_spikes)
        
        # Should not raise error
        assert True
    
    def test_multiple_adapters_compatibility(self, device):
        """Test that multiple adapters can coexist."""
        # Create homeostatic layer
        config = HomeostasisConfig()
        layer = HomeostaticSpikingLayer(
            in_features=64,
            out_features=32,
            config=config,
            device=device
        )
        
        # Create STDP adapter
        stdp_adapter = STDPAdapter(layer.stdp)
        
        # Create hybrid adapter
        hybrid_adapter = SNNHybridAdapter(layer)
        
        # Both should work
        assert stdp_adapter is not None
        assert hybrid_adapter is not None


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestAdapterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_adapter_with_none_scheduler(self, homeostatic_layer):
        """Test hybrid adapter with None scheduler."""
        adapter = SNNHybridAdapter(homeostatic_layer, hybrid_scheduler=None)
        
        # Should create default scheduler or work without
        assert adapter.snn_layer is not None
    
    def test_factory_with_invalid_options(self):
        """Test factory with invalid configuration."""
        # Should handle gracefully or raise clear error
        try:
            snn = create_adapted_snn(
                in_features=0,  # Invalid
                out_features=32,
                use_homeostasis=True
            )
        except (ValueError, AssertionError):
            # Expected behavior
            pass


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
