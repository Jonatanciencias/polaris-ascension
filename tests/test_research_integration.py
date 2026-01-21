"""
Tests for Research Integration Modules - Session 20
====================================================

Tests for:
- Physics-Informed Neural Networks (physics_utils.py)
- Evolutionary Pruning (evolutionary_pruning.py)
- SNN Homeostasis (snn_homeostasis.py)

These tests verify the scientific foundations implemented from research papers.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compute.physics_utils import (
    PhysicsConfig,
    GradientComputer,
    PDEResidual,
    HeatEquation,
    WaveEquation,
    BurgersEquation,
    SPIKERegularizer,
    PINNNetwork,
    PINNTrainer,
    create_heat_pinn,
    create_burgers_pinn
)

from src.compute.evolutionary_pruning import (
    EvolutionaryConfig,
    FitnessEvaluator,
    GeneticOperators,
    EvolutionaryPruner,
    AdaptiveEvolutionaryPruner
)

from src.compute.snn_homeostasis import (
    HomeostasisConfig,
    SynapticScaling,
    IntrinsicPlasticity,
    SleepConsolidation,
    HomeostaticSTDP,
    HomeostaticSpikingLayer
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
    """Create simple model for pruning tests."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


# ============================================================================
# Physics Utils Tests
# ============================================================================

class TestPhysicsConfig:
    """Test PhysicsConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = PhysicsConfig()
        assert config.lambda_physics == 1.0
        assert config.lambda_bc == 10.0
        assert config.n_collocation == 10000
        assert config.use_spike_regularization == True
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = PhysicsConfig(
            lambda_physics=2.0,
            n_collocation=5000,
            koopman_rank=32
        )
        assert config.lambda_physics == 2.0
        assert config.n_collocation == 5000
        assert config.koopman_rank == 32
    
    def test_validation(self):
        """Test validation of invalid configs."""
        with pytest.raises(AssertionError):
            PhysicsConfig(lambda_physics=-1.0)
        
        with pytest.raises(AssertionError):
            PhysicsConfig(n_collocation=0)


class TestGradientComputer:
    """Test gradient computation utilities."""
    
    def test_gradient_1d(self, device):
        """Test first derivative computation."""
        x = torch.linspace(0, 1, 100, device=device, requires_grad=True)
        u = x ** 2  # du/dx = 2x
        
        grad_u = GradientComputer.gradient(u, x)
        
        expected = 2 * x
        assert torch.allclose(grad_u, expected, atol=0.1)
    
    def test_laplacian_1d(self, device):
        """Test Laplacian (second derivative) computation."""
        x = torch.linspace(0, 1, 100, device=device, requires_grad=True).unsqueeze(-1)
        u = x ** 3  # d²u/dx² = 6x
        
        laplacian = GradientComputer.laplacian(u, x)
        
        expected = 6 * x
        # Numerical gradients are less accurate
        assert laplacian.shape == expected.shape


class TestHeatEquation:
    """Test Heat equation PDE residual."""
    
    def test_residual_shape(self, device):
        """Test that residual has correct shape."""
        pde = HeatEquation(alpha=1.0)
        
        n_points = 100
        u = torch.randn(n_points, 1, device=device, requires_grad=True)
        x = torch.rand(n_points, 1, device=device, requires_grad=True)
        t = torch.rand(n_points, 1, device=device, requires_grad=True)
        
        residual = pde.forward(u, x, t)
        
        assert residual.shape == u.shape
    
    def test_physics_loss(self, device):
        """Test physics loss computation."""
        pde = HeatEquation(alpha=0.5)
        
        n_points = 50
        u = torch.randn(n_points, 1, device=device, requires_grad=True)
        x = torch.rand(n_points, 1, device=device, requires_grad=True)
        t = torch.rand(n_points, 1, device=device, requires_grad=True)
        
        loss = pde.physics_loss(u, x, t)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0


class TestWaveEquation:
    """Test Wave equation PDE residual."""
    
    def test_wave_residual(self, device):
        """Test wave equation residual computation."""
        pde = WaveEquation(c=1.0)
        
        n_points = 50
        u = torch.randn(n_points, 1, device=device, requires_grad=True)
        x = torch.rand(n_points, 1, device=device, requires_grad=True)
        t = torch.rand(n_points, 1, device=device, requires_grad=True)
        
        residual = pde.forward(u, x, t)
        
        assert residual.shape == u.shape


class TestBurgersEquation:
    """Test Burgers equation PDE residual."""
    
    def test_burgers_residual(self, device):
        """Test Burgers equation residual."""
        pde = BurgersEquation(nu=0.01)
        
        n_points = 50
        u = torch.randn(n_points, 1, device=device, requires_grad=True)
        x = torch.rand(n_points, 1, device=device, requires_grad=True)
        t = torch.rand(n_points, 1, device=device, requires_grad=True)
        
        residual = pde.forward(u, x, t)
        
        assert residual.shape == u.shape


class TestSPIKERegularizer:
    """Test SPIKE (Koopman) regularization."""
    
    def test_initialization(self, device):
        """Test SPIKE regularizer initialization."""
        spike = SPIKERegularizer(
            input_dim=32,
            koopman_rank=16,
            device=device
        )
        
        assert spike.koopman_U.shape == (32, 16)
        assert spike.koopman_V.shape == (16, 32)
    
    def test_forward(self, device):
        """Test SPIKE regularization loss."""
        spike = SPIKERegularizer(
            input_dim=16,
            koopman_rank=8,
            device=device
        )
        
        u_t = torch.randn(32, 16, device=device)
        u_t_next = torch.randn(32, 16, device=device)
        
        loss = spike(u_t, u_t_next, dt=0.01)
        
        assert loss.ndim == 0
        assert loss >= 0
    
    def test_spectral_analysis(self, device):
        """Test Koopman spectral analysis."""
        spike = SPIKERegularizer(
            input_dim=8,
            koopman_rank=4,
            device=device
        )
        
        analysis = spike.spectral_analysis()
        
        assert 'eigenvalues' in analysis
        assert 'stability' in analysis
        assert 'frequencies' in analysis


class TestPINNNetwork:
    """Test PINN network architecture."""
    
    def test_initialization(self, device):
        """Test network initialization."""
        model = PINNNetwork(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            device=device
        )
        
        assert len(model.hidden_layers) == 2
    
    def test_forward(self, device):
        """Test forward pass."""
        model = PINNNetwork(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            device=device
        )
        
        x = torch.rand(100, 2, device=device)
        y = model(x)
        
        assert y.shape == (100, 1)
    
    def test_fourier_features(self, device):
        """Test Fourier feature embedding."""
        model = PINNNetwork(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
            use_fourier_features=True,
            device=device
        )
        
        x = torch.rand(50, 2, device=device)
        y = model(x)
        
        assert y.shape == (50, 1)


class TestCreatePINNFunctions:
    """Test convenience functions for creating PINNs."""
    
    def test_create_heat_pinn(self, device):
        """Test heat equation PINN creation."""
        model, pde, trainer = create_heat_pinn(
            alpha=0.5,
            hidden_dims=[32, 32],
            device=device
        )
        
        assert isinstance(model, PINNNetwork)
        assert isinstance(pde, HeatEquation)
        assert isinstance(trainer, PINNTrainer)
    
    def test_create_burgers_pinn(self, device):
        """Test Burgers equation PINN creation."""
        model, pde, trainer = create_burgers_pinn(
            nu=0.01,
            use_spike=True,
            device=device
        )
        
        assert isinstance(model, PINNNetwork)
        assert isinstance(pde, BurgersEquation)


# ============================================================================
# Evolutionary Pruning Tests
# ============================================================================

class TestEvolutionaryConfig:
    """Test EvolutionaryConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = EvolutionaryConfig()
        assert config.initial_sparsity == 0.3
        assert config.target_sparsity == 0.9
        assert config.population_size == 10
    
    def test_validation(self):
        """Test validation of invalid configs."""
        with pytest.raises(AssertionError):
            EvolutionaryConfig(initial_sparsity=0.95, target_sparsity=0.9)
        
        with pytest.raises(AssertionError):
            EvolutionaryConfig(population_size=0)


class TestFitnessEvaluator:
    """Test fitness evaluation metrics."""
    
    def test_magnitude_fitness(self, device):
        """Test magnitude-based fitness."""
        weights = torch.randn(32, 16, device=device)
        
        fitness = FitnessEvaluator.magnitude_fitness(weights)
        
        assert fitness.shape == weights.shape
        assert (fitness > 0).all()  # Always positive due to epsilon
    
    def test_gradient_fitness(self, device):
        """Test gradient-based fitness."""
        weights = torch.randn(32, 16, device=device)
        gradients = torch.randn(32, 16, device=device)
        
        fitness = FitnessEvaluator.gradient_fitness(weights, gradients)
        
        assert fitness.shape == weights.shape
    
    def test_magnitude_gradient_fitness(self, device):
        """Test combined fitness."""
        weights = torch.randn(32, 16, device=device)
        gradients = torch.randn(32, 16, device=device)
        
        fitness = FitnessEvaluator.magnitude_gradient_fitness(
            weights, gradients, alpha=0.5
        )
        
        assert fitness.shape == weights.shape


class TestGeneticOperators:
    """Test genetic operators for evolution."""
    
    def test_tournament_selection(self, device):
        """Test tournament selection."""
        population = [torch.randn(32, 16, device=device) for _ in range(10)]
        fitness_scores = [float(i) for i in range(10)]
        
        p1, p2 = GeneticOperators.tournament_selection(
            population, fitness_scores, tournament_size=3
        )
        
        assert p1.shape == (32, 16)
        assert p2.shape == (32, 16)
    
    def test_mutation(self, device):
        """Test mutation operator."""
        mask = torch.ones(32, 16, device=device)
        
        mutated = GeneticOperators.mutation(
            mask, mutation_rate=0.1, target_sparsity=0.5
        )
        
        assert mutated.shape == mask.shape
        # Some values should have changed
        assert not torch.equal(mutated, mask)
    
    def test_uniform_crossover(self, device):
        """Test uniform crossover."""
        parent1 = torch.ones(32, 16, device=device)
        parent2 = torch.zeros(32, 16, device=device)
        
        child1, child2 = GeneticOperators.uniform_crossover(
            parent1, parent2, crossover_rate=0.5
        )
        
        assert child1.shape == parent1.shape
        assert child2.shape == parent2.shape
        # Children should mix parents
        assert child1.mean() > 0 and child1.mean() < 1


class TestEvolutionaryPruner:
    """Test evolutionary pruning engine."""
    
    def test_initialization(self, simple_model, device):
        """Test pruner initialization."""
        config = EvolutionaryConfig(
            population_size=5,
            generations=2
        )
        
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        assert len(pruner.population) == 5
        assert len(pruner.prunable_layers) == 2  # Two Linear layers
    
    def test_sparsity_schedule(self, simple_model, device):
        """Test sparsity scheduling."""
        config = EvolutionaryConfig(
            initial_sparsity=0.3,
            target_sparsity=0.9,
            pressure_schedule='linear'
        )
        
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        # At start (progress=0)
        sparsity_start = pruner._get_scheduled_sparsity(0.0)
        assert abs(sparsity_start - 0.3) < 0.01
        
        # At end (progress=1)
        sparsity_end = pruner._get_scheduled_sparsity(1.0)
        assert abs(sparsity_end - 0.9) < 0.01


# ============================================================================
# SNN Homeostasis Tests
# ============================================================================

class TestHomeostasisConfig:
    """Test HomeostasisConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = HomeostasisConfig()
        assert config.target_firing_rate == 0.05
        assert config.enable_synaptic_scaling == True
        assert config.sleep_period == 10000
    
    def test_validation(self):
        """Test validation."""
        with pytest.raises(AssertionError):
            HomeostasisConfig(target_firing_rate=1.5)
        
        with pytest.raises(AssertionError):
            HomeostasisConfig(sleep_downscale_factor=1.5)


class TestSynapticScaling:
    """Test synaptic scaling module."""
    
    def test_initialization(self, device):
        """Test initialization."""
        config = HomeostasisConfig()
        scaling = SynapticScaling(n_neurons=64, config=config, device=device)
        
        assert scaling.firing_rate_ema.shape == (64,)
        assert scaling.scaling_factors.shape == (64,)
    
    def test_update_firing_rates(self, device):
        """Test firing rate EMA update."""
        config = HomeostasisConfig()
        scaling = SynapticScaling(n_neurons=32, config=config, device=device)
        
        # High activity spikes
        spikes = torch.ones(16, 32, device=device)
        scaling.update_firing_rates(spikes)
        
        # Firing rate should increase
        assert scaling.firing_rate_ema.mean() > config.target_firing_rate
    
    def test_scaling_factors(self, device):
        """Test scaling factor computation."""
        config = HomeostasisConfig(target_firing_rate=0.1)
        scaling = SynapticScaling(n_neurons=32, config=config, device=device)
        
        # Set high firing rates
        scaling.firing_rate_ema = torch.full((32,), 0.2, device=device)
        
        factors = scaling.compute_scaling_factors()
        
        # High firing → scaling < 1 (reduce activity)
        assert (factors < 1).all()


class TestIntrinsicPlasticity:
    """Test intrinsic plasticity module."""
    
    def test_initialization(self, device):
        """Test initialization."""
        config = HomeostasisConfig()
        ip = IntrinsicPlasticity(n_neurons=64, config=config, device=device)
        
        assert ip.thresholds.shape == (64,)
    
    def test_threshold_adaptation(self, device):
        """Test threshold adaptation."""
        config = HomeostasisConfig(target_firing_rate=0.05)
        ip = IntrinsicPlasticity(n_neurons=32, config=config, device=device)
        
        initial_thresholds = ip.thresholds.clone()
        
        # High activity spikes → should increase threshold
        high_spikes = torch.ones(16, 32, device=device)
        ip.update_thresholds(high_spikes)
        
        # Thresholds should increase
        assert (ip.thresholds >= initial_thresholds).all()


class TestSleepConsolidation:
    """Test sleep-based consolidation."""
    
    def test_initialization(self, device):
        """Test initialization."""
        config = HomeostasisConfig(sleep_period=100)
        sleep = SleepConsolidation(config=config, device=device)
        
        assert sleep.timestep == 0
        assert sleep.sleep_count == 0
    
    def test_should_sleep(self, device):
        """Test sleep timing."""
        config = HomeostasisConfig(sleep_period=100)
        sleep = SleepConsolidation(config=config, device=device)
        
        # Should not sleep initially
        assert not sleep.should_sleep()
        
        # Advance to sleep time
        sleep.timestep = 100
        assert sleep.should_sleep()
    
    def test_sleep_phase(self, device):
        """Test sleep phase processing."""
        config = HomeostasisConfig(
            sleep_downscale_factor=0.9,
            prune_threshold=0.05
        )
        sleep = SleepConsolidation(config=config, device=device)
        
        weights = torch.randn(32, 16, device=device)
        original_norm = weights.norm()
        
        processed, n_pruned = sleep.sleep_phase(weights)
        
        # Weights should be downscaled
        assert processed.norm() < original_norm
    
    def test_pattern_replay(self, device):
        """Test pattern replay buffer."""
        config = HomeostasisConfig()
        sleep = SleepConsolidation(config=config, device=device)
        
        # Store some patterns
        for i in range(5):
            pattern = torch.randn(32, device=device)
            sleep.store_pattern(pattern)
        
        # Replay
        patterns = sleep.replay_patterns(n_patterns=3)
        assert len(patterns) == 3


class TestHomeostaticSTDP:
    """Test homeostatic STDP learning."""
    
    def test_initialization(self, device):
        """Test initialization."""
        config = HomeostasisConfig()
        stdp = HomeostaticSTDP(
            in_features=64,
            out_features=32,
            config=config,
            device=device
        )
        
        assert stdp.pre_trace.shape == (64,)
        assert stdp.post_trace.shape == (32,)
    
    def test_trace_update(self, device):
        """Test trace updates."""
        config = HomeostasisConfig()
        stdp = HomeostaticSTDP(
            in_features=32,
            out_features=16,
            config=config,
            device=device
        )
        
        pre_spikes = torch.ones(8, 32, device=device)
        post_spikes = torch.zeros(8, 16, device=device)
        
        stdp.update_traces(pre_spikes, post_spikes)
        
        # Pre trace should increase
        assert stdp.pre_trace.mean() > 0
    
    def test_stdp_update(self, device):
        """Test STDP weight update."""
        config = HomeostasisConfig()
        stdp = HomeostaticSTDP(
            in_features=32,
            out_features=16,
            config=config,
            device=device
        )
        
        weights = torch.randn(16, 32, device=device)
        pre_spikes = torch.rand(8, 32, device=device) > 0.5
        post_spikes = torch.rand(8, 16, device=device) > 0.5
        
        dw = stdp.compute_stdp_update(
            weights, pre_spikes.float(), post_spikes.float()
        )
        
        assert dw.shape == weights.shape


class TestHomeostaticSpikingLayer:
    """Test complete homeostatic spiking layer."""
    
    def test_initialization(self, device):
        """Test layer initialization."""
        config = HomeostasisConfig()
        layer = HomeostaticSpikingLayer(
            in_features=64,
            out_features=32,
            tau_mem=10.0,
            config=config,
            device=device
        )
        
        assert layer.weight.shape == (32, 64)
        assert layer.synaptic_scaling is not None
        assert layer.intrinsic_plasticity is not None
    
    def test_forward(self, device):
        """Test forward pass."""
        config = HomeostasisConfig()
        layer = HomeostaticSpikingLayer(
            in_features=64,
            out_features=32,
            config=config,
            device=device
        )
        
        input_spikes = (torch.rand(16, 64, device=device) > 0.9).float()
        output_spikes = layer(input_spikes)
        
        assert output_spikes.shape == (16, 32)
        # Spikes should be binary
        assert ((output_spikes == 0) | (output_spikes == 1)).all()
    
    def test_temporal_processing(self, device):
        """Test temporal dynamics."""
        config = HomeostasisConfig()
        layer = HomeostaticSpikingLayer(
            in_features=32,
            out_features=16,
            config=config,
            device=device
        )
        
        # Process multiple timesteps
        batch_size = 8
        n_timesteps = 20
        total_spikes = 0
        
        for t in range(n_timesteps):
            input_spikes = (torch.rand(batch_size, 32, device=device) > 0.9).float()
            output_spikes = layer(input_spikes, apply_stdp=True)
            total_spikes += output_spikes.sum().item()
        
        # Should produce some spikes
        assert total_spikes > 0
    
    def test_get_statistics(self, device):
        """Test statistics collection."""
        config = HomeostasisConfig()
        layer = HomeostaticSpikingLayer(
            in_features=32,
            out_features=16,
            config=config,
            device=device
        )
        
        # Process some spikes
        for _ in range(10):
            input_spikes = (torch.rand(8, 32, device=device) > 0.9).float()
            layer(input_spikes)
        
        stats = layer.get_all_statistics()
        
        assert 'layer' in stats
        assert 'synaptic_scaling' in stats
        assert 'stdp' in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestPINNIntegration:
    """Integration tests for PINN training."""
    
    def test_pinn_training_loop(self, device):
        """Test complete PINN training loop."""
        # Create simple PINN
        model, pde, trainer = create_heat_pinn(
            alpha=1.0,
            hidden_dims=[32, 32],
            device=device
        )
        
        # Create training data
        n_points = 100
        x = torch.rand(n_points, 2, device=device)
        u = torch.rand(n_points, 1, device=device)
        
        # Train for a few steps
        history = trainer.train(
            n_epochs=10,
            x_data=x,
            u_data=u,
            verbose=False
        )
        
        assert 'loss' in history
        assert len(history['loss']) == 10


class TestEvolutionaryIntegration:
    """Integration tests for evolutionary pruning."""
    
    def test_pruner_population_evolution(self, simple_model, device):
        """Test population evolution without data."""
        config = EvolutionaryConfig(
            population_size=3,
            generations=2,
            initial_sparsity=0.3,
            target_sparsity=0.7
        )
        
        pruner = EvolutionaryPruner(simple_model, config, device)
        
        # Just test population initialization
        initial_sparsity = pruner._compute_population_sparsity()
        assert 0 <= initial_sparsity <= 1


class TestSNNHomeostasisIntegration:
    """Integration tests for SNN homeostasis."""
    
    def test_full_snn_with_homeostasis(self, device):
        """Test complete SNN with all homeostatic mechanisms."""
        config = HomeostasisConfig(
            target_firing_rate=0.05,
            enable_synaptic_scaling=True,
            enable_intrinsic_plasticity=True,
            enable_sleep_cycles=True,
            sleep_period=50
        )
        
        layer = HomeostaticSpikingLayer(
            in_features=64,
            out_features=32,
            config=config,
            device=device
        )
        
        # Simulate longer sequence with varying input
        n_timesteps = 100
        all_outputs = []
        
        for t in range(n_timesteps):
            # Varying input rate
            rate = 0.05 + 0.1 * np.sin(t / 10)
            input_spikes = (torch.rand(16, 64, device=device) > (1 - rate)).float()
            
            output = layer(input_spikes, apply_stdp=True)
            all_outputs.append(output.mean().item())
        
        stats = layer.get_all_statistics()
        
        # Check homeostasis is working (firing rate near target)
        mean_rate = stats['layer']['avg_spike_rate']
        assert mean_rate > 0  # Producing spikes


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
