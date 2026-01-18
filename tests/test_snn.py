"""
Comprehensive tests for Spiking Neural Networks (SNN) module.

Test Coverage:
-------------
1. LIF Neuron dynamics and state management
2. Spiking Layer forward/backward propagation
3. STDP learning rules and weight updates
4. Spike encoding methods (rate, temporal)
5. Spike decoding methods
6. Integration tests for complete SNN workflows
7. Edge cases and error handling

Session 13 - Test Suite
Author: Legacy GPU AI Platform Team
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.compute.snn import (
    LIFNeuron,
    LIFParams,
    SpikingLayer,
    STDPLearning,
    STDPParams,
    RateEncoder,
    TemporalEncoder,
    SpikeDecoder,
    spike_function,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def lif_params():
    """Return standard LIF parameters."""
    return LIFParams(
        tau_mem=10.0,
        v_thresh=1.0,
        v_reset=0.0,
        v_rest=0.0,
        refractory_period=2,
        dt=1.0
    )


@pytest.fixture
def stdp_params():
    """Return standard STDP parameters."""
    return STDPParams(
        a_plus=0.01,
        a_minus=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0
    )


# ============================================================================
# Test LIFParams
# ============================================================================

class TestLIFParams:
    """Test LIF parameter validation."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = LIFParams()
        assert params.tau_mem == 10.0
        assert params.v_thresh == 1.0
        assert params.v_reset == 0.0
        assert params.v_rest == 0.0
        assert params.refractory_period == 2
        assert params.dt == 1.0
    
    def test_custom_params(self):
        """Test custom parameter initialization."""
        params = LIFParams(tau_mem=15.0, v_thresh=0.8)
        assert params.tau_mem == 15.0
        assert params.v_thresh == 0.8
    
    def test_invalid_tau_mem(self):
        """Test that negative tau_mem raises error."""
        with pytest.raises(AssertionError):
            LIFParams(tau_mem=-5.0)
    
    def test_invalid_threshold(self):
        """Test that v_thresh <= v_reset raises error."""
        with pytest.raises(AssertionError):
            LIFParams(v_thresh=0.5, v_reset=1.0)
    
    def test_invalid_refractory(self):
        """Test that negative refractory period raises error."""
        with pytest.raises(AssertionError):
            LIFParams(refractory_period=-1)


# ============================================================================
# Test LIFNeuron
# ============================================================================

class TestLIFNeuron:
    """Test LIF neuron dynamics."""
    
    def test_initialization(self, lif_params, device):
        """Test neuron initialization."""
        neuron = LIFNeuron(n_neurons=128, params=lif_params, device=device)
        assert neuron.n_neurons == 128
        assert neuron.params == lif_params
        assert neuron.device == device
    
    def test_state_reset(self, device):
        """Test state reset functionality."""
        neuron = LIFNeuron(n_neurons=10, device=device)
        neuron.reset_state(batch_size=4)
        
        assert neuron.v_mem.shape == (4, 10)
        assert neuron.refractory_count.shape == (4, 10)
        assert torch.allclose(neuron.v_mem, torch.zeros_like(neuron.v_mem))
    
    def test_forward_shape(self, device):
        """Test forward pass output shape."""
        neuron = LIFNeuron(n_neurons=64, device=device)
        neuron.reset_state(batch_size=8)
        
        input_current = torch.rand(8, 64, device=device)
        spikes = neuron(input_current)
        
        assert spikes.shape == (8, 64)
        assert spikes.dtype == torch.float32
        assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes
    
    def test_spike_generation(self, device):
        """Test that strong input generates spikes."""
        neuron = LIFNeuron(n_neurons=10, device=device)
        neuron.reset_state(batch_size=1)
        
        # Strong constant input should cause spikes
        strong_input = torch.ones(1, 10, device=device) * 0.5
        
        total_spikes = 0
        for _ in range(50):
            spikes = neuron(strong_input)
            total_spikes += spikes.sum().item()
        
        # Should have generated some spikes
        assert total_spikes > 0
    
    def test_no_spikes_weak_input(self, device):
        """Test that weak input doesn't generate spikes."""
        neuron = LIFNeuron(n_neurons=10, device=device)
        neuron.reset_state(batch_size=1)
        
        # Very weak input should not cause spikes
        weak_input = torch.ones(1, 10, device=device) * 0.01
        
        total_spikes = 0
        for _ in range(20):
            spikes = neuron(weak_input)
            total_spikes += spikes.sum().item()
        
        assert total_spikes == 0
    
    def test_membrane_potential_decay(self, device):
        """Test that membrane potential decays without input."""
        params = LIFParams(tau_mem=10.0, v_thresh=10.0)  # High threshold
        neuron = LIFNeuron(n_neurons=1, params=params, device=device)
        neuron.reset_state(batch_size=1)
        
        # Inject current to build up potential
        neuron.v_mem = torch.tensor([[0.8]], device=device)
        v_initial = neuron.v_mem.clone()
        
        # No input - should decay
        zero_input = torch.zeros(1, 1, device=device)
        neuron(zero_input)
        
        assert neuron.v_mem[0, 0] < v_initial[0, 0]
    
    def test_reset_after_spike(self, device):
        """Test that membrane resets after spike."""
        params = LIFParams(v_thresh=0.5, v_reset=0.0)
        neuron = LIFNeuron(n_neurons=1, params=params, device=device)
        neuron.reset_state(batch_size=1)
        
        # Input that will cause spike
        neuron(torch.tensor([[0.6]], device=device))
        
        # Membrane should be at reset potential
        assert neuron.v_mem[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    
    def test_refractory_period(self, device):
        """Test refractory period prevents immediate re-spiking."""
        params = LIFParams(v_thresh=0.5, refractory_period=3)
        neuron = LIFNeuron(n_neurons=1, params=params, device=device)
        neuron.reset_state(batch_size=1)
        
        # Cause spike
        spikes1 = neuron(torch.tensor([[1.0]], device=device))
        assert spikes1[0, 0] == 1.0
        
        # Should not spike again immediately despite strong input
        spikes2 = neuron(torch.tensor([[1.0]], device=device))
        assert spikes2[0, 0] == 0.0
    
    def test_statistics_tracking(self, device):
        """Test spike statistics tracking."""
        neuron = LIFNeuron(n_neurons=10, device=device)
        neuron.reset_state(batch_size=2)
        
        input_current = torch.ones(2, 10, device=device) * 0.2
        
        for _ in range(50):
            neuron(input_current)
        
        stats = neuron.get_statistics()
        assert 'spike_rate' in stats
        assert 'spike_count' in stats
        assert 'total_updates' in stats
        assert stats['total_updates'] == 50 * 2 * 10
    
    def test_get_state(self, device):
        """Test state retrieval."""
        neuron = LIFNeuron(n_neurons=5, device=device)
        neuron.reset_state(batch_size=2)
        
        state = neuron.get_state()
        assert 'v_mem' in state
        assert 'refractory_count' in state
        assert state['v_mem'].shape == (2, 5)


# ============================================================================
# Test SpikingLayer
# ============================================================================

class TestSpikingLayer:
    """Test spiking layer functionality."""
    
    def test_initialization(self, lif_params, device):
        """Test layer initialization."""
        layer = SpikingLayer(
            in_features=784,
            out_features=128,
            params=lif_params,
            device=device
        )
        
        assert layer.in_features == 784
        assert layer.out_features == 128
        assert layer.weight.shape == (128, 784)
        assert layer.bias.shape == (128,)
    
    def test_no_bias(self, device):
        """Test layer without bias."""
        layer = SpikingLayer(100, 50, bias=False, device=device)
        assert layer.bias is None
    
    def test_forward_shape(self, device):
        """Test forward pass shape."""
        layer = SpikingLayer(20, 10, device=device)
        layer.reset_state(batch_size=4)
        
        spike_input = torch.rand(4, 20, device=device) > 0.8
        spike_output = layer(spike_input.float())
        
        assert spike_output.shape == (4, 10)
        assert torch.all((spike_output == 0) | (spike_output == 1))
    
    def test_temporal_processing(self, device):
        """Test processing spike train over time."""
        layer = SpikingLayer(10, 5, device=device)
        layer.reset_state(batch_size=2)
        
        t_steps = 50
        spike_outputs = []
        
        for t in range(t_steps):
            spike_input = torch.rand(2, 10, device=device) > 0.9
            spike_output = layer(spike_input.float())
            spike_outputs.append(spike_output)
        
        assert len(spike_outputs) == t_steps
        
        # Check that we got some output spikes
        total_output_spikes = sum(s.sum().item() for s in spike_outputs)
        assert total_output_spikes > 0
    
    def test_gradient_flow(self, device):
        """Test that gradients flow through layer."""
        layer = SpikingLayer(10, 5, device=device)
        layer.reset_state(batch_size=2)
        
        # Use continuous input that requires grad
        continuous_input = torch.rand(2, 10, device=device, requires_grad=True)
        
        # Forward pass
        spike_output = layer(continuous_input)
        
        # Create loss from membrane potential (continuous) for gradient test
        # In actual training, surrogate gradients would be used
        loss = layer.neuron.v_mem.sum()
        loss.backward()
        
        # Check gradients exist (through membrane dynamics, not spikes)
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_state_reset_between_sequences(self, device):
        """Test state reset between different sequences."""
        layer = SpikingLayer(10, 5, device=device)
        
        # Process first sequence
        layer.reset_state(batch_size=2)
        for _ in range(20):
            spike_input = torch.rand(2, 10, device=device) > 0.9
            layer(spike_input.float())
        
        stats1 = layer.get_statistics()
        
        # Reset and process second sequence
        layer.reset_state(batch_size=2)
        stats2 = layer.get_statistics()
        
        # Statistics should be reset
        assert stats2['spike_count'] == 0
        assert stats2['total_updates'] == 0


# ============================================================================
# Test STDP Learning
# ============================================================================

class TestSTDPLearning:
    """Test STDP learning mechanism."""
    
    def test_initialization(self, stdp_params, device):
        """Test STDP initialization."""
        stdp = STDPLearning(
            weight_shape=(64, 128),
            params=stdp_params,
            device=device
        )
        
        assert stdp.weight_shape == (64, 128)
        assert stdp.trace_pre.shape == (128,)
        assert stdp.trace_post.shape == (64,)
    
    def test_trace_decay(self, device):
        """Test trace decay over time."""
        stdp = STDPLearning(weight_shape=(10, 20), device=device)
        
        # Set initial traces
        stdp.trace_pre = torch.ones(20, device=device)
        stdp.trace_post = torch.ones(10, device=device)
        
        # No spikes - traces should decay
        weights = torch.rand(10, 20, device=device)
        spikes_pre = torch.zeros(2, 20, device=device)
        spikes_post = torch.zeros(2, 10, device=device)
        
        stdp.update(spikes_pre, spikes_post, weights)
        
        # Traces should have decayed
        assert torch.all(stdp.trace_pre < 1.0)
        assert torch.all(stdp.trace_post < 1.0)
    
    def test_weight_potentiation(self, device):
        """Test LTP: pre-spike before post-spike strengthens weight."""
        params = STDPParams(a_plus=0.1, a_minus=0.0)  # Only potentiation
        stdp = STDPLearning(weight_shape=(5, 5), params=params, device=device)
        
        weights = torch.ones(5, 5, device=device) * 0.5
        
        # Pre-spike first
        spikes_pre = torch.zeros(1, 5, device=device)
        spikes_pre[0, 0] = 1.0
        spikes_post = torch.zeros(1, 5, device=device)
        stdp.update(spikes_pre, spikes_post, weights)
        
        # Post-spike second (with pre-trace present)
        spikes_pre = torch.zeros(1, 5, device=device)
        spikes_post = torch.zeros(1, 5, device=device)
        spikes_post[0, 0] = 1.0
        new_weights = stdp.update(spikes_pre, spikes_post, weights)
        
        # Weight [0, 0] should increase
        assert new_weights[0, 0] > weights[0, 0]
    
    def test_weight_depression(self, device):
        """Test LTD: post-spike before pre-spike weakens weight."""
        params = STDPParams(a_plus=0.0, a_minus=0.1)  # Only depression
        stdp = STDPLearning(weight_shape=(5, 5), params=params, device=device)
        
        weights = torch.ones(5, 5, device=device) * 0.5
        
        # Post-spike first
        spikes_pre = torch.zeros(1, 5, device=device)
        spikes_post = torch.zeros(1, 5, device=device)
        spikes_post[0, 0] = 1.0
        stdp.update(spikes_pre, spikes_post, weights)
        
        # Pre-spike second (with post-trace present)
        spikes_pre = torch.zeros(1, 5, device=device)
        spikes_pre[0, 0] = 1.0
        spikes_post = torch.zeros(1, 5, device=device)
        new_weights = stdp.update(spikes_pre, spikes_post, weights)
        
        # Weight [0, 0] should decrease
        assert new_weights[0, 0] < weights[0, 0]
    
    def test_weight_bounds(self, device):
        """Test that weights stay within [w_min, w_max]."""
        params = STDPParams(a_plus=1.0, w_min=0.0, w_max=1.0)
        stdp = STDPLearning(weight_shape=(5, 5), params=params, device=device)
        
        # Start with weights near max
        weights = torch.ones(5, 5, device=device) * 0.99
        
        # Large potentiation
        spikes_pre = torch.ones(1, 5, device=device)
        spikes_post = torch.ones(1, 5, device=device)
        
        for _ in range(10):
            weights = stdp.update(spikes_pre, spikes_post, weights)
        
        # Weights should be clamped to w_max
        assert torch.all(weights <= params.w_max)
        assert torch.all(weights >= params.w_min)


# ============================================================================
# Test Encoding
# ============================================================================

class TestRateEncoder:
    """Test rate encoding."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = RateEncoder(t_steps=100, max_rate=200, method='poisson')
        assert encoder.t_steps == 100
        assert encoder.max_rate == 200
        assert encoder.method == 'poisson'
    
    def test_poisson_encoding_shape(self):
        """Test Poisson encoding output shape."""
        encoder = RateEncoder(t_steps=50, method='poisson')
        input_data = torch.rand(10)
        
        spike_train = encoder.encode(input_data)
        assert spike_train.shape == (50, 10)
    
    def test_poisson_encoding_rate(self):
        """Test that spike rate matches input value."""
        encoder = RateEncoder(t_steps=1000, max_rate=100, method='poisson')
        input_data = torch.tensor([0.2, 0.5, 0.8])
        
        spike_train = encoder.encode(input_data)
        spike_rates = spike_train.mean(dim=0)
        
        # Rates should roughly match input (with Poisson noise)
        expected_rates = input_data * 100 * 1.0 / 1000  # rate * dt / 1000
        assert torch.allclose(spike_rates, expected_rates, atol=0.05)
    
    def test_constant_encoding(self):
        """Test constant rate encoding."""
        encoder = RateEncoder(t_steps=100, max_rate=100, method='constant')
        input_data = torch.tensor([0.5])
        
        spike_train = encoder.encode(input_data)
        
        # Should have regular spikes
        assert spike_train.sum() > 0
    
    def test_batch_encoding(self):
        """Test encoding with batch dimension."""
        encoder = RateEncoder(t_steps=50, method='poisson')
        input_data = torch.rand(4, 10)  # batch=4, features=10
        
        spike_train = encoder.encode(input_data)
        assert spike_train.shape == (50, 4, 10)


class TestTemporalEncoder:
    """Test temporal encoding."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = TemporalEncoder(t_steps=100)
        assert encoder.t_steps == 100
        assert encoder.t_max == 100
    
    def test_encoding_shape(self):
        """Test encoding output shape."""
        encoder = TemporalEncoder(t_steps=100)
        input_data = torch.rand(10)
        
        spike_train = encoder.encode(input_data)
        assert spike_train.shape == (100, 10)
    
    def test_latency_ordering(self):
        """Test that higher values spike earlier."""
        encoder = TemporalEncoder(t_steps=100)
        input_data = torch.tensor([0.2, 0.5, 0.9])
        
        spike_train = encoder.encode(input_data)
        
        # Find spike times
        spike_times = []
        for i in range(3):
            spikes = torch.where(spike_train[:, i] > 0)[0]
            if len(spikes) > 0:
                spike_times.append(spikes[0].item())
            else:
                spike_times.append(float('inf'))
        
        # Higher input → earlier spike → lower time
        assert spike_times[2] < spike_times[1] < spike_times[0]
    
    def test_zero_input_no_spike(self):
        """Test that zero input produces no spike."""
        encoder = TemporalEncoder(t_steps=100)
        input_data = torch.tensor([0.0])
        
        spike_train = encoder.encode(input_data)
        assert spike_train.sum() == 0


# ============================================================================
# Test Decoding
# ============================================================================

class TestSpikeDecoder:
    """Test spike decoding."""
    
    def test_rate_decoding(self):
        """Test rate decoding."""
        decoder = SpikeDecoder(method='rate')
        
        # Create spike train with known rate
        spike_train = torch.zeros(100, 3)
        spike_train[:20, 0] = 1.0  # 20% rate
        spike_train[:50, 1] = 1.0  # 50% rate
        spike_train[:80, 2] = 1.0  # 80% rate
        
        decoded = decoder.decode(spike_train)
        
        assert decoded[0] == pytest.approx(0.2, abs=0.01)
        assert decoded[1] == pytest.approx(0.5, abs=0.01)
        assert decoded[2] == pytest.approx(0.8, abs=0.01)
    
    def test_temporal_decoding(self):
        """Test temporal decoding."""
        decoder = SpikeDecoder(method='temporal', t_max=100)
        
        # Create spike train with known timing
        spike_train = torch.zeros(100, 3)
        spike_train[10, 0] = 1.0   # Early spike → high value
        spike_train[50, 1] = 1.0   # Mid spike → mid value
        spike_train[90, 2] = 1.0   # Late spike → low value
        
        decoded = decoder.decode(spike_train)
        
        # Early spike → higher value
        assert decoded[0] > decoded[1] > decoded[2]
    
    def test_encode_decode_consistency(self):
        """Test that encoding + decoding recovers original values."""
        encoder = RateEncoder(t_steps=1000, max_rate=200, method='poisson')
        decoder = SpikeDecoder(method='rate')
        
        input_data = torch.tensor([0.2, 0.5, 0.8])
        
        spike_train = encoder.encode(input_data)
        decoded = decoder.decode(spike_train)
        
        # Normalize decoded to match encoding range
        # max_rate * dt / 1000 = spike probability per timestep
        expected_spike_rate = input_data * 200 * 1.0 / 1000.0
        
        # Should approximately recover original spike rate (with Poisson noise)
        error = torch.abs(expected_spike_rate - decoded).mean()
        assert error < 0.02  # Within 2% error for spike rate


# ============================================================================
# Integration Tests
# ============================================================================

class TestSNNIntegration:
    """Test complete SNN workflows."""
    
    def test_simple_snn_network(self, device):
        """Test simple two-layer SNN."""
        # Create two-layer network
        layer1 = SpikingLayer(20, 10, device=device)
        layer2 = SpikingLayer(10, 5, device=device)
        
        layer1.reset_state(batch_size=2)
        layer2.reset_state(batch_size=2)
        
        # Process spike train
        for t in range(50):
            spike_input = torch.rand(2, 20, device=device) > 0.9
            hidden_spikes = layer1(spike_input.float())
            output_spikes = layer2(hidden_spikes)
        
        # Should complete without errors
        assert output_spikes.shape == (2, 5)
    
    def test_rate_encoding_to_snn(self, device):
        """Test rate encoding → SNN → decoding pipeline."""
        # Encode input
        encoder = RateEncoder(t_steps=100, max_rate=100, method='poisson')
        input_data = torch.tensor([[0.5, 0.8]], device=device)
        spike_train = encoder.encode(input_data)
        
        # Process through SNN
        layer = SpikingLayer(2, 3, device=device)
        layer.reset_state(batch_size=1)
        
        output_spikes = []
        for t in range(100):
            out = layer(spike_train[t])
            output_spikes.append(out)
        
        output_train = torch.stack(output_spikes, dim=0)
        
        # Decode output
        decoder = SpikeDecoder(method='rate')
        output_data = decoder.decode(output_train)
        
        assert output_data.shape == (1, 3)
        assert output_data.sum() > 0  # Should have some activity
    
    def test_stdp_learning_on_layer(self, device):
        """Test STDP learning on spiking layer."""
        layer = SpikingLayer(10, 5, device=device)
        layer.reset_state(batch_size=1)
        
        stdp = STDPLearning(weight_shape=(5, 10), device=device)
        
        weights_before = layer.weight.data.clone()
        
        # Run several timesteps with STDP
        for t in range(50):
            spike_input = torch.rand(1, 10, device=device) > 0.9
            spike_output = layer(spike_input.float())
            
            # Update weights with STDP
            layer.weight.data = stdp.update(
                spike_input.float(),
                spike_output,
                layer.weight.data
            )
        
        # Weights should have changed
        weight_change = (layer.weight.data - weights_before).abs().sum()
        assert weight_change > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestSNNPerformance:
    """Test SNN performance characteristics."""
    
    def test_event_sparsity(self, device):
        """Test that spike activity is sparse."""
        neuron = LIFNeuron(n_neurons=256, device=device)
        neuron.reset_state(batch_size=16)
        
        # Moderate input
        input_current = torch.rand(16, 256, device=device) * 0.3
        
        total_spikes = 0
        total_elements = 0
        
        for _ in range(100):
            spikes = neuron(input_current)
            total_spikes += spikes.sum().item()
            total_elements += spikes.numel()
        
        sparsity = 1.0 - (total_spikes / total_elements)
        
        # Should be >90% sparse (biologically realistic)
        assert sparsity > 0.9
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_acceleration(self):
        """Test that GPU provides speedup."""
        import time
        
        n_neurons = 1024
        t_steps = 100
        
        # CPU
        neuron_cpu = LIFNeuron(n_neurons=n_neurons, device='cpu')
        neuron_cpu.reset_state(batch_size=32)
        input_cpu = torch.rand(32, n_neurons)
        
        start = time.time()
        for _ in range(t_steps):
            neuron_cpu(input_cpu)
        cpu_time = time.time() - start
        
        # GPU
        neuron_gpu = LIFNeuron(n_neurons=n_neurons, device='cuda')
        neuron_gpu.reset_state(batch_size=32)
        input_gpu = torch.rand(32, n_neurons, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(t_steps):
            neuron_gpu(input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # GPU should be faster (at least 2x)
        assert gpu_time < cpu_time / 2


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
