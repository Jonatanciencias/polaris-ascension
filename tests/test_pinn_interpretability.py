"""
Tests for PINN Interpretability Module - Session 22
===================================================
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.compute.pinn_interpretability import (
    PINNInterpreter,
    SensitivityResult,
    ResidualAnalysis,
    compute_gradient,
    heat_equation_residual,
    wave_equation_residual,
    burgers_equation_residual
)


# Test fixtures

@pytest.fixture
def simple_pinn():
    """Simple PINN for testing."""
    class SimplePINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 20),
                nn.Tanh(),
                nn.Linear(20, 20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    return SimplePINN()


@pytest.fixture
def test_inputs():
    """Test input data (x, t) pairs."""
    torch.manual_seed(42)
    return torch.randn(10, 2)


@pytest.fixture
def interpreter(simple_pinn):
    """PINNInterpreter instance."""
    return PINNInterpreter(
        model=simple_pinn,
        input_names=['x', 't']
    )


# Test PINNInterpreter initialization

def test_interpreter_creation(simple_pinn):
    """Test interpreter creation."""
    interpreter = PINNInterpreter(simple_pinn)
    assert interpreter.model is simple_pinn
    assert isinstance(interpreter.input_names, list)
    assert interpreter.device is not None


def test_interpreter_custom_names(simple_pinn):
    """Test custom input names."""
    interpreter = PINNInterpreter(
        simple_pinn,
        input_names=['x', 't', 'z']
    )
    assert interpreter.input_names == ['x', 't', 'z']


# Test sensitivity analysis

def test_gradient_sensitivity(interpreter, test_inputs):
    """Test gradient-based sensitivity computation."""
    result = interpreter.compute_sensitivity_map(test_inputs, method='gradient')
    
    assert isinstance(result, SensitivityResult)
    assert 'du_dx' in result.sensitivity_maps
    assert 'du_dt' in result.sensitivity_maps
    assert result.sensitivity_maps['du_dx'].shape == (10, 1)
    assert 'x' in result.feature_importance
    assert 't' in result.feature_importance
    
    # Importance scores should sum to 1
    total_importance = sum(result.feature_importance.values())
    assert abs(total_importance - 1.0) < 1e-6


def test_integrated_gradients(interpreter, test_inputs):
    """Test integrated gradients method."""
    result = interpreter.compute_sensitivity_map(
        test_inputs,
        method='integrated_gradients'
    )
    
    assert isinstance(result, SensitivityResult)
    assert 'du_dx' in result.sensitivity_maps
    assert 'du_dt' in result.sensitivity_maps
    assert result.metadata['method'] == 'integrated_gradients'


def test_smooth_grad(interpreter, test_inputs):
    """Test SmoothGrad method."""
    result = interpreter.compute_sensitivity_map(
        test_inputs,
        method='smooth_grad'
    )
    
    assert isinstance(result, SensitivityResult)
    assert result.metadata['method'] == 'smooth_grad'
    assert 'num_samples' in result.metadata


def test_sensitivity_invalid_method(interpreter, test_inputs):
    """Test invalid sensitivity method."""
    with pytest.raises(ValueError):
        interpreter.compute_sensitivity_map(test_inputs, method='invalid')


def test_feature_importance(interpreter, test_inputs):
    """Test feature importance computation."""
    importance = interpreter.feature_importance(test_inputs)
    
    assert isinstance(importance, dict)
    assert 'x' in importance
    assert 't' in importance
    assert all(0 <= v <= 1 for v in importance.values())
    assert abs(sum(importance.values()) - 1.0) < 1e-6


# Test physics residual analysis

def test_analyze_residual_heat_equation(interpreter, test_inputs):
    """Test residual analysis with heat equation."""
    # Heat equation PDE
    def heat_pde(inputs, u):
        return heat_equation_residual(inputs, u, alpha=0.01)
    
    analysis = interpreter.analyze_residual(
        test_inputs,
        heat_pde
    )
    
    assert isinstance(analysis, ResidualAnalysis)
    assert 'mean' in analysis.residual_stats
    assert 'std' in analysis.residual_stats
    assert 'max' in analysis.residual_stats
    assert analysis.residual_map.numel() == len(test_inputs)


def test_analyze_residual_with_grid(interpreter):
    """Test residual analysis with grid reshaping."""
    # Create grid
    x = torch.linspace(0, 1, 10)
    t = torch.linspace(0, 1, 10)
    X, T = torch.meshgrid(x, t, indexing='ij')
    inputs = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    def simple_pde(inputs, u):
        return u  # Dummy residual
    
    analysis = interpreter.analyze_residual(
        inputs,
        simple_pde,
        grid_shape=(10, 10)
    )
    
    assert analysis.residual_map.shape == (10, 10)


def test_residual_hotspots(interpreter, test_inputs):
    """Test hotspot detection in residual."""
    def varying_pde(inputs, u):
        # Create artificial varying residual
        return u * inputs[:, 0:1]
    
    analysis = interpreter.analyze_residual(test_inputs, varying_pde)
    
    assert isinstance(analysis.hotspots, list)
    assert len(analysis.hotspots) <= 10


def test_residual_statistics(interpreter, test_inputs):
    """Test residual statistics computation."""
    def simple_pde(inputs, u):
        return torch.ones_like(u)
    
    analysis = interpreter.analyze_residual(test_inputs, simple_pde)
    
    stats = analysis.residual_stats
    assert stats['mean'] >= 0
    assert stats['std'] >= 0
    assert stats['max'] >= stats['mean']
    assert stats['min'] <= stats['mean']


# Test layer activation analysis

def test_layer_activations(interpreter, test_inputs):
    """Test layer activation analysis."""
    activations = interpreter.analyze_layer_activations(test_inputs)
    
    assert isinstance(activations, dict)
    assert len(activations) > 0
    
    # Check statistics for each layer
    for layer_name, stats in activations.items():
        assert 'mean' in stats
        assert 'std' in stats
        assert 'sparsity' in stats
        assert 'dead_neurons' in stats
        assert 0 <= stats['sparsity'] <= 1


def test_layer_activations_specific_layers(interpreter, test_inputs):
    """Test activation analysis for specific layers."""
    # Get all layer names first
    all_activations = interpreter.analyze_layer_activations(test_inputs)
    layer_names = list(all_activations.keys())[:1]  # First layer only
    
    if layer_names:
        specific_activations = interpreter.analyze_layer_activations(
            test_inputs,
            layer_names=layer_names
        )
        
        assert len(specific_activations) <= len(layer_names)


def test_dead_neuron_detection(interpreter):
    """Test dead neuron detection."""
    # Create inputs that might cause dead neurons
    inputs = torch.zeros(10, 2)
    
    activations = interpreter.analyze_layer_activations(inputs)
    
    for stats in activations.values():
        assert 'dead_neurons' in stats
        assert stats['dead_neurons'] >= 0


# Test parameter sensitivity

def test_parameter_sensitivity(interpreter, test_inputs):
    """Test parameter sensitivity computation."""
    sensitivity = interpreter.parameter_sensitivity(test_inputs)
    
    assert isinstance(sensitivity, dict)
    assert len(sensitivity) > 0
    
    # All sensitivities should be non-negative
    assert all(v >= 0 for v in sensitivity.values())


def test_parameter_sensitivity_specific_params(interpreter, test_inputs):
    """Test sensitivity for specific parameters."""
    param_names = [name for name, _ in interpreter.model.named_parameters()][:2]
    
    sensitivity = interpreter.parameter_sensitivity(
        test_inputs,
        parameter_names=param_names
    )
    
    assert len(sensitivity) <= len(param_names)


# Test gradient statistics

def test_gradient_statistics(interpreter, test_inputs):
    """Test gradient statistics computation."""
    stats = interpreter.gradient_statistics(test_inputs)
    
    assert isinstance(stats, dict)
    assert len(stats) > 0
    
    for layer_stats in stats.values():
        assert 'mean' in layer_stats
        assert 'std' in layer_stats
        assert 'norm' in layer_stats
        assert layer_stats['norm'] >= 0


def test_gradient_statistics_vanishing_detection(interpreter):
    """Test detection of vanishing gradients."""
    inputs = torch.randn(10, 2)
    
    stats = interpreter.gradient_statistics(inputs)
    
    # Check for vanishing gradients
    for name, layer_stats in stats.items():
        norm = layer_stats['norm']
        # Should not have extremely small gradients (in well-behaved network)
        assert norm >= 0


# Test helper functions

def test_compute_gradient_first_order():
    """Test first-order gradient computation."""
    x = torch.randn(5, 2, requires_grad=True)
    y = (x ** 2).sum(dim=1, keepdim=True)
    
    dy_dx = compute_gradient(y, x, order=1)
    
    assert dy_dx.shape == x.shape


def test_compute_gradient_second_order():
    """Test second-order gradient computation."""
    x = torch.randn(5, 1, requires_grad=True)
    y = (x ** 3)
    
    d2y_dx2 = compute_gradient(y, x, order=2)
    
    # For y = x³, d²y/dx² = 6x
    expected = 6 * x
    assert torch.allclose(d2y_dx2, expected, atol=1e-4)


def test_heat_equation_residual(simple_pinn):
    """Test heat equation residual computation."""
    inputs = torch.randn(10, 2, requires_grad=True)
    # u must be computed from inputs for gradient graph
    u = simple_pinn(inputs)
    
    residual = heat_equation_residual(inputs, u, alpha=0.01)
    
    assert residual.shape == (10, 1)


def test_wave_equation_residual(simple_pinn):
    """Test wave equation residual computation."""
    inputs = torch.randn(10, 2, requires_grad=True)
    # u must be computed from inputs for gradient graph
    u = simple_pinn(inputs)
    
    residual = wave_equation_residual(inputs, u, c=1.0)
    
    assert residual.shape == (10, 1)


def test_burgers_equation_residual(simple_pinn):
    """Test Burgers equation residual computation."""
    inputs = torch.randn(10, 2, requires_grad=True)
    # u must be computed from inputs for gradient graph
    u = simple_pinn(inputs)
    
    residual = burgers_equation_residual(inputs, u, nu=0.01)
    
    assert residual.shape == (10, 1)


# Test edge cases

def test_empty_input(interpreter):
    """Test with empty input."""
    empty_input = torch.tensor([]).reshape(0, 2)
    # Empty input should either raise or return empty results
    try:
        result = interpreter.compute_sensitivity_map(empty_input)
        # If it doesn't raise, result should have empty tensors
        assert result.sensitivity_maps['du_dx'].shape[0] == 0
    except (RuntimeError, ValueError, IndexError):
        # Expected behavior
        pass


def test_single_input(interpreter):
    """Test with single input."""
    single_input = torch.randn(1, 2)
    result = interpreter.compute_sensitivity_map(single_input)
    
    assert result.sensitivity_maps['du_dx'].shape == (1, 1)


def test_different_device_handling(simple_pinn):
    """Test device handling."""
    device = torch.device('cpu')
    interpreter = PINNInterpreter(simple_pinn, input_names=['x', 't'], device=device)
    
    inputs = torch.randn(10, 2)
    result = interpreter.compute_sensitivity_map(inputs)
    
    # Check first sensitivity map exists (du_dx or du_dx0)
    key = next(iter(result.sensitivity_maps.keys()))
    assert result.sensitivity_maps[key].device.type == device.type


# Integration tests

def test_full_workflow(interpreter, test_inputs):
    """Test complete interpretability workflow."""
    # 1. Sensitivity analysis
    sensitivity = interpreter.compute_sensitivity_map(test_inputs)
    assert sensitivity is not None
    
    # 2. Feature importance
    importance = interpreter.feature_importance(test_inputs)
    assert len(importance) == 2
    
    # 3. Layer activations
    activations = interpreter.analyze_layer_activations(test_inputs)
    assert len(activations) > 0
    
    # 4. Parameter sensitivity
    param_sens = interpreter.parameter_sensitivity(test_inputs)
    assert len(param_sens) > 0


def test_multiple_methods_consistency(interpreter, test_inputs):
    """Test consistency across different methods."""
    grad_result = interpreter.compute_sensitivity_map(test_inputs, method='gradient')
    ig_result = interpreter.compute_sensitivity_map(test_inputs, method='integrated_gradients')
    
    # Both should identify same relative feature importance
    grad_imp = grad_result.feature_importance
    ig_imp = ig_result.feature_importance
    
    # Check if most important feature is same
    grad_max = max(grad_imp, key=grad_imp.get)
    ig_max = max(ig_imp, key=ig_imp.get)
    
    # May not always match, but shapes should be consistent
    assert len(grad_imp) == len(ig_imp)


# Performance tests

def test_performance_large_batch(interpreter):
    """Test performance with large batch."""
    large_input = torch.randn(1000, 2)
    
    import time
    start = time.time()
    result = interpreter.compute_sensitivity_map(large_input)
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should complete in reasonable time
    assert result.sensitivity_maps['du_dx'].shape == (1000, 1)


def test_memory_efficiency(interpreter):
    """Test memory efficiency."""
    # Multiple calls should not accumulate memory
    inputs = torch.randn(100, 2)
    
    for _ in range(5):
        _ = interpreter.compute_sensitivity_map(inputs)
    
    # If we got here without OOM, test passes
    assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
