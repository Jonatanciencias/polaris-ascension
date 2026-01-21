"""
PINN Interpretability Module - Session 22
==========================================

Explainability and interpretability tools for Physics-Informed Neural Networks (PINNs).

Key Features:
-------------
1. Sensitivity Analysis
   - Gradient-based sensitivity maps (∂u/∂x, ∂u/∂t)
   - Parameter sensitivity
   - Input feature importance
   
2. Physics Residual Visualization
   - PDE residual heatmaps
   - Error distribution analysis
   - Boundary condition validation
   
3. Layer Activation Analysis
   - Internal representation analysis
   - Feature visualization
   - Activation statistics

4. Saliency Maps
   - Input saliency
   - Gradient-based attribution
   - Integrated gradients

Papers Implemented:
-------------------
1. Krishnapriyan et al. (2021): "Characterizing possible failure modes in PINNs"
2. Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework"
3. Sundararajan et al. (2017): "Axiomatic Attribution for Deep Networks"

Author: Session 22 Implementation
Date: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    
    sensitivity_maps: Dict[str, torch.Tensor]  # e.g., {'du_dx': tensor, 'du_dt': tensor}
    feature_importance: Dict[str, float]  # e.g., {'x': 0.7, 't': 0.3}
    layer_contributions: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


@dataclass
class ResidualAnalysis:
    """Results from physics residual analysis."""
    
    residual_map: torch.Tensor  # Spatial distribution of PDE residual
    residual_stats: Dict[str, float]  # {'mean': x, 'std': y, 'max': z}
    error_distribution: torch.Tensor  # Histogram of errors
    hotspots: List[Tuple[float, float]]  # Locations with high residual


class PINNInterpreter:
    """
    Interpretability tools for Physics-Informed Neural Networks.
    
    Provides methods to understand and visualize PINN behavior:
    - Sensitivity to input variables
    - Physics residual distribution
    - Layer activation patterns
    - Feature importance
    
    Example:
    --------
    >>> interpreter = PINNInterpreter(pinn_model)
    >>> 
    >>> # Compute sensitivity maps
    >>> sensitivity = interpreter.compute_sensitivity_map(test_points)
    >>> # {'du_dx': tensor(...), 'du_dt': tensor(...)}
    >>> 
    >>> # Analyze physics residual
    >>> residual = interpreter.analyze_residual(domain, pde_fn)
    >>> # ResidualAnalysis(residual_map=..., residual_stats=...)
    >>> 
    >>> # Feature importance
    >>> importance = interpreter.feature_importance(test_points)
    >>> # {'x': 0.7, 't': 0.3}
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize PINN interpreter.
        
        Args:
            model: PINN model to interpret
            input_names: Names of input variables (e.g., ['x', 't'])
            device: Device for computation
        """
        self.model = model
        self.input_names = input_names or ['x0', 'x1']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Cache for gradients and activations
        self._gradient_cache: Dict[str, torch.Tensor] = {}
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
    def compute_sensitivity_map(
        self,
        inputs: torch.Tensor,
        output_idx: int = 0,
        method: str = 'gradient'
    ) -> SensitivityResult:
        """
        Compute sensitivity of output to input variables.
        
        For PINNs solving u(x,t), computes ∂u/∂x and ∂u/∂t.
        
        Args:
            inputs: Input tensor (batch_size, input_dim)
            output_idx: Which output to analyze (for multi-output models)
            method: 'gradient', 'integrated_gradients', or 'smooth_grad'
            
        Returns:
            SensitivityResult with sensitivity maps and importance scores
            
        Example:
        --------
        >>> inputs = torch.tensor([[0.5, 0.1], [0.7, 0.2]])  # (x, t) pairs
        >>> result = interpreter.compute_sensitivity_map(inputs)
        >>> print(result.sensitivity_maps['du_dx'])
        tensor([[0.45], [0.62]])  # Sensitivity to x
        """
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        
        if method == 'gradient':
            return self._gradient_sensitivity(inputs, output_idx)
        elif method == 'integrated_gradients':
            return self._integrated_gradients(inputs, output_idx)
        elif method == 'smooth_grad':
            return self._smooth_grad(inputs, output_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gradient_sensitivity(
        self,
        inputs: torch.Tensor,
        output_idx: int
    ) -> SensitivityResult:
        """Compute gradient-based sensitivity."""
        # Forward pass
        outputs = self.model(inputs)
        if outputs.dim() > 1 and outputs.size(1) > 1:
            outputs = outputs[:, output_idx]
        
        # Compute gradients for each input dimension
        sensitivity_maps = {}
        feature_importance = {}
        
        for i, name in enumerate(self.input_names[:inputs.size(1)]):
            # Compute ∂u/∂x_i
            grad = torch.autograd.grad(
                outputs=outputs.sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            sensitivity_maps[f'du_d{name}'] = grad[:, i:i+1].detach()
            
            # Feature importance: mean absolute gradient
            feature_importance[name] = grad[:, i].abs().mean().item()
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        return SensitivityResult(
            sensitivity_maps=sensitivity_maps,
            feature_importance=feature_importance
        )
    
    def _integrated_gradients(
        self,
        inputs: torch.Tensor,
        output_idx: int,
        num_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> SensitivityResult:
        """
        Compute integrated gradients (Sundararajan et al., 2017).
        
        More robust attribution method that satisfies axioms:
        - Sensitivity: If input changes, attribution changes
        - Implementation invariance: Equivalent networks give same attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        interpolated = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (inputs - baseline).unsqueeze(0)
        
        # Compute gradients along path
        sensitivity_maps = {}
        
        for i, name in enumerate(self.input_names[:inputs.size(1)]):
            grad_sum = torch.zeros_like(inputs[:, i:i+1])
            
            for alpha_inputs in interpolated:
                alpha_inputs.requires_grad_(True)
                outputs = self.model(alpha_inputs)
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    outputs = outputs[:, output_idx]
                
                grad = torch.autograd.grad(
                    outputs=outputs.sum(),
                    inputs=alpha_inputs,
                    retain_graph=False
                )[0]
                
                grad_sum += grad[:, i:i+1].detach()
            
            # Integrated gradients = (x - baseline) * mean_gradient
            integrated_grad = (inputs[:, i:i+1] - baseline[:, i:i+1]) * grad_sum / num_steps
            sensitivity_maps[f'du_d{name}'] = integrated_grad
        
        # Feature importance
        feature_importance = {
            name: sensitivity_maps[f'du_d{name}'].abs().mean().item()
            for name in self.input_names[:inputs.size(1)]
        }
        
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        return SensitivityResult(
            sensitivity_maps=sensitivity_maps,
            feature_importance=feature_importance,
            metadata={'method': 'integrated_gradients', 'num_steps': num_steps}
        )
    
    def _smooth_grad(
        self,
        inputs: torch.Tensor,
        output_idx: int,
        num_samples: int = 50,
        noise_level: float = 0.1
    ) -> SensitivityResult:
        """
        Compute SmoothGrad (Smilkov et al., 2017).
        
        Reduces noise by averaging gradients over noisy samples.
        """
        sensitivity_accumulator = {
            name: torch.zeros_like(inputs[:, i:i+1])
            for i, name in enumerate(self.input_names[:inputs.size(1)])
        }
        
        for _ in range(num_samples):
            # Add Gaussian noise
            noise = torch.randn_like(inputs) * noise_level * inputs.std()
            noisy_inputs = inputs + noise
            noisy_inputs.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(noisy_inputs)
            if outputs.dim() > 1 and outputs.size(1) > 1:
                outputs = outputs[:, output_idx]
            
            # Gradients
            for i, name in enumerate(self.input_names[:inputs.size(1)]):
                grad = torch.autograd.grad(
                    outputs=outputs.sum(),
                    inputs=noisy_inputs,
                    retain_graph=True if i < inputs.size(1) - 1 else False
                )[0]
                sensitivity_accumulator[name] += grad[:, i:i+1].detach()
        
        # Average over samples
        sensitivity_maps = {
            f'du_d{name}': sens / num_samples
            for name, sens in sensitivity_accumulator.items()
        }
        
        # Feature importance
        feature_importance = {
            name: sensitivity_maps[f'du_d{name}'].abs().mean().item()
            for name in self.input_names[:inputs.size(1)]
        }
        
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        return SensitivityResult(
            sensitivity_maps=sensitivity_maps,
            feature_importance=feature_importance,
            metadata={'method': 'smooth_grad', 'num_samples': num_samples}
        )
    
    def analyze_residual(
        self,
        domain_points: torch.Tensor,
        pde_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> ResidualAnalysis:
        """
        Analyze physics residual (PDE error) across domain.
        
        For a PDE: F(u, ∂u/∂x, ∂²u/∂x², ...) = 0
        Computes residual R = |F(u_pred, ...)|
        
        Args:
            domain_points: Points in domain (N, input_dim)
            pde_function: Function computing PDE residual
                          Signature: pde_function(inputs, u_pred) -> residual
            grid_shape: Shape for reshaping residual into 2D grid (for heatmap)
            
        Returns:
            ResidualAnalysis with residual distribution and statistics
            
        Example:
        --------
        >>> # For heat equation: ∂u/∂t - α * ∂²u/∂x² = 0
        >>> def heat_pde(inputs, u):
        >>>     du_dt = grad(u, inputs, 1)  # ∂u/∂t
        >>>     d2u_dx2 = grad(grad(u, inputs, 0), inputs, 0)  # ∂²u/∂x²
        >>>     return du_dt - 0.01 * d2u_dx2
        >>> 
        >>> points = torch.tensor([[x, t] for x, t in domain])
        >>> analysis = interpreter.analyze_residual(points, heat_pde)
        >>> print(f"Mean residual: {analysis.residual_stats['mean']:.6f}")
        """
        domain_points = domain_points.to(self.device)
        domain_points.requires_grad_(True)
        
        # Forward pass
        u_pred = self.model(domain_points)
        
        # Compute PDE residual
        residual = pde_function(domain_points, u_pred)
        residual_abs = residual.abs().detach()
        
        # Reshape to grid if requested
        if grid_shape is not None:
            try:
                residual_map = residual_abs.view(*grid_shape)
            except RuntimeError:
                warnings.warn(f"Cannot reshape {residual_abs.shape} to {grid_shape}")
                residual_map = residual_abs
        else:
            residual_map = residual_abs
        
        # Statistics
        residual_stats = {
            'mean': residual_abs.mean().item(),
            'std': residual_abs.std().item(),
            'max': residual_abs.max().item(),
            'min': residual_abs.min().item(),
            'median': residual_abs.median().item()
        }
        
        # Find hotspots (top 1% highest residual)
        threshold = torch.quantile(residual_abs, 0.99)
        hotspot_indices = torch.where(residual_abs.squeeze() > threshold)[0]
        hotspots = [
            tuple(domain_points[idx].detach().cpu().numpy())
            for idx in hotspot_indices[:10]  # Top 10
        ]
        
        # Error distribution (histogram)
        error_distribution = torch.histc(
            residual_abs,
            bins=50,
            min=0,
            max=residual_abs.max().item()
        )
        
        return ResidualAnalysis(
            residual_map=residual_map,
            residual_stats=residual_stats,
            error_distribution=error_distribution,
            hotspots=hotspots
        )
    
    def analyze_layer_activations(
        self,
        inputs: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze activation patterns in network layers.
        
        Args:
            inputs: Input tensor
            layer_names: Specific layers to analyze (None = all)
            
        Returns:
            Dict mapping layer names to activation statistics
            
        Example:
        --------
        >>> activations = interpreter.analyze_layer_activations(test_points)
        >>> print(activations['hidden_layer_1'])
        {'mean': 0.15, 'std': 0.42, 'sparsity': 0.23, 'dead_neurons': 5}
        """
        inputs = inputs.to(self.device)
        
        # Register forward hooks
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks on all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if layer_names is None or name in layer_names:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations
        analysis = {}
        for name, activation in activations.items():
            # Flatten to (batch, features)
            act_flat = activation.view(activation.size(0), -1)
            
            analysis[name] = {
                'mean': act_flat.mean().item(),
                'std': act_flat.std().item(),
                'min': act_flat.min().item(),
                'max': act_flat.max().item(),
                'sparsity': (act_flat.abs() < 1e-6).float().mean().item(),
                'dead_neurons': (act_flat.abs().mean(dim=0) < 1e-6).sum().item(),
                'activation_range': (act_flat.max() - act_flat.min()).item()
            }
        
        return analysis
    
    def parameter_sensitivity(
        self,
        inputs: torch.Tensor,
        parameter_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute sensitivity of output to model parameters.
        
        Useful for understanding which parameters are most important.
        
        Args:
            inputs: Input tensor
            parameter_names: Specific parameters to analyze
            
        Returns:
            Dict mapping parameter names to sensitivity scores
        """
        inputs = inputs.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = outputs.mean()  # Aggregate output
        
        # Compute gradients
        gradients = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False,
            retain_graph=False
        )
        
        # Sensitivity = |∂loss/∂param|
        sensitivity = {}
        for (name, param), grad in zip(self.model.named_parameters(), gradients):
            if parameter_names is None or name in parameter_names:
                if grad is not None:
                    sensitivity[name] = grad.abs().mean().item()
        
        return sensitivity
    
    def feature_importance(
        self,
        inputs: torch.Tensor,
        method: str = 'gradient'
    ) -> Dict[str, float]:
        """
        Compute feature importance for input variables.
        
        Wrapper around compute_sensitivity_map for simpler API.
        
        Args:
            inputs: Input tensor
            method: 'gradient', 'integrated_gradients', or 'smooth_grad'
            
        Returns:
            Dict mapping input feature names to importance scores (0-1)
            
        Example:
        --------
        >>> importance = interpreter.feature_importance(test_points)
        >>> print(importance)
        {'x': 0.73, 't': 0.27}  # x is more important than t
        """
        result = self.compute_sensitivity_map(inputs, method=method)
        return result.feature_importance
    
    def gradient_statistics(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute gradient statistics for network parameters.
        
        Useful for diagnosing training issues (vanishing/exploding gradients).
        
        Returns:
            Dict mapping layer names to gradient statistics
        """
        inputs = inputs.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = outputs.mean()
        
        # Zero existing gradients
        self.model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect gradient statistics
        stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
        
        return stats


def compute_gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    order: int = 1
) -> torch.Tensor:
    """
    Compute derivatives of outputs w.r.t. inputs.
    
    Helper function for computing ∂u/∂x, ∂²u/∂x², etc.
    
    Args:
        outputs: Model outputs (batch_size, output_dim)
        inputs: Model inputs (batch_size, input_dim)
        order: Derivative order (1 for first derivative, 2 for second, etc.)
        
    Returns:
        Derivative tensor
        
    Example:
    --------
    >>> u = model(inputs)  # u(x,t)
    >>> du_dx = compute_gradient(u, inputs[:, 0:1], order=1)
    >>> d2u_dx2 = compute_gradient(u, inputs[:, 0:1], order=2)
    """
    grad = outputs
    for _ in range(order):
        grad = torch.autograd.grad(
            grad.sum(),
            inputs,
            create_graph=True,
            retain_graph=True
        )[0]
    return grad


# Convenience functions for common PDE residuals

def heat_equation_residual(
    inputs: torch.Tensor,
    u: torch.Tensor,
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Residual for 1D heat equation: ∂u/∂t - α * ∂²u/∂x² = 0.
    
    Args:
        inputs: (x, t) coordinates
        u: Solution u(x, t)
        alpha: Thermal diffusivity
    """
    du_dt = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0][:, 1:2]
    
    du_dx = torch.autograd.grad(u.sum(), inputs, create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), inputs, create_graph=True)[0][:, 0:1]
    
    return du_dt - alpha * d2u_dx2


def wave_equation_residual(
    inputs: torch.Tensor,
    u: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Residual for 1D wave equation: ∂²u/∂t² - c² * ∂²u/∂x² = 0.
    
    Args:
        inputs: (x, t) coordinates
        u: Solution u(x, t)
        c: Wave speed
    """
    du_dt = torch.autograd.grad(u.sum(), inputs, create_graph=True, retain_graph=True)[0][:, 1:2]
    d2u_dt2 = torch.autograd.grad(du_dt.sum(), inputs, create_graph=True)[0][:, 1:2]
    
    du_dx = torch.autograd.grad(u.sum(), inputs, create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), inputs, create_graph=True)[0][:, 0:1]
    
    return d2u_dt2 - c**2 * d2u_dx2


def burgers_equation_residual(
    inputs: torch.Tensor,
    u: torch.Tensor,
    nu: float = 0.01
) -> torch.Tensor:
    """
    Residual for Burgers' equation: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0.
    
    Args:
        inputs: (x, t) coordinates
        u: Solution u(x, t)
        nu: Viscosity
    """
    du_dt = torch.autograd.grad(u.sum(), inputs, create_graph=True, retain_graph=True)[0][:, 1:2]
    du_dx = torch.autograd.grad(u.sum(), inputs, create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), inputs, create_graph=True)[0][:, 0:1]
    
    return du_dt + u * du_dx - nu * d2u_dx2
