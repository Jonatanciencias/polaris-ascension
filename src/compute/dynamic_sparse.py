"""
Dynamic Sparse Training - RigL Implementation
==============================================

Session 11: Train sparse networks from scratch without the prune-retrain cycle.

Implements:
- RigL (Rigged Lottery): Drop lowest magnitude, grow highest gradient
- Dynamic Sparsity Allocation: Non-uniform sparsity per layer
- Gradient-based importance scoring

References:
- Evci et al. (2020) "Rigging the Lottery: Making All Tickets Winners"
- Mostafa & Wang (2019) "Parameter Efficient Training of Deep CNNs"
- Gale et al. (2019) "The State of Sparsity in Deep Neural Networks"

Version: 0.6.0-dev
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class RigLConfig:
    """
    Configuration for RigL (Rigged Lottery) sparse training.
    
    Attributes:
        sparsity: Target sparsity (fraction of weights to prune)
        T_end: Stop updating masks at this step
        delta_T: Update mask every delta_T steps
        alpha: Fraction of connections to drop and regrow each update
        grad_accumulation_steps: Steps to accumulate gradients for growth
    """
    sparsity: float = 0.9
    T_end: int = 10000
    delta_T: int = 100
    alpha: float = 0.3  # Drop/grow 30% of pruned connections
    grad_accumulation_steps: int = 1


class RigLPruner:
    """
    RigL: Rigged Lottery sparse training.
    
    Instead of pruning a pre-trained model, RigL trains sparse from scratch:
    1. Initialize with random sparse mask
    2. Periodically:
       - Drop: Remove lowest magnitude weights
       - Grow: Add connections with highest gradients
    3. Maintain constant sparsity throughout training
    
    Key advantage: No retraining needed, often better accuracy than static pruning.
    
    Algorithm:
    ----------
    At each update step t (every delta_T steps):
    
    1. Drop phase:
       n_drop = alpha * n_pruned  # Drop 30% of pruned connections
       drop_indices = argsort(|w|)[where mask == 1][:n_drop]
       mask[drop_indices] = 0
    
    2. Grow phase:
       n_grow = n_drop  # Maintain constant sparsity
       grow_indices = argsort(|grad|)[where mask == 0][-n_grow:]
       mask[grow_indices] = 1
       w[grow_indices] = small_random_init
    
    3. Apply mask:
       w = w * mask
    
    References:
    ----------
    Evci et al. (2020) "Rigging the Lottery: Making All Tickets Winners"
    - arXiv:1911.11134
    - Introduces RigL algorithm
    - Shows superior performance vs static pruning
    
    Example:
    -------
    >>> rigl = RigLPruner(sparsity=0.9, T_end=10000, delta_T=100)
    >>> 
    >>> # Training loop
    >>> for step in range(training_steps):
    ...     # Forward pass, compute gradients
    ...     loss.backward()
    ...     
    ...     # Update mask if needed
    ...     if rigl.should_update(step):
    ...         weights, mask = rigl.update_mask(
    ...             weights, gradients, step
    ...         )
    ...     
    ...     # Update weights
    ...     optimizer.step()
    """
    
    def __init__(
        self,
        sparsity: float = 0.9,
        T_end: int = 10000,
        delta_T: int = 100,
        alpha: float = 0.3,
        initial_distribution: str = "ERK",  # Erdos-Renyi-Kernel
        grad_accumulation_steps: int = 1,
    ):
        """
        Initialize RigL pruner.
        
        Args:
            sparsity: Target sparsity (0.0-1.0)
            T_end: Stop updating masks after this step
            delta_T: Update mask every delta_T steps
            alpha: Fraction of pruned connections to drop/grow each update
            initial_distribution: How to distribute initial sparsity ("uniform" or "ERK")
            grad_accumulation_steps: Accumulate gradients over N steps for growth
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0.0, 1.0), got {sparsity}")
        
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Alpha must be in (0.0, 1.0], got {alpha}")
        
        self.sparsity = sparsity
        self.T_end = T_end
        self.delta_T = delta_T
        self.alpha = alpha
        self.initial_distribution = initial_distribution
        self.grad_accumulation_steps = grad_accumulation_steps
        
        # Tracking
        self.current_step = 0
        self.update_history = []
        self.accumulated_gradients = {}
        self.grad_step_counter = 0
        
    def initialize_mask(
        self,
        weights: np.ndarray,
        layer_name: str = "layer"
    ) -> np.ndarray:
        """
        Initialize sparse mask for a layer.
        
        For uniform distribution, randomly prune to target sparsity.
        For ERK (Erdos-Renyi-Kernel), use layer-wise sparsity based on dimensions.
        
        Args:
            weights: Weight tensor
            layer_name: Name of layer (for tracking)
            
        Returns:
            Binary mask (1=keep, 0=prune)
        """
        # Create mask
        mask = np.ones_like(weights, dtype=np.float32)
        
        # Compute number of weights to keep
        total_params = weights.size
        params_to_keep = int(total_params * (1 - self.sparsity))
        
        if params_to_keep < 1:
            warnings.warn(f"Layer {layer_name}: All parameters pruned")
            params_to_keep = 1
        
        # Random initialization
        flat_mask = mask.flatten()
        prune_indices = np.random.choice(
            total_params, 
            size=total_params - params_to_keep,
            replace=False
        )
        flat_mask[prune_indices] = 0
        
        mask = flat_mask.reshape(weights.shape)
        
        actual_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        self.update_history.append({
            "step": 0,
            "layer": layer_name,
            "action": "initialize",
            "sparsity": actual_sparsity,
            "params_kept": int(np.count_nonzero(mask)),
        })
        
        return mask
    
    def should_update(self, step: int) -> bool:
        """
        Check if mask should be updated at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if should update mask
        """
        if step > self.T_end:
            return False
        
        if step == 0:
            return False
        
        return (step % self.delta_T) == 0
    
    def accumulate_gradients(
        self,
        gradients: np.ndarray,
        layer_name: str
    ):
        """
        Accumulate gradients for growth phase.
        
        Args:
            gradients: Gradient tensor
            layer_name: Name of layer
        """
        if layer_name not in self.accumulated_gradients:
            self.accumulated_gradients[layer_name] = np.zeros_like(gradients)
        
        # Accumulate absolute gradients
        self.accumulated_gradients[layer_name] += np.abs(gradients)
    
    def update_mask(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
        mask: np.ndarray,
        step: int,
        layer_name: str = "layer"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update mask using RigL algorithm: drop lowest magnitude, grow highest gradient.
        
        Args:
            weights: Current weights
            gradients: Current gradients
            mask: Current mask
            step: Current training step
            layer_name: Name of layer
            
        Returns:
            Tuple of (updated_weights, updated_mask)
        """
        # Accumulate gradients
        self.accumulate_gradients(gradients, layer_name)
        self.grad_step_counter += 1
        
        # Only update if we've accumulated enough gradients
        if self.grad_step_counter < self.grad_accumulation_steps:
            return weights * mask, mask
        
        # Reset counter
        self.grad_step_counter = 0
        
        # Get accumulated gradients
        accumulated_grads = self.accumulated_gradients.get(
            layer_name, 
            np.abs(gradients)
        )
        
        # Reset accumulation
        self.accumulated_gradients[layer_name] = np.zeros_like(gradients)
        
        # Compute number of connections to drop/grow
        # In RigL, we drop/grow a fraction of the PRUNED connections
        # This maintains constant sparsity while allowing topology changes
        n_pruned = int(np.sum(mask == 0))
        n_active = int(np.sum(mask == 1))
        
        if n_pruned == 0:
            # No pruned connections, cannot drop/grow
            return weights * mask, mask
        
        # Calculate drop/grow based on pruned count
        n_drop = max(1, int(self.alpha * n_pruned))
        n_grow = n_drop  # Maintain constant sparsity
        
        # Ensure we don't drop more than available active connections
        n_drop = min(n_drop, n_active - 1)  # Keep at least 1 active
        n_grow = min(n_grow, n_pruned)      # Can't grow more than pruned
        
        if n_drop == 0 or n_grow == 0:
            return weights * mask, mask
        
        # Make a copy to avoid modifying original
        new_mask = mask.copy()
        new_weights = weights.copy()
        
        # DROP PHASE: Remove lowest magnitude weights
        # Only consider currently active weights (mask == 1)
        active_indices = np.where(mask == 1)
        active_magnitudes = np.abs(weights[active_indices])
        
        # Get indices of smallest magnitudes
        if len(active_magnitudes) > n_drop:
            # Use argpartition to find smallest n_drop elements efficiently
            drop_partition_idx = np.argpartition(active_magnitudes, n_drop)[:n_drop]
            
            # Convert local indices to global coordinates
            drop_coords = tuple(arr[drop_partition_idx] for arr in active_indices)
            
            # Apply drop to new mask
            new_mask[drop_coords] = 0
        
        # GROW PHASE: Add connections with highest gradients
        # Only consider currently inactive weights (mask == 0)
        # Use NEW mask to account for just-dropped connections
        inactive_indices = np.where(new_mask == 0)
        inactive_gradients = accumulated_grads[inactive_indices]
        
        if len(inactive_gradients) >= n_grow:
            # Use argpartition to find largest n_grow elements efficiently
            grow_partition_idx = np.argpartition(inactive_gradients, -n_grow)[-n_grow:]
            
            # Convert local indices to global coordinates
            grow_coords = tuple(arr[grow_partition_idx] for arr in inactive_indices)
            
            # Apply grow to new mask
            new_mask[grow_coords] = 1
            
            # Initialize new weights with small random values
            new_weights[grow_coords] = np.random.randn(len(grow_coords[0])) * 0.01
        
        # Apply mask to weights
        new_weights = new_weights * new_mask
        
        # Track update
        actual_sparsity = 1.0 - (np.count_nonzero(new_mask) / new_mask.size)
        self.update_history.append({
            "step": step,
            "layer": layer_name,
            "action": "update",
            "n_drop": n_drop,
            "n_grow": n_grow,
            "sparsity": actual_sparsity,
            "params_kept": int(np.count_nonzero(new_mask)),
        })
        
        self.current_step = step
        
        return new_weights, new_mask
    
    def get_update_schedule(self) -> List[int]:
        """
        Get list of steps where mask updates occur.
        
        Returns:
            List of update steps
        """
        return list(range(self.delta_T, self.T_end + 1, self.delta_T))
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about RigL training.
        
        Returns:
            Dictionary with statistics
        """
        if not self.update_history:
            return {}
        
        updates = [h for h in self.update_history if h["action"] == "update"]
        
        return {
            "total_updates": len(updates),
            "steps_updated": [h["step"] for h in updates],
            "final_sparsity": self.update_history[-1]["sparsity"] if updates else 0.0,
            "avg_drop_per_update": np.mean([h["n_drop"] for h in updates]) if updates else 0,
            "avg_grow_per_update": np.mean([h["n_grow"] for h in updates]) if updates else 0,
            "total_connections_changed": sum(h["n_drop"] + h["n_grow"] for h in updates),
        }


class DynamicSparsityAllocator:
    """
    Dynamic per-layer sparsity allocation based on sensitivity.
    
    Instead of uniform sparsity across all layers, allocate sparsity based on:
    - Layer dimensions (larger layers can be sparser)
    - Gradient magnitudes (more important layers get lower sparsity)
    - Loss sensitivity (layers with higher impact get lower sparsity)
    
    This often improves accuracy vs uniform sparsity.
    
    Algorithm:
    ----------
    1. Compute layer sensitivities:
       sensitivity[l] = ||∇L/∇w_l||_2  (gradient magnitude)
    
    2. Allocate sparsity inversely proportional to sensitivity:
       sparsity[l] ∝ 1 / (sensitivity[l] + ε)
    
    3. Normalize to achieve target overall sparsity:
       Σ(sparsity[l] * params[l]) / Σ(params[l]) = target_sparsity
    
    References:
    ----------
    Mostafa & Wang (2019) "Parameter Efficient Training of Deep CNNs"
    - Dynamic sparsity reparameterization (DSR)
    - Layer-wise sensitivity analysis
    
    Example:
    -------
    >>> allocator = DynamicSparsityAllocator(target_sparsity=0.9)
    >>> 
    >>> # Compute sensitivities
    >>> gradients = {"layer1": grad1, "layer2": grad2}
    >>> sensitivities = allocator.compute_sensitivities(gradients)
    >>> 
    >>> # Allocate sparsity
    >>> model_weights = {"layer1": w1, "layer2": w2}
    >>> sparsities = allocator.allocate_sparsity(
    ...     model_weights, sensitivities
    ... )
    >>> 
    >>> print(sparsities)
    >>> # {'layer1': 0.95, 'layer2': 0.85}  # More sensitive layer2 gets lower sparsity
    """
    
    def __init__(
        self,
        target_sparsity: float = 0.9,
        method: str = "gradient",  # "gradient", "hessian", "uniform"
        eps: float = 1e-8,
    ):
        """
        Initialize dynamic sparsity allocator.
        
        Args:
            target_sparsity: Overall target sparsity across model
            method: Sensitivity computation method
            eps: Small constant for numerical stability
        """
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError(f"Target sparsity must be in [0.0, 1.0), got {target_sparsity}")
        
        self.target_sparsity = target_sparsity
        self.method = method
        self.eps = eps
        self.allocation_history = []
        
    def compute_sensitivities(
        self,
        gradients: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute per-layer sensitivities based on gradients.
        
        Args:
            gradients: Dictionary of {layer_name: gradient_tensor}
            
        Returns:
            Dictionary of {layer_name: sensitivity_score}
        """
        sensitivities = {}
        
        for layer_name, grad in gradients.items():
            if self.method == "gradient":
                # L2 norm of gradients
                sensitivity = np.linalg.norm(grad.flatten())
            elif self.method == "gradient_mean":
                # Mean absolute gradient
                sensitivity = np.mean(np.abs(grad))
            elif self.method == "uniform":
                # No sensitivity (uniform sparsity)
                sensitivity = 1.0
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            sensitivities[layer_name] = sensitivity
        
        return sensitivities
    
    def allocate_sparsity(
        self,
        model_weights: Dict[str, np.ndarray],
        sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Allocate per-layer sparsity based on sensitivities.
        
        More sensitive layers get lower sparsity (keep more parameters).
        
        Args:
            model_weights: Dictionary of {layer_name: weights}
            sensitivities: Dictionary of {layer_name: sensitivity_score}
            
        Returns:
            Dictionary of {layer_name: layer_sparsity}
        """
        # Get layer sizes
        layer_sizes = {
            name: weights.size
            for name, weights in model_weights.items()
        }
        
        total_params = sum(layer_sizes.values())
        target_pruned_params = self.target_sparsity * total_params
        
        if self.method == "uniform":
            # Uniform sparsity across all layers
            layer_sparsities = {
                name: self.target_sparsity
                for name in model_weights.keys()
            }
        else:
            # Dynamic allocation based on sensitivities
            # Higher sensitivity = more important = LOWER sparsity (keep more params)
            
            # Invert sensitivities: 1/sens means less important layers get higher values
            inv_sensitivities = {
                name: 1.0 / (sens + self.eps)
                for name, sens in sensitivities.items()
            }
            
            # Normalize to sum to 1
            total_inv_sens = sum(inv_sensitivities.values())
            if total_inv_sens == 0:
                total_inv_sens = 1.0
                
            # Each layer gets a "weight" for how many params it should prune
            # Higher inv_sens = less important = more pruning
            pruning_weights = {
                name: inv_sens / total_inv_sens
                for name, inv_sens in inv_sensitivities.items()
            }
            
            # Initial allocation: distribute target_pruned_params proportionally
            layer_pruned_counts = {}
            total_allocated = 0
            
            for name in sorted(model_weights.keys()):
                # This layer gets this fraction of pruned params
                desired_pruned = pruning_weights[name] * target_pruned_params
                
                # Clamp to valid range [0, size-1]
                max_prunable = layer_sizes[name] - 1
                actual_pruned = min(desired_pruned, max_prunable)
                actual_pruned = max(0, actual_pruned)
                
                layer_pruned_counts[name] = actual_pruned
                total_allocated += actual_pruned
            
            # If we couldn't allocate enough (due to clamping), redistribute deficit
            if total_allocated < target_pruned_params - 1:
                deficit = target_pruned_params - total_allocated
                
                # Try to add more pruning to layers that have room
                for name in sorted(model_weights.keys(), key=lambda n: pruning_weights[n], reverse=True):
                    if deficit <= 0:
                        break
                    
                    current_pruned = layer_pruned_counts[name]
                    max_prunable = layer_sizes[name] - 1
                    available_room = max_prunable - current_pruned
                    
                    if available_room > 0:
                        add_pruned = min(deficit, available_room)
                        layer_pruned_counts[name] += add_pruned
                        deficit -= add_pruned
            
            # Convert counts to sparsity ratios
            layer_sparsities = {
                name: layer_pruned_counts[name] / layer_sizes[name]
                for name in model_weights.keys()
            }
        
        # Final clamp to [0, 0.99]
        layer_sparsities = {
            name: min(0.99, max(0.0, sparsity))
            for name, sparsity in layer_sparsities.items()
        }
        
        # Record allocation
        self.allocation_history.append({
            "target_sparsity": self.target_sparsity,
            "layer_sparsities": layer_sparsities.copy(),
            "sensitivities": sensitivities.copy(),
        })
        
        return layer_sparsities
    
    def get_statistics(self) -> Dict:
        """
        Get allocation statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.allocation_history:
            return {}
        
        latest = self.allocation_history[-1]
        sparsities = list(latest["layer_sparsities"].values())
        
        return {
            "num_allocations": len(self.allocation_history),
            "target_sparsity": self.target_sparsity,
            "min_layer_sparsity": min(sparsities) if sparsities else 0.0,
            "max_layer_sparsity": max(sparsities) if sparsities else 0.0,
            "mean_layer_sparsity": np.mean(sparsities) if sparsities else 0.0,
            "std_layer_sparsity": np.std(sparsities) if sparsities else 0.0,
        }
