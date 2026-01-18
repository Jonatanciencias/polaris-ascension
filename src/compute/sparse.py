"""
Sparse Operations for AMD GCN Architecture
==========================================

This module implements sparse tensor operations optimized for the AMD GCN
(Graphics Core Next) architecture, specifically targeting the wavefront
execution model of Polaris and Vega GPUs.

Theoretical Foundation:
----------------------
AMD GCN GPUs execute instructions in wavefronts of 64 threads. Sparse
operations can achieve significant speedups when:
- Sparsity > 70% (3x+ speedup potential)
- Non-zero elements align with wavefront boundaries
- Memory access patterns favor coalesced reads

Implementation Strategy:
-----------------------
1. CSR (Compressed Sparse Row) format for row-major operations
2. Block-sparse patterns aligned to 64-element boundaries
3. Dynamic sparsity detection with automatic format selection

Target Performance:
------------------
On RX 580 (8GB, 2304 cores):
- Dense: ~6 TFLOPS (FP32)
- Sparse (90%): ~18 TFLOPS effective (theoretical)
- Memory bandwidth: 256 GB/s

References:
----------
- AMD GCN Architecture Whitepaper
- "Exploiting Sparsity on AMD GPUs" - adapted from research
- deep_philosophy.md - Original design notes

Version: 0.5.0-dev (Planned for 0.6.0)
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SparseTensorConfig:
    """Configuration for sparse tensor operations."""
    target_sparsity: float = 0.7  # 70% zeros
    wavefront_size: int = 64      # AMD GCN wavefront
    block_size: int = 64          # Align to wavefront
    min_speedup_threshold: float = 1.5  # Minimum 1.5x to use sparse


class SparseOperations:
    """
    Sparse tensor operations optimized for AMD GCN architecture.
    
    This class provides sparse matrix multiplication and other operations
    that exploit the wavefront execution model of AMD GPUs.
    
    Note: Currently a placeholder - full implementation in v0.6.0
    """
    
    def __init__(
        self,
        target_density: float = 0.3,
        gpu_family: str = "polaris",
        config: Optional[SparseTensorConfig] = None
    ):
        """
        Initialize sparse operations handler.
        
        Args:
            target_density: Expected density (1 - sparsity) of tensors
            gpu_family: Target GPU family ("polaris", "vega", "navi")
            config: Optional configuration override
        """
        self.target_density = target_density
        self.gpu_family = gpu_family
        self.config = config or SparseTensorConfig()
        
        # GPU-specific optimizations
        self._wavefront_sizes = {
            "polaris": 64,
            "vega": 64,
            "navi": 32,  # RDNA uses wave32
        }
        
        self._setup_gpu_params()
    
    def _setup_gpu_params(self):
        """Configure parameters based on GPU family."""
        self.wavefront_size = self._wavefront_sizes.get(
            self.gpu_family, 64
        )
        
    def analyze_sparsity(
        self, 
        tensor: np.ndarray
    ) -> dict:
        """
        Analyze tensor sparsity and recommend optimization strategy.
        
        Args:
            tensor: Input tensor to analyze
            
        Returns:
            dict with sparsity metrics and recommendations
        """
        total_elements = tensor.size
        zero_elements = np.sum(tensor == 0)
        sparsity = zero_elements / total_elements
        
        # Calculate potential speedup
        if sparsity > 0.9:
            potential_speedup = 3.0
            recommendation = "highly_sparse"
        elif sparsity > 0.7:
            potential_speedup = 2.0
            recommendation = "sparse"
        elif sparsity > 0.5:
            potential_speedup = 1.3
            recommendation = "moderate_sparse"
        else:
            potential_speedup = 1.0
            recommendation = "dense"
            
        return {
            "sparsity": sparsity,
            "density": 1 - sparsity,
            "total_elements": total_elements,
            "zero_elements": zero_elements,
            "nonzero_elements": total_elements - zero_elements,
            "potential_speedup": potential_speedup,
            "recommendation": recommendation,
            "wavefront_aligned": (total_elements % self.wavefront_size) == 0,
        }
    
    def to_csr(
        self, 
        dense_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dense tensor to CSR (Compressed Sparse Row) format.
        
        Args:
            dense_tensor: 2D dense matrix
            
        Returns:
            Tuple of (values, column_indices, row_pointers)
        """
        if dense_tensor.ndim != 2:
            raise ValueError("CSR conversion requires 2D matrix")
            
        rows, cols = dense_tensor.shape
        values = []
        col_indices = []
        row_pointers = [0]
        
        for i in range(rows):
            for j in range(cols):
                if dense_tensor[i, j] != 0:
                    values.append(dense_tensor[i, j])
                    col_indices.append(j)
            row_pointers.append(len(values))
            
        return {
            "values": np.array(values, dtype=dense_tensor.dtype),
            "col_indices": np.array(col_indices, dtype=np.int32),
            "row_pointers": np.array(row_pointers, dtype=np.int32),
        }
    
    def sparse_matmul(
        self,
        weight_matrix: np.ndarray,
        input_tensor: np.ndarray,
        use_csr: bool = True
    ) -> np.ndarray:
        """
        Sparse matrix multiplication optimized for GCN.
        
        Note: Currently falls back to dense multiplication.
        GPU-accelerated version planned for v0.6.0
        
        Args:
            weight_matrix: Sparse weight matrix
            input_tensor: Dense input tensor
            use_csr: Whether to use CSR format (default True)
            
        Returns:
            Result tensor
        """
        # Analyze sparsity
        analysis = self.analyze_sparsity(weight_matrix)
        
        # TODO v0.6.0: Implement GPU-accelerated sparse matmul
        # For now, use numpy dense multiplication
        if analysis["recommendation"] in ["highly_sparse", "sparse"]:
            # Placeholder: Would use CSR-based GPU kernel
            pass
            
        return np.matmul(weight_matrix, input_tensor)


class MagnitudePruner:
    """
    Magnitude-based weight pruning.
    
    Prunes weights based on their absolute magnitude, removing smallest weights
    first. This is the simplest and most commonly used pruning method.
    
    Formula:
        mask[i] = 1 if |w[i]| >= threshold else 0
        
    Where threshold is computed to achieve target sparsity.
    
    References:
    - Han et al. (2015) "Learning both Weights and Connections for Efficient 
      Neural Networks"
    - Zhu & Gupta (2017) "To prune, or not to prune: exploring the efficacy 
      of pruning for model compression"
    
    Example:
        >>> pruner = MagnitudePruner(sparsity=0.7)
        >>> pruned_weights, mask = pruner.prune_layer(weights)
        >>> print(f"Sparsity: {pruner.measure_sparsity(pruned_weights):.1%}")
    """
    
    def __init__(
        self,
        sparsity: float = 0.7,
        scope: str = "local"  # "local" or "global"
    ):
        """
        Initialize magnitude pruner.
        
        Args:
            sparsity: Target sparsity (fraction of weights to prune)
            scope: Pruning scope - "local" (per-layer) or "global" (whole model)
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0.0, 1.0), got {sparsity}")
        
        self.sparsity = sparsity
        self.scope = scope
        self.pruning_history = []
        
    def prune_layer(
        self,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune a single layer based on weight magnitude.
        
        Args:
            weights: Weight tensor to prune
            bias: Optional bias (not pruned)
            
        Returns:
            Tuple of (pruned_weights, mask)
            - pruned_weights: Weights with small values zeroed
            - mask: Binary mask (1=keep, 0=prune)
        """
        # Calculate threshold using percentile
        weight_magnitude = np.abs(weights)
        threshold = np.percentile(weight_magnitude, self.sparsity * 100)
        
        # Create binary mask
        mask = (weight_magnitude >= threshold).astype(np.float32)
        
        # Apply mask
        pruned_weights = weights * mask
        
        # Record pruning stats
        actual_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
        self.pruning_history.append({
            "layer_shape": weights.shape,
            "target_sparsity": self.sparsity,
            "actual_sparsity": actual_sparsity,
            "threshold": threshold,
            "num_pruned": int(mask.size - np.count_nonzero(mask)),
            "num_remaining": int(np.count_nonzero(mask)),
        })
        
        return pruned_weights, mask
    
    def prune_model(
        self,
        model_weights: dict,
        layer_names: Optional[list] = None
    ) -> Tuple[dict, dict]:
        """
        Prune multiple layers with global or local scope.
        
        Args:
            model_weights: Dictionary of {layer_name: weights}
            layer_names: Optional list of layer names to prune (default: all)
            
        Returns:
            Tuple of (pruned_weights, masks)
        """
        if layer_names is None:
            layer_names = list(model_weights.keys())
        
        pruned_weights = {}
        masks = {}
        
        if self.scope == "global":
            # Global pruning: compute threshold across all layers
            all_weights = np.concatenate([
                model_weights[name].flatten() 
                for name in layer_names
            ])
            global_threshold = np.percentile(
                np.abs(all_weights), 
                self.sparsity * 100
            )
            
            for name in layer_names:
                weights = model_weights[name]
                mask = (np.abs(weights) >= global_threshold).astype(np.float32)
                pruned_weights[name] = weights * mask
                masks[name] = mask
                
        else:
            # Local pruning: prune each layer independently
            for name in layer_names:
                pruned_weights[name], masks[name] = self.prune_layer(
                    model_weights[name]
                )
        
        return pruned_weights, masks
    
    def measure_sparsity(self, weights: np.ndarray) -> float:
        """
        Measure actual sparsity of pruned weights.
        
        Args:
            weights: Weight tensor
            
        Returns:
            Sparsity as fraction of zero weights
        """
        return 1.0 - (np.count_nonzero(weights) / weights.size)
    
    def get_compression_stats(self, masks: dict) -> dict:
        """
        Calculate compression statistics across all pruned layers.
        
        Args:
            masks: Dictionary of binary masks
            
        Returns:
            dict with compression metrics
        """
        total_params = sum(mask.size for mask in masks.values())
        remaining_params = sum(np.count_nonzero(mask) for mask in masks.values())
        
        return {
            "total_parameters": total_params,
            "remaining_parameters": remaining_params,
            "pruned_parameters": total_params - remaining_params,
            "compression_ratio": total_params / remaining_params if remaining_params > 0 else 0,
            "sparsity": 1.0 - (remaining_params / total_params),
            "memory_reduction": f"{(1.0 - remaining_params/total_params) * 100:.1f}%",
        }


class StructuredPruner:
    """
    Structured pruning that removes entire channels, filters, or attention heads.
    
    Unlike unstructured (magnitude) pruning, structured pruning removes groups
    of weights, maintaining dense computation patterns that are GPU-friendly.
    
    Advantages:
    - No need for sparse kernels (uses regular dense ops)
    - Better GPU utilization (coalesced memory access)
    - Actual speedup without specialized hardware
    
    References:
    - Li et al. (2017) "Pruning Filters for Efficient ConvNets"
    - Liu et al. (2017) "Learning Efficient Convolutional Networks through 
      Network Slimming"
    - Michel et al. (2019) "Are Sixteen Heads Really Better than One?" (for attention)
    
    Example:
        >>> pruner = StructuredPruner(sparsity=0.5, granularity="channel")
        >>> pruned_weights = pruner.prune_conv_layer(conv_weights)
        >>> # Output channels reduced by 50%
    """
    
    def __init__(
        self,
        sparsity: float = 0.5,
        granularity: str = "channel",  # "channel", "filter", "head"
        importance_metric: str = "l1"  # "l1", "l2", "taylor"
    ):
        """
        Initialize structured pruner.
        
        Args:
            sparsity: Fraction of structures to prune
            granularity: What to prune ("channel", "filter", "head")
            importance_metric: How to score importance ("l1", "l2", "taylor")
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0.0, 1.0), got {sparsity}")
        
        self.sparsity = sparsity
        self.granularity = granularity
        self.importance_metric = importance_metric
        self.pruning_history = []
        
    def _compute_importance_scores(
        self,
        weights: np.ndarray,
        axis: int
    ) -> np.ndarray:
        """
        Compute importance scores for each structure along axis.
        
        Args:
            weights: Weight tensor
            axis: Axis along which to compute scores
            
        Returns:
            1D array of importance scores
        """
        if self.importance_metric == "l1":
            # L1 norm: sum of absolute values
            scores = np.sum(np.abs(weights), axis=tuple(
                i for i in range(weights.ndim) if i != axis
            ))
        elif self.importance_metric == "l2":
            # L2 norm: sum of squares
            scores = np.sqrt(np.sum(weights ** 2, axis=tuple(
                i for i in range(weights.ndim) if i != axis
            )))
        elif self.importance_metric == "taylor":
            # Taylor expansion: |weight * gradient|
            # For now, use L1 (gradient not available in inference)
            scores = np.sum(np.abs(weights), axis=tuple(
                i for i in range(weights.ndim) if i != axis
            ))
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")
        
        return scores
    
    def prune_channels(
        self,
        weights: np.ndarray,
        channel_axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune entire output channels.
        
        For Conv2D: (out_channels, in_channels, H, W)
        For Linear: (out_features, in_features)
        
        Args:
            weights: Weight tensor
            channel_axis: Axis representing output channels (default 0)
            
        Returns:
            Tuple of (pruned_weights, channel_indices_kept)
        """
        num_channels = weights.shape[channel_axis]
        num_to_prune = int(num_channels * self.sparsity)
        num_to_keep = max(1, num_channels - num_to_prune)  # Ensure at least 1 channel
        
        # Compute importance scores
        scores = self._compute_importance_scores(weights, channel_axis)
        
        # Select channels to keep (highest scores)
        indices_to_keep = np.argsort(scores)[-num_to_keep:]
        indices_to_keep = np.sort(indices_to_keep)  # Maintain order
        
        # Select channels
        pruned_weights = np.take(weights, indices_to_keep, axis=channel_axis)
        
        # Record stats
        self.pruning_history.append({
            "granularity": "channel",
            "original_channels": num_channels,
            "remaining_channels": num_to_keep,
            "pruned_channels": num_to_prune,
            "actual_sparsity": num_to_prune / num_channels,
            "original_shape": weights.shape,
            "pruned_shape": pruned_weights.shape,
        })
        
        return pruned_weights, indices_to_keep
    
    def prune_filters(
        self,
        weights: np.ndarray,
        filter_axis: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune entire input filters/channels.
        
        Args:
            weights: Conv weight tensor (out_ch, in_ch, H, W)
            filter_axis: Axis representing input filters (default 1)
            
        Returns:
            Tuple of (pruned_weights, filter_indices_kept)
        """
        # Same logic as prune_channels but on different axis
        return self.prune_channels(weights, channel_axis=filter_axis)
    
    def prune_attention_heads(
        self,
        attention_weights: np.ndarray,
        num_heads: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune entire attention heads.
        
        Args:
            attention_weights: Attention weight matrix (d_model, d_model)
            num_heads: Number of attention heads
            
        Returns:
            Tuple of (pruned_weights, head_indices_kept)
        """
        d_model = attention_weights.shape[0]
        head_dim = d_model // num_heads
        
        # Split into heads and compute importance per head
        weights_per_head = attention_weights.reshape(num_heads, head_dim, d_model)
        head_scores = np.sum(np.abs(weights_per_head), axis=(1, 2))
        
        # Select heads to keep
        num_to_prune = int(num_heads * self.sparsity)
        num_to_keep = num_heads - num_to_prune
        
        if num_to_keep < 1:
            raise ValueError("Cannot prune all attention heads")
        
        indices_to_keep = np.argsort(head_scores)[-num_to_keep:]
        indices_to_keep = np.sort(indices_to_keep)
        
        # Reconstruct weight matrix with remaining heads
        kept_weights = weights_per_head[indices_to_keep]
        pruned_weights = kept_weights.reshape(num_to_keep * head_dim, d_model)
        
        self.pruning_history.append({
            "granularity": "attention_head",
            "original_heads": num_heads,
            "remaining_heads": num_to_keep,
            "pruned_heads": num_to_prune,
        })
        
        return pruned_weights, indices_to_keep
    
    def measure_sparsity(self, weights: np.ndarray) -> float:
        """
        Measure sparsity of pruned weights (for structured, based on zeroed structures).
        
        Args:
            weights: Weight tensor
            
        Returns:
            Estimated sparsity
        """
        return 1.0 - (np.count_nonzero(weights) / weights.size)


class GradualPruner:
    """
    Gradual iterative pruning with polynomial decay schedule.
    
    Instead of pruning all at once, gradually increase sparsity over training.
    This allows the network to adapt and often maintains better accuracy.
    
    Schedule formula:
        s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (n * Δt))³
        
    Where:
    - s(t): sparsity at step t
    - s_i: initial sparsity
    - s_f: final sparsity
    - t_0: begin pruning step
    - n: pruning frequency
    - Δt: time delta
    
    References:
    - Zhu & Gupta (2017) "To prune, or not to prune: exploring the efficacy 
      of pruning for model compression"
    - Gale et al. (2019) "The State of Sparsity in Deep Neural Networks"
    
    Example:
        >>> pruner = GradualPruner(
        ...     initial_sparsity=0.0,
        ...     final_sparsity=0.9,
        ...     begin_step=1000,
        ...     end_step=10000
        ... )
        >>> for step in range(training_steps):
        ...     if pruner.should_prune(step):
        ...         weights = pruner.prune_step(weights, step)
    """
    
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        begin_step: int = 0,
        end_step: int = 10000,
        frequency: int = 100,
        pruning_method: str = "magnitude"  # "magnitude" or "structured"
    ):
        """
        Initialize gradual pruner.
        
        Args:
            initial_sparsity: Starting sparsity
            final_sparsity: Target sparsity at end
            begin_step: Step to start pruning
            end_step: Step to reach final sparsity
            frequency: Prune every N steps
            pruning_method: Base pruning method to use
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        self.pruning_method = pruning_method
        
        # Create base pruner
        if pruning_method == "magnitude":
            self.base_pruner = MagnitudePruner(sparsity=initial_sparsity)
        elif pruning_method == "structured":
            self.base_pruner = StructuredPruner(sparsity=initial_sparsity)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
        
        self.current_step = 0
        self.pruning_schedule = []
        
    def compute_sparsity(self, step: int) -> float:
        """
        Compute target sparsity for given step using polynomial decay.
        
        Args:
            step: Current training step
            
        Returns:
            Target sparsity for this step
        """
        if step < self.begin_step:
            return self.initial_sparsity
        
        if step >= self.end_step:
            return self.final_sparsity
        
        # Polynomial decay (cubic)
        progress = (step - self.begin_step) / (self.end_step - self.begin_step)
        sparsity = self.final_sparsity + (
            (self.initial_sparsity - self.final_sparsity) * 
            ((1.0 - progress) ** 3)
        )
        
        return sparsity
    
    def should_prune(self, step: int) -> bool:
        """
        Check if pruning should occur at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if should prune
        """
        if step < self.begin_step or step > self.end_step:
            return False
        
        return (step - self.begin_step) % self.frequency == 0
    
    def prune_step(
        self,
        weights: np.ndarray,
        step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one pruning step.
        
        Args:
            weights: Weights to prune
            step: Current training step
            
        Returns:
            Tuple of (pruned_weights, mask)
        """
        target_sparsity = self.compute_sparsity(step)
        
        # Update base pruner sparsity
        self.base_pruner.sparsity = target_sparsity
        
        # Prune
        if self.pruning_method == "magnitude":
            pruned_weights, mask = self.base_pruner.prune_layer(weights)
        elif self.pruning_method == "structured":
            pruned_weights, indices = self.base_pruner.prune_channels(weights)
            # Create mask for structured pruning
            mask = np.ones_like(weights)
            # Mark pruned channels as 0
            all_indices = set(range(weights.shape[0]))
            pruned_indices = all_indices - set(indices)
            for idx in pruned_indices:
                mask[idx] = 0
        
        # Record schedule
        self.pruning_schedule.append({
            "step": step,
            "target_sparsity": target_sparsity,
            "actual_sparsity": self.base_pruner.measure_sparsity(pruned_weights),
        })
        
        self.current_step = step
        return pruned_weights, mask
    
    def get_schedule(self, num_steps: Optional[int] = None) -> list:
        """
        Get the pruning schedule for visualization.
        
        Args:
            num_steps: Number of steps to generate schedule for
            
        Returns:
            List of (step, sparsity) tuples
        """
        if num_steps is None:
            num_steps = self.end_step + 1000
        
        schedule = []
        for step in range(0, num_steps, self.frequency):
            sparsity = self.compute_sparsity(step)
            schedule.append((step, sparsity))
        
        return schedule


class FineTuningScheduler:
    """
    Fine-tuning scheduler for sparse networks after pruning.
    
    After pruning to target sparsity, fine-tune the remaining weights to
    recover accuracy. This class provides:
    - Learning rate schedules optimized for sparse fine-tuning
    - Layer-wise learning rates (different rates for pruned vs unpruned layers)
    - Early stopping based on validation metrics
    
    Typical workflow:
    1. Prune network to target sparsity
    2. Fine-tune remaining weights with lower learning rate
    3. Monitor validation accuracy and stop early if needed
    
    References:
    - Han et al. (2015) "Learning both Weights and Connections"
    - Zhu & Gupta (2017) pruning paper recommends fine-tuning
    
    Example:
        >>> scheduler = FineTuningScheduler(
        ...     initial_lr=0.001,
        ...     min_lr=0.00001,
        ...     patience=5
        ... )
        >>> 
        >>> # After pruning
        >>> for epoch in range(fine_tune_epochs):
        ...     lr = scheduler.get_lr(epoch)
        ...     train_epoch(model, lr)
        ...     
        ...     val_loss = validate(model)
        ...     should_stop = scheduler.step(val_loss)
        ...     if should_stop:
        ...         break
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        min_lr: float = 0.00001,
        patience: int = 5,
        factor: float = 0.5,
        mode: str = "cosine",  # "cosine", "exponential", "step"
        warmup_epochs: int = 0,
    ):
        """
        Initialize fine-tuning scheduler.
        
        Args:
            initial_lr: Starting learning rate for fine-tuning
            min_lr: Minimum learning rate (don't go below this)
            patience: Number of epochs without improvement before LR reduction
            factor: Factor to multiply LR by when reducing
            mode: LR schedule type
            warmup_epochs: Number of warmup epochs at start
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        
        self.current_epoch = 0
        self.current_lr = initial_lr
        self.best_metric = float('inf')
        self.epochs_without_improvement = 0
        self.lr_history = []
        
    def get_lr(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Learning rate for this epoch
        """
        # Warmup phase
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        
        # Adjust for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        
        if self.mode == "cosine":
            # Cosine annealing
            import math
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * adjusted_epoch / 100)
            )
        elif self.mode == "exponential":
            # Exponential decay
            lr = self.initial_lr * (self.factor ** adjusted_epoch)
        elif self.mode == "step":
            # Step decay (every 10 epochs)
            lr = self.initial_lr * (self.factor ** (adjusted_epoch // 10))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Clamp to min_lr
        lr = max(lr, self.min_lr)
        
        return lr
    
    def step(self, metric: float) -> bool:
        """
        Update scheduler based on validation metric.
        
        Args:
            metric: Validation metric (lower is better, e.g., loss)
            
        Returns:
            True if should stop training (early stopping triggered)
        """
        self.current_epoch += 1
        self.current_lr = self.get_lr(self.current_epoch)
        self.lr_history.append(self.current_lr)
        
        # Check if metric improved
        if metric < self.best_metric:
            self.best_metric = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Early stopping
        if self.epochs_without_improvement >= self.patience * 2:
            return True  # Stop training
        
        # Reduce LR if plateaued
        if self.epochs_without_improvement >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
        
        return False
    
    def get_statistics(self) -> dict:
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "current_epoch": self.current_epoch,
            "current_lr": self.current_lr,
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "total_lr_reductions": len([
                i for i in range(1, len(self.lr_history))
                if self.lr_history[i] < self.lr_history[i-1]
            ]),
        }


def apply_mask_to_gradients(
    gradients: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply sparse mask to gradients during fine-tuning.
    
    During fine-tuning, we want to prevent pruned weights from being updated.
    This function zeros out gradients for pruned weights.
    
    Args:
        gradients: Gradient tensor
        mask: Binary mask (1=keep, 0=pruned)
        
    Returns:
        Masked gradients
    """
    return gradients * mask


def create_sparse_layer(
    in_features: int,
    out_features: int,
    sparsity: float = 0.9,
    gpu_family: str = "polaris"
) -> dict:
    """
    Factory function to create a sparse layer configuration.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        sparsity: Target sparsity (0.0-1.0)
        gpu_family: Target GPU family
        
    Returns:
        dict with layer configuration
    """
    return {
        "type": "sparse_linear",
        "in_features": in_features,
        "out_features": out_features,
        "sparsity": sparsity,
        "gpu_family": gpu_family,
        "wavefront_aligned": (in_features % 64 == 0) and (out_features % 64 == 0),
        "implementation_status": "v0.6.0",
    }
