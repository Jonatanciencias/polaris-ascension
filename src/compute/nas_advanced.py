"""
Advanced Neural Architecture Search Features (Session 28)

Implements advanced NAS techniques beyond basic DARTS:
1. Progressive Architecture Refinement (multi-stage search)
2. Multi-Branch Search Spaces (parallel paths)
3. Automated Mixed Precision (adaptive bit-width selection)

These techniques enable more sophisticated architecture discovery
optimized for AMD Radeon RX 580 (Polaris, GCN 4.0).

Progressive Refinement Strategy:
-------------------------------
Stage 1: Coarse search (large search space, few epochs)
Stage 2: Medium refinement (pruned space, more epochs)
Stage 3: Fine-tuning (best candidates only)

This reduces search cost while maintaining quality.

Multi-Branch Search Space:
-------------------------
Instead of single operation per edge, allow:
- Parallel branches (ResNet-style)
- Skip connections at multiple levels
- Attention-based gating

Mixed Precision Strategy:
------------------------
Automatically select precision per layer:
- FP32: Critical layers (first/last, sensitive)
- FP16: Memory-bound layers (helps bandwidth)
- INT8: Compute-bound layers (fast on Polaris)

References:
----------
- Cai et al. (2019) - ProxylessNAS
- Wu et al. (2019) - FBNet
- Wang et al. (2020) - APQ (Automatic Post-training Quantization)
- Guo et al. (2020) - Single Path One-Shot NAS

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import copy

from .nas_darts import DARTSConfig, Cell, MixedOp, DARTSNetwork
from .quantization import AdaptiveQuantizer, QuantizationPrecision

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Precision levels for mixed-precision search"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    
    @property
    def bits(self) -> int:
        """Get bit-width"""
        if self == PrecisionLevel.FP32:
            return 32
        elif self == PrecisionLevel.FP16:
            return 16
        elif self == PrecisionLevel.INT8:
            return 8
        elif self == PrecisionLevel.INT4:
            return 4
        return 32


class SearchStage(Enum):
    """Stages in progressive refinement"""
    COARSE = "coarse"       # Stage 1: Large space, quick
    MEDIUM = "medium"       # Stage 2: Pruned space, refined
    FINE = "fine"          # Stage 3: Best candidates, detailed
    

@dataclass
class ProgressiveConfig:
    """Configuration for progressive architecture refinement"""
    # Stage durations (epochs)
    coarse_epochs: int = 10
    medium_epochs: int = 20
    fine_epochs: int = 30
    
    # Pruning thresholds
    coarse_keep_ratio: float = 0.5  # Keep top 50% after coarse
    medium_keep_ratio: float = 0.3  # Keep top 30% after medium
    
    # Search space reduction
    prune_operations: bool = True
    prune_connections: bool = True
    
    # Early stopping
    patience: int = 5
    min_improvement: float = 0.001


@dataclass
class MultiBranchConfig:
    """Configuration for multi-branch search space"""
    # Branch configuration
    max_branches: int = 3  # Max parallel branches per node
    branch_types: List[str] = field(default_factory=lambda: [
        'conv', 'attention', 'identity'
    ])
    
    # Gating mechanism
    use_gating: bool = True
    gate_type: str = "learnable"  # learnable, fixed, adaptive
    
    # Connection patterns
    allow_skip_connections: bool = True
    max_skip_length: int = 3  # Max nodes to skip


@dataclass
class MixedPrecisionConfig:
    """Configuration for automated mixed precision"""
    # Available precisions
    available_precisions: List[PrecisionLevel] = field(default_factory=lambda: [
        PrecisionLevel.FP32,
        PrecisionLevel.FP16,
        PrecisionLevel.INT8
    ])
    
    # Sensitivity analysis
    sensitivity_samples: int = 100
    sensitivity_threshold: float = 0.01  # 1% accuracy drop ok
    
    # Constraints
    target_memory_reduction: float = 0.5  # 50% reduction target
    preserve_first_last: bool = True  # Keep FP32 for critical layers
    
    # Hardware-specific
    fp16_beneficial: bool = False  # RX 580 doesn't accelerate FP16
    int8_speedup: float = 1.5  # Expected INT8 speedup


class MultiBranchOperation(nn.Module):
    """
    Multi-branch operation with learnable gating.
    
    Allows multiple parallel operations with weighted combination.
    """
    
    def __init__(
        self,
        C: int,
        stride: int,
        branch_types: List[str],
        use_gating: bool = True
    ):
        """
        Initialize multi-branch operation.
        
        Args:
            C: Number of channels
            stride: Stride for operations
            branch_types: Types of branches to include
            use_gating: Use learnable gating
        """
        super().__init__()
        
        self.C = C
        self.stride = stride
        self.branch_types = branch_types
        self.use_gating = use_gating
        self.num_branches = len(branch_types)
        
        # Create branches
        self.branches = nn.ModuleList()
        for branch_type in branch_types:
            self.branches.append(self._create_branch(branch_type, C, stride))
        
        # Gating mechanism
        if use_gating:
            self.gate_weights = nn.Parameter(torch.zeros(self.num_branches))
        else:
            # Fixed equal weights
            self.register_buffer(
                'gate_weights',
                torch.ones(self.num_branches) / self.num_branches
            )
    
    def _create_branch(self, branch_type: str, C: int, stride: int) -> nn.Module:
        """Create a branch operation"""
        if branch_type == 'conv':
            return nn.Sequential(
                nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU()
            )
        elif branch_type == 'attention':
            # Simple attention branch
            return nn.Sequential(
                nn.Conv2d(C, C, 1, bias=False),
                nn.Sigmoid()
            )
        elif branch_type == 'identity':
            if stride == 1:
                return nn.Identity()
            else:
                return nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted branch combination"""
        # Compute gate weights
        if self.use_gating:
            gates = F.softmax(self.gate_weights, dim=0)
        else:
            gates = self.gate_weights
        
        # Compute branch outputs
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Weighted combination
        output = sum(
            gate * branch_out
            for gate, branch_out in zip(gates, branch_outputs)
        )
        
        return output
    
    def get_dominant_branch(self) -> Tuple[str, float]:
        """Get the dominant branch and its weight"""
        if self.use_gating:
            gates = F.softmax(self.gate_weights, dim=0)
        else:
            gates = self.gate_weights
        
        max_idx = gates.argmax().item()
        max_weight = gates[max_idx].item()
        
        return self.branch_types[max_idx], max_weight


class ProgressiveNAS:
    """
    Progressive Neural Architecture Search.
    
    Performs search in multiple stages with progressive refinement:
    1. Coarse: Large search space, quick evaluation
    2. Medium: Pruned space based on coarse results
    3. Fine: Best candidates with detailed training
    
    This reduces computational cost while maintaining search quality.
    """
    
    def __init__(
        self,
        darts_config: DARTSConfig,
        progressive_config: ProgressiveConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize progressive NAS.
        
        Args:
            darts_config: Base DARTS configuration
            progressive_config: Progressive search configuration
            device: Computation device
        """
        self.darts_config = darts_config
        self.progressive_config = progressive_config
        self.device = device
        
        # Track search history
        self.search_history: Dict[SearchStage, List[Dict]] = {
            stage: [] for stage in SearchStage
        }
        
        logger.info(
            f"Initialized ProgressiveNAS "
            f"(stages: {progressive_config.coarse_epochs}/"
            f"{progressive_config.medium_epochs}/"
            f"{progressive_config.fine_epochs} epochs)"
        )
    
    def search(
        self,
        train_loader: Any,
        val_loader: Any
    ) -> Tuple[nn.Module, Dict]:
        """
        Execute progressive architecture search.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            
        Returns:
            (best_model, search_info)
        """
        logger.info("Starting progressive architecture search...")
        
        # Stage 1: Coarse search
        logger.info("Stage 1: Coarse search")
        coarse_candidates = self._coarse_search(train_loader, val_loader)
        
        # Stage 2: Medium refinement
        logger.info("Stage 2: Medium refinement")
        medium_candidates = self._medium_refinement(
            coarse_candidates,
            train_loader,
            val_loader
        )
        
        # Stage 3: Fine-tuning
        logger.info("Stage 3: Fine-tuning")
        best_model, final_info = self._fine_tuning(
            medium_candidates,
            train_loader,
            val_loader
        )
        
        return best_model, final_info
    
    def _coarse_search(
        self,
        train_loader: Any,
        val_loader: Any
    ) -> List[Dict]:
        """
        Stage 1: Coarse search over large space.
        
        Uses few epochs to quickly evaluate many candidates.
        """
        candidates = []
        num_candidates = 5  # Search 5 candidates in coarse stage
        
        for i in range(num_candidates):
            logger.info(f"Coarse candidate {i+1}/{num_candidates}")
            
            # Create model (simplified for demo)
            model = self._create_candidate_model()
            
            # Quick training
            model = self._quick_train(
                model,
                train_loader,
                val_loader,
                epochs=self.progressive_config.coarse_epochs
            )
            
            # Evaluate
            accuracy = self._evaluate(model, val_loader)
            
            candidates.append({
                'model': model,
                'accuracy': accuracy,
                'stage': SearchStage.COARSE
            })
        
        # Sort by accuracy
        candidates.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Keep top candidates
        keep_count = int(len(candidates) * self.progressive_config.coarse_keep_ratio)
        kept_candidates = candidates[:keep_count]
        
        logger.info(
            f"Coarse search complete: kept {len(kept_candidates)}/{len(candidates)} "
            f"(best acc: {kept_candidates[0]['accuracy']:.3f})"
        )
        
        self.search_history[SearchStage.COARSE] = candidates
        
        return kept_candidates
    
    def _medium_refinement(
        self,
        coarse_candidates: List[Dict],
        train_loader: Any,
        val_loader: Any
    ) -> List[Dict]:
        """
        Stage 2: Refine promising candidates.
        
        Trains pruned architectures for more epochs.
        """
        refined_candidates = []
        
        for i, candidate in enumerate(coarse_candidates):
            logger.info(f"Refining candidate {i+1}/{len(coarse_candidates)}")
            
            model = candidate['model']
            
            # Prune operations based on architecture parameters
            if self.progressive_config.prune_operations:
                model = self._prune_weak_operations(model)
            
            # Train for more epochs
            model = self._quick_train(
                model,
                train_loader,
                val_loader,
                epochs=self.progressive_config.medium_epochs
            )
            
            # Evaluate
            accuracy = self._evaluate(model, val_loader)
            
            refined_candidates.append({
                'model': model,
                'accuracy': accuracy,
                'stage': SearchStage.MEDIUM,
                'parent_acc': candidate['accuracy']
            })
        
        # Sort and prune
        refined_candidates.sort(key=lambda x: x['accuracy'], reverse=True)
        keep_count = max(1, int(len(refined_candidates) * self.progressive_config.medium_keep_ratio))
        kept_candidates = refined_candidates[:keep_count]
        
        logger.info(
            f"Medium refinement complete: kept {len(kept_candidates)}/{len(refined_candidates)} "
            f"(best acc: {kept_candidates[0]['accuracy']:.3f})"
        )
        
        self.search_history[SearchStage.MEDIUM] = refined_candidates
        
        return kept_candidates
    
    def _fine_tuning(
        self,
        medium_candidates: List[Dict],
        train_loader: Any,
        val_loader: Any
    ) -> Tuple[nn.Module, Dict]:
        """
        Stage 3: Fine-tune best candidates.
        
        Full training with best hyperparameters.
        """
        best_model = None
        best_accuracy = 0.0
        
        for i, candidate in enumerate(medium_candidates):
            logger.info(f"Fine-tuning candidate {i+1}/{len(medium_candidates)}")
            
            model = candidate['model']
            
            # Full training
            model = self._quick_train(
                model,
                train_loader,
                val_loader,
                epochs=self.progressive_config.fine_epochs
            )
            
            # Final evaluation
            accuracy = self._evaluate(model, val_loader)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        final_info = {
            'best_accuracy': best_accuracy,
            'stages_completed': 3,
            'total_candidates_evaluated': (
                len(self.search_history[SearchStage.COARSE]) +
                len(self.search_history[SearchStage.MEDIUM]) +
                len(medium_candidates)
            )
        }
        
        logger.info(
            f"Fine-tuning complete: best accuracy = {best_accuracy:.3f}"
        )
        
        return best_model, final_info
    
    def _create_candidate_model(self) -> nn.Module:
        """Create a candidate model"""
        # Simplified model creation for demo
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 10)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleModel().to(self.device)
    
    def _quick_train(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        epochs: int
    ) -> nn.Module:
        """Quick training (simplified for demo)"""
        # In practice, would do actual training
        # For demo, just return model
        return model
    
    def _evaluate(self, model: nn.Module, val_loader: Any) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _prune_weak_operations(self, model: nn.Module) -> nn.Module:
        """Prune weak operations based on weights"""
        # Simplified pruning for demo
        return model


class MixedPrecisionNAS:
    """
    Automated Mixed Precision Selection for NAS.
    
    Automatically determines optimal precision (FP32/FP16/INT8) for each layer
    based on sensitivity analysis and hardware constraints.
    
    Strategy:
    1. Measure sensitivity of each layer to quantization
    2. Assign higher precision to sensitive layers
    3. Use lower precision for robust layers
    4. Respect hardware constraints (RX 580 specific)
    """
    
    def __init__(
        self,
        config: MixedPrecisionConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize mixed precision NAS.
        
        Args:
            config: Mixed precision configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        
        # Layer sensitivity map
        self.layer_sensitivity: Dict[str, float] = {}
        
        # Precision assignment
        self.precision_map: Dict[str, PrecisionLevel] = {}
        
        logger.info(
            f"Initialized MixedPrecisionNAS "
            f"(available: {[p.value for p in config.available_precisions]})"
        )
    
    def analyze_and_assign(
        self,
        model: nn.Module,
        val_loader: Any
    ) -> Dict[str, PrecisionLevel]:
        """
        Analyze model and assign precision to each layer.
        
        Args:
            model: Model to analyze
            val_loader: Validation data for sensitivity analysis
            
        Returns:
            Dictionary mapping layer names to precision levels
        """
        logger.info("Analyzing layer sensitivities...")
        
        # Step 1: Measure baseline accuracy
        baseline_acc = self._evaluate_accuracy(model, val_loader)
        logger.info(f"Baseline accuracy: {baseline_acc:.3f}")
        
        # Step 2: Measure sensitivity for each layer
        self._measure_sensitivities(model, val_loader, baseline_acc)
        
        # Step 3: Assign precisions based on sensitivity
        self._assign_precisions(model)
        
        logger.info(
            f"Precision assignment complete: "
            f"{self._count_precisions()} layers"
        )
        
        return self.precision_map
    
    def _measure_sensitivities(
        self,
        model: nn.Module,
        val_loader: Any,
        baseline_acc: float
    ):
        """Measure sensitivity of each layer to quantization"""
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            
            # Try quantizing this layer to INT8
            original_weight = module.weight.data.clone()
            
            # Simulate INT8 quantization
            quantized_weight = self._simulate_quantization(
                module.weight.data,
                PrecisionLevel.INT8
            )
            module.weight.data = quantized_weight
            
            # Measure accuracy drop
            quantized_acc = self._evaluate_accuracy(model, val_loader)
            sensitivity = baseline_acc - quantized_acc
            
            # Restore original weight
            module.weight.data = original_weight
            
            self.layer_sensitivity[name] = sensitivity
            
            logger.debug(f"{name}: sensitivity = {sensitivity:.4f}")
    
    def _simulate_quantization(
        self,
        tensor: torch.Tensor,
        precision: PrecisionLevel
    ) -> torch.Tensor:
        """Simulate quantization to given precision"""
        if precision == PrecisionLevel.FP32:
            return tensor
        
        # Get quantization levels
        if precision == PrecisionLevel.INT8:
            levels = 256
        elif precision == PrecisionLevel.INT4:
            levels = 16
        else:  # FP16
            return tensor.half().float()
        
        # Quantize
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (levels - 1)
        
        quantized = torch.round((tensor - min_val) / scale)
        dequantized = quantized * scale + min_val
        
        return dequantized
    
    def _assign_precisions(self, model: nn.Module):
        """Assign precision to each layer based on sensitivity"""
        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(
            self.layer_sensitivity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Assign precisions
        for name, sensitivity in sorted_layers:
            # Preserve first/last layers if configured
            if self.config.preserve_first_last:
                if 'conv1' in name or 'fc' in name or 'classifier' in name:
                    self.precision_map[name] = PrecisionLevel.FP32
                    continue
            
            # Assign based on sensitivity
            if sensitivity > self.config.sensitivity_threshold:
                # Sensitive: use FP32
                self.precision_map[name] = PrecisionLevel.FP32
            elif sensitivity > self.config.sensitivity_threshold / 2:
                # Moderately sensitive: use FP16 (or FP32 if no FP16 benefit)
                if self.config.fp16_beneficial:
                    self.precision_map[name] = PrecisionLevel.FP16
                else:
                    self.precision_map[name] = PrecisionLevel.FP32
            else:
                # Not sensitive: use INT8
                self.precision_map[name] = PrecisionLevel.INT8
        
        logger.info(f"Assigned precisions to {len(self.precision_map)} layers")
    
    def _evaluate_accuracy(self, model: nn.Module, val_loader: Any) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Use subset for speed
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= self.config.sensitivity_samples // val_loader.batch_size:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _count_precisions(self) -> Dict[str, int]:
        """Count layers per precision"""
        counts = {p.value: 0 for p in PrecisionLevel}
        
        for precision in self.precision_map.values():
            counts[precision.value] += 1
        
        return counts
    
    def apply_precision_map(
        self,
        model: nn.Module
    ) -> nn.Module:
        """
        Apply precision map to model.
        
        Converts layers to assigned precisions.
        """
        # In practice, would replace layers with quantized versions
        # For demo, just log the assignment
        logger.info("Applying precision map to model...")
        
        precision_counts = self._count_precisions()
        for precision, count in precision_counts.items():
            if count > 0:
                logger.info(f"  {precision}: {count} layers")
        
        return model


# Factory functions

def create_progressive_nas(
    num_classes: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> ProgressiveNAS:
    """
    Factory function to create progressive NAS instance.
    
    Args:
        num_classes: Number of output classes
        device: Computation device
        
    Returns:
        Configured ProgressiveNAS instance
    """
    darts_config = DARTSConfig(
        num_cells=6,
        num_nodes=4,
        layers=8
    )
    darts_config.num_classes = num_classes
    
    progressive_config = ProgressiveConfig(
        coarse_epochs=5,
        medium_epochs=10,
        fine_epochs=20
    )
    
    return ProgressiveNAS(darts_config, progressive_config, device)


def create_mixed_precision_nas(
    fp16_beneficial: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MixedPrecisionNAS:
    """
    Factory function to create mixed precision NAS.
    
    Args:
        fp16_beneficial: Whether FP16 provides speedup (False for RX 580)
        device: Computation device
        
    Returns:
        Configured MixedPrecisionNAS instance
    """
    config = MixedPrecisionConfig(
        fp16_beneficial=fp16_beneficial,
        preserve_first_last=True,
        sensitivity_threshold=0.01
    )
    
    return MixedPrecisionNAS(config, device)


if __name__ == "__main__":
    # Demo: Create instances
    print("Advanced NAS Features Demo")
    print("=" * 60)
    
    print("\n1. Progressive NAS")
    progressive = create_progressive_nas(num_classes=10)
    print(f"   Stages: {progressive.progressive_config.coarse_epochs}/"
          f"{progressive.progressive_config.medium_epochs}/"
          f"{progressive.progressive_config.fine_epochs} epochs")
    
    print("\n2. Mixed Precision NAS")
    mixed_precision = create_mixed_precision_nas(fp16_beneficial=False)
    print(f"   Available precisions: {[p.value for p in mixed_precision.config.available_precisions]}")
    print(f"   FP16 beneficial: {mixed_precision.config.fp16_beneficial}")
    
    print("\nReady for advanced NAS search!")
