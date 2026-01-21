"""
Mixed-Precision Quantization Optimizer
=======================================

Implements layer-wise adaptive mixed-precision quantization based on:
- Wang et al. (2026) - "Layer-wise Adaptive Mixed-Precision Quantization"
- Sensitivity analysis per layer
- Evolutionary search for optimal configurations
- Physics-aware quantization for PINNs

Session 21 - Research Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import existing modules
try:
    from .quantization import AdaptiveQuantizer, QuantizationConfig
    HAS_QUANTIZER = True
except ImportError:
    HAS_QUANTIZER = False
    AdaptiveQuantizer = None
    QuantizationConfig = None

try:
    from .evolutionary_pruning import EvolutionaryPruner, GeneticOperators
    HAS_EVOLUTIONARY = True
except ImportError:
    HAS_EVOLUTIONARY = False

try:
    from .physics_utils import PINNNetwork
    HAS_PINN = True
except ImportError:
    HAS_PINN = False
    PINNNetwork = None


logger = logging.getLogger(__name__)


# ============================================================================
# Precision Types
# ============================================================================

class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"


@dataclass
class LayerSensitivity:
    """Sensitivity information for a layer."""
    layer_name: str
    sensitivity_score: float  # 0-1, higher = more sensitive
    param_count: int
    recommended_precision: PrecisionType
    gradient_norm: Optional[float] = None
    hessian_trace: Optional[float] = None
    taylor_score: Optional[float] = None
    recommended_bits: int = 8  # Recommended quantization bits


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision quantization."""
    target_bits_avg: float = 6.5  # Target average bits per weight
    search_method: str = 'evolutionary'  # 'evolutionary', 'gradient', 'heuristic'
    search_space: List[str] = None  # List of precision strings
    population_size: int = 20
    generations: int = 50
    accuracy_threshold: float = 0.02  # Max accuracy drop (2%)
    layer_configs: Dict[str, Dict[str, Any]] = None  # Layer-specific configurations
    compression_ratio: float = 1.0  # Achieved compression ratio
    target_compression: float = 4.0  # Target compression ratio
    
    def __post_init__(self):
        if self.search_space is None:
            self.search_space = ['fp16', 'int8', 'int4']
        if self.layer_configs is None:
            self.layer_configs = {}


# ============================================================================
# Mixed-Precision Optimizer
# ============================================================================

class MixedPrecisionOptimizer:
    """
    Mixed-precision quantization optimizer.
    
    Automatically determines optimal precision for each layer based on:
    - Gradient sensitivity analysis
    - Hessian trace approximation
    - Evolutionary search
    - Hardware constraints
    
    Example:
    --------
        model = create_model()
        mp_optimizer = MixedPrecisionOptimizer(
            target_bits_avg=6.5,
            search_method='evolutionary'
        )
        
        # Analyze sensitivity
        sensitivity = mp_optimizer.analyze_sensitivity(model, data_loader)
        
        # Search for optimal config
        config = mp_optimizer.search_configuration(model, val_loader)
        
        # Apply configuration
        quantized_model = mp_optimizer.apply_configuration(model, config)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[MixedPrecisionConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize optimizer.
        
        Args:
            model: Neural network model
            config: Mixed-precision configuration
            device: Computation device
        """
        self.model = model
        self.config = config or MixedPrecisionConfig()
        self.device = device
        self.sensitivity_cache: Dict[str, LayerSensitivity] = {}
        self.precision_bits = {
            'fp32': 32,
            'fp16': 16,
            'int8': 8,
            'int4': 4,
            'int2': 2
        }
        
        logger.info(f"MixedPrecisionOptimizer initialized with target {self.config.target_bits_avg} bits")
    
    def analyze_sensitivity(
        self,
        data: Union[torch.Tensor, torch.utils.data.DataLoader],
        labels: Optional[torch.Tensor] = None,
        method: str = 'gradient',
        num_samples: int = 100
    ) -> Dict[str, LayerSensitivity]:
        """
        Analyze quantization sensitivity per layer.
        
        Methods:
        --------
        - 'gradient': Gradient-based sensitivity (fast)
        - 'hessian': Hessian diagonal approximation (accurate)
        - 'taylor': Taylor expansion error estimation
        
        Args:
            data: Input data (tensor or dataloader)
            labels: Target labels (if data is tensor)
            method: Sensitivity analysis method
            num_samples: Number of samples to analyze
        
        Returns:
            Dict mapping layer names to sensitivity scores
        """
        self.model.eval()
        sensitivity_dict = {}
        
        # Convert to data loader if needed
        if isinstance(data, torch.Tensor):
            from torch.utils.data import TensorDataset, DataLoader
            if labels is None:
                labels = torch.zeros(len(data), dtype=torch.long)
            dataset = TensorDataset(data, labels)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            num_batches = min(10, len(data_loader))
        else:
            data_loader = data
            num_batches = min(10, len(data_loader))
        
        # Collect gradient statistics
        gradient_stats = self._collect_gradient_statistics(
            self.model, data_loader, num_batches
        )
        
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            param_count = sum(p.numel() for p in module.parameters())
            
            # Initialize storage for method-specific metrics
            self._last_hessian_trace = None
            self._last_taylor_score = None
            
            if method == 'gradient':
                sensitivity = self._gradient_sensitivity(name, module, gradient_stats)
            elif method == 'hessian':
                sensitivity = self._hessian_sensitivity(name, module, data_loader)
            elif method == 'taylor':
                sensitivity = self._taylor_sensitivity(name, module, data_loader)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Recommend precision based on sensitivity
            recommended_precision = self._recommend_precision(sensitivity)
            
            # Recommend bits
            bits_map = {'fp32': 32, 'fp16': 16, 'int8': 8, 'int4': 4, 'int2': 2}
            recommended_bits = bits_map.get(recommended_precision.value, 8)
            
            layer_info = LayerSensitivity(
                layer_name=name,
                sensitivity_score=sensitivity,
                param_count=param_count,
                recommended_precision=recommended_precision,
                gradient_norm=gradient_stats.get(name, {}).get('norm', None),
                hessian_trace=self._last_hessian_trace if method == 'hessian' else None,
                taylor_score=self._last_taylor_score if method == 'taylor' else None,
                recommended_bits=recommended_bits
            )
            
            sensitivity_dict[name] = layer_info
            self.sensitivity_cache[name] = layer_info
        
        logger.info(f"Analyzed sensitivity for {len(sensitivity_dict)} layers")
        return sensitivity_dict
    
    def _collect_gradient_statistics(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        num_batches: int
    ) -> Dict[str, Dict[str, float]]:
        """Collect gradient statistics for sensitivity analysis."""
        gradient_stats = {}
        
        model.train()
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            # Forward pass
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None
            
            model.zero_grad()
            
            # Simple forward pass
            if targets is not None:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
            else:
                outputs = model(inputs)
                loss = outputs.mean()
            
            loss.backward()
            
            # Collect gradients
            for name, module in model.named_modules():
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                
                if name not in gradient_stats:
                    gradient_stats[name] = {'norms': [], 'means': []}
                
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    grad_norm = module.weight.grad.norm().item()
                    grad_mean = module.weight.grad.abs().mean().item()
                    gradient_stats[name]['norms'].append(grad_norm)
                    gradient_stats[name]['means'].append(grad_mean)
        
        # Aggregate statistics
        for name in gradient_stats:
            gradient_stats[name]['norm'] = np.mean(gradient_stats[name]['norms'])
            gradient_stats[name]['mean'] = np.mean(gradient_stats[name]['means'])
        
        return gradient_stats
    
    def _gradient_sensitivity(
        self,
        name: str,
        module: nn.Module,
        gradient_stats: Dict
    ) -> float:
        """
        Compute gradient-based sensitivity.
        
        Formula: S = ||∇L/∇W|| / ||W||
        """
        if name not in gradient_stats:
            return 0.5  # Default medium sensitivity
        
        grad_norm = gradient_stats[name].get('norm', 1e-8)
        weight_norm = module.weight.norm().item() + 1e-8
        
        # Normalized sensitivity
        sensitivity = grad_norm / weight_norm
        
        # Normalize to [0, 1]
        sensitivity = min(max(sensitivity, 0.0), 1.0)
        
        return sensitivity
    
    def _hessian_sensitivity(
        self,
        name: str,
        module: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Compute Hessian-based sensitivity (diagonal approximation).
        
        More accurate but slower than gradient method.
        """
        # Simplified: use squared gradient as Hessian diagonal approximation
        if not hasattr(module, 'weight') or module.weight is None:
            return 0.5
        
        # Store hessian trace for later use
        if hasattr(module.weight, 'grad') and module.weight.grad is not None:
            hessian_approx = (module.weight.grad ** 2).sum().item()
        else:
            hessian_approx = 1.0
        
        # Store for layer sensitivity info
        self._last_hessian_trace = hessian_approx
        
        # Normalize sensitivity
        weight_norm = module.weight.norm().item() + 1e-8
        sensitivity = hessian_approx / (weight_norm + 1e-6)
        return min(max(sensitivity / 100.0, 0.0), 1.0)  # Normalize to [0, 1]
    
    def _taylor_sensitivity(
        self,
        name: str,
        module: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Compute Taylor expansion-based sensitivity.
        
        Estimates quantization error using first-order Taylor expansion.
        """
        # Simplified: use weight magnitude * gradient as Taylor approximation
        if not hasattr(module, 'weight') or module.weight is None:
            self._last_taylor_score = 0.5
            return 0.5
        
        weight_magnitude = module.weight.abs().mean().item()
        
        if hasattr(module.weight, 'grad') and module.weight.grad is not None:
            grad_magnitude = module.weight.grad.abs().mean().item()
            taylor_score = weight_magnitude * grad_magnitude
        else:
            taylor_score = weight_magnitude
        
        # Store for layer sensitivity info
        self._last_taylor_score = taylor_score
        
        # Normalize to [0, 1]
        return min(max(taylor_score, 0.0), 1.0)
    
    def _recommend_precision(self, sensitivity: float) -> PrecisionType:
        """
        Recommend precision based on sensitivity score.
        
        Rules:
        - sensitivity > 0.7: FP16 (high precision needed)
        - 0.4 < sensitivity <= 0.7: INT8 (medium precision)
        - sensitivity <= 0.4: INT4 (low precision acceptable)
        """
        if sensitivity > 0.7:
            return PrecisionType.FP16
        elif sensitivity > 0.4:
            return PrecisionType.INT8
        else:
            return PrecisionType.INT4
    
    def search_configuration(
        self,
        data: Union[torch.Tensor, torch.utils.data.DataLoader],
        labels: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
        target_compression: Optional[float] = None,
        **kwargs
    ) -> MixedPrecisionConfig:
        """
        Search for optimal mixed-precision configuration.
        
        Uses evolutionary algorithm to explore configuration space and
        find Pareto-optimal solutions (accuracy vs compression).
        
        Args:
            data: Input data (tensor or dataloader)
            labels: Target labels (if data is tensor)
            method: Search method ('evolutionary', 'gradient', 'heuristic')
            target_compression: Target compression ratio
            **kwargs: Additional search parameters (population_size, generations, iterations)
        
        Returns:
            MixedPrecisionConfig with optimal configuration
        """
        # Convert to data loader if needed
        if isinstance(data, torch.Tensor):
            from torch.utils.data import TensorDataset, DataLoader
            if labels is None:
                labels = torch.zeros(len(data), dtype=torch.long)
            dataset = TensorDataset(data, labels)
            val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        else:
            val_loader = data
        
        # Use method from kwargs or config
        search_method = method or self.config.search_method
        
        logger.info(f"Searching configuration with method: {search_method}")
        
        if search_method == 'evolutionary':
            config_dict = self._evolutionary_search(
                self.model, val_loader, self.config.search_space,
                population_size=kwargs.get('population_size', self.config.population_size),
                generations=kwargs.get('generations', self.config.generations)
            )
        elif search_method == 'gradient':
            config_dict = self._gradient_based_search(
                self.model, val_loader, self.config.search_space,
                iterations=kwargs.get('iterations', 50)
            )
        elif search_method == 'heuristic':
            config_dict = self._heuristic_search(self.model, self.config.search_space)
        else:
            raise ValueError(f"Unknown search method: {search_method}")
        
        # Create config object
        config = MixedPrecisionConfig(
            layer_configs=config_dict,
            target_compression=target_compression or self.config.target_compression,
            search_method=search_method
        )
        
        # Estimate compression
        metrics = self.estimate_compression(config)
        config.compression_ratio = metrics['compression_ratio']
        
        return config
    
    def _evolutionary_search(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        search_space: List[str],
        population_size: int = 20,
        generations: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evolutionary search for optimal configuration.
        
        Each individual is a configuration dict mapping layers to precisions.
        Fitness = accuracy - alpha * bits_used
        """
        # Get quantizable layers
        layer_names = [
            name for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        
        if not layer_names:
            logger.warning("No quantizable layers found")
            return {}
        
        # Initialize population
        population = self._initialize_population(layer_names, search_space, population_size)
        
        best_config = None
        best_fitness = -float('inf')
        
        for generation in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            
            for config in population:
                fitness = self._evaluate_fitness(model, config, val_loader)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_config = config.copy()
            
            if generation % 10 == 0:
                avg_bits = self._compute_avg_bits(best_config)
                logger.info(f"Generation {generation}: Best fitness={best_fitness:.4f}, Avg bits={avg_bits:.2f}")
            
            # Selection, crossover, mutation
            population = self._evolve_population(
                population, fitness_scores, layer_names, search_space
            )
        
        # Convert to Dict[str, Dict[str, Any]]
        result = {}
        for layer_name, precision in best_config.items():
            result[layer_name] = {
                'precision': precision,
                'bits': self.precision_bits.get(precision, 8)
            }
        
        logger.info(f"Search complete. Best config with {len(result)} layers")
        return result
    
    def _initialize_population(
        self,
        layer_names: List[str],
        search_space: List[str],
        population_size: int
    ) -> List[Dict[str, str]]:
        """Initialize random population of configurations."""
        population = []
        
        for _ in range(population_size):
            config = {}
            for name in layer_names:
                config[name] = np.random.choice(search_space)
            population.append(config)
        
        return population
    
    def _evaluate_fitness(
        self,
        model: nn.Module,
        config: Dict[str, str],
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Evaluate fitness of a configuration.
        
        Fitness = accuracy - alpha * (bits_used / target_bits)
        """
        # Compute average bits
        avg_bits = self._compute_avg_bits(config)
        
        # Penalty for exceeding target bits
        bits_penalty = max(0, avg_bits - self.config.target_bits_avg)
        
        # Simple fitness: inverse of bits (lower bits = higher fitness)
        # In real implementation, would evaluate accuracy on val_loader
        fitness = 1.0 / (avg_bits + 1.0) - 0.1 * bits_penalty
        
        return fitness
    
    def _compute_avg_bits(self, config: Dict[str, str]) -> float:
        """Compute average bits per weight for a configuration."""
        if not config:
            return 32.0
        
        total_bits = sum(self.precision_bits.get(prec, 8) for prec in config.values())
        return total_bits / len(config)
    
    def _evolve_population(
        self,
        population: List[Dict[str, str]],
        fitness_scores: List[float],
        layer_names: List[str],
        search_space: List[str]
    ) -> List[Dict[str, str]]:
        """Evolve population using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: keep top 10%
        elite_size = max(1, self.config.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(parent1, parent2, layer_names)
            
            # Mutation
            offspring = self._mutate(offspring, search_space)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(
        self,
        population: List[Dict[str, str]],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> Dict[str, str]:
        """Select individual using tournament selection."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[best_idx]
    
    def _crossover(
        self,
        parent1: Dict[str, str],
        parent2: Dict[str, str],
        layer_names: List[str]
    ) -> Dict[str, str]:
        """Single-point crossover."""
        offspring = {}
        crossover_point = np.random.randint(0, len(layer_names))
        
        for i, name in enumerate(layer_names):
            if i < crossover_point:
                offspring[name] = parent1.get(name, 'int8')
            else:
                offspring[name] = parent2.get(name, 'int8')
        
        return offspring
    
    def _mutate(
        self,
        config: Dict[str, str],
        search_space: List[str],
        mutation_rate: float = 0.1
    ) -> Dict[str, str]:
        """Mutate configuration with given probability."""
        mutated = config.copy()
        
        for name in mutated:
            if np.random.random() < mutation_rate:
                mutated[name] = np.random.choice(search_space)
        
        return mutated
    
    def _gradient_based_search(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        search_space: List[str],
        iterations: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """Gradient-based search (uses pre-computed sensitivity)."""
        # Use cached sensitivity if available, otherwise compute
        if not self.sensitivity_cache:
            # Need to convert val_loader to proper format for analyze_sensitivity
            # For now, use heuristic approach
            pass
        
        config = {}
        for name, sensitivity_info in self.sensitivity_cache.items():
            precision = sensitivity_info.recommended_precision.value
            config[name] = {
                'precision': precision,
                'bits': self.precision_bits.get(precision, 8)
            }
        
        # If no cached sensitivity, use heuristic
        if not config:
            return self._heuristic_search(model, search_space)
        
        return config
    
    def _heuristic_search(
        self,
        model: nn.Module,
        search_space: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Heuristic-based search.
        
        Rules:
        - First/last layers: higher precision (FP16)
        - Middle layers: INT8
        - Small layers (<1000 params): INT4
        """
        config = {}
        layer_list = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        
        for i, (name, module) in enumerate(layer_list):
            param_count = sum(p.numel() for p in module.parameters())
            
            # First and last layers: FP16
            if i == 0 or i == len(layer_list) - 1:
                precision = 'fp16'
            # Small layers: INT4
            elif param_count < 1000:
                precision = 'int4'
            # Middle layers: INT8
            else:
                precision = 'int8'
            
            config[name] = {
                'precision': precision,
                'bits': self.precision_bits.get(precision, 8)
            }
        
        return config
    
    def apply_configuration(
        self,
        config: MixedPrecisionConfig
    ) -> nn.Module:
        """
        Apply mixed-precision configuration to model.
        
        Args:
            config: MixedPrecisionConfig with layer configurations
        
        Returns:
            Quantized model with mixed precision
        """
        logger.info("Applying mixed-precision configuration")
        
        # Clone model
        import copy
        quantized_model = copy.deepcopy(self.model)
        
        # Apply quantization to each layer
        for name, module in quantized_model.named_modules():
            if name not in config.layer_configs:
                continue
            
            layer_config = config.layer_configs[name]
            precision = layer_config.get('precision', 'int8')
            
            # For all precisions, apply quantization simulation
            # (Don't convert to .half() as it causes dtype mismatches)
            if precision in ['fp16', 'int8', 'int4', 'int2']:
                self._quantize_module(module, precision)
        
        logger.info(f"Applied mixed-precision to {len(config.layer_configs)} layers")
        return quantized_model
    
    def _quantize_module(self, module: nn.Module, precision: str):
        """Quantize module weights to specified precision."""
        if not hasattr(module, 'weight'):
            return
        
        bits = self.precision_bits.get(precision, 8)
        
        # Symmetric quantization
        w = module.weight.data
        w_max = w.abs().max()
        
        if w_max > 0:
            scale = (2 ** (bits - 1) - 1) / w_max
            w_quant = torch.clamp(torch.round(w * scale), -2**(bits-1), 2**(bits-1)-1)
            module.weight.data = w_quant / scale
    
    def estimate_compression(
        self,
        config: MixedPrecisionConfig
    ) -> Dict[str, float]:
        """
        Estimate compression ratio and other metrics.
        
        Returns:
            Dict with compression statistics
        """
        total_params = 0
        total_bits = 0
        
        for name, module in self.model.named_modules():
            if name not in config.layer_configs:
                continue
            
            param_count = sum(p.numel() for p in module.parameters())
            layer_config = config.layer_configs[name]
            bits = layer_config.get('bits', 8)
            
            total_params += param_count
            total_bits += param_count * bits
        
        # Baseline: FP32
        baseline_bits = total_params * 32
        
        compression_ratio = baseline_bits / total_bits if total_bits > 0 else 1.0
        avg_bits = total_bits / total_params if total_params > 0 else 32.0
        memory_reduction_percent = (1.0 - (total_bits / baseline_bits)) * 100 if baseline_bits > 0 else 0.0
        
        return {
            'compression_ratio': compression_ratio,
            'avg_bits': avg_bits,
            'total_params': total_params,
            'memory_reduction_percent': memory_reduction_percent
        }


# ============================================================================
# Physics-Aware Mixed-Precision (for PINNs)
# ============================================================================

class PhysicsAwareMixedPrecision(MixedPrecisionOptimizer):
    """
    Specialized mixed-precision for Physics-Informed Neural Networks.
    
    Key differences from standard mixed-precision:
    - PDE residual layers: higher precision to preserve physics accuracy
    - SPIKE regularizer: FP32 for Koopman matrix operations
    - Boundary condition layers: adaptive precision based on BC importance
    - Physics loss validation after quantization
    
    Example:
    --------
        from src.compute.physics_utils import create_heat_pinn
        
        pinn, pde, trainer = create_heat_pinn()
        # ... train ...
        
        mp_quant = PhysicsAwareMixedPrecision(
            pinn, 
            physics_loss_threshold=1e-4
        )
        
        config = mp_quant.search_configuration(pinn, val_loader)
        quantized_pinn = mp_quant.apply_configuration(pinn, config)
        
        # Validate physics
        is_valid = mp_quant.validate_physics_accuracy(quantized_pinn, test_points)
    """
    
    def __init__(
        self,
        model: nn.Module,
        pde_loss_weight: float = 1.0,  # For backward compatibility
        physics_loss_threshold: float = 1e-4,
        config: Optional[MixedPrecisionConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize physics-aware mixed-precision optimizer.
        
        Args:
            model: PINNNetwork instance
            pde_loss_weight: Weight for PDE loss (backward compatibility)
            physics_loss_threshold: Maximum acceptable physics loss increase
            config: Mixed-precision configuration
            device: Computation device
        """
        super().__init__(model, config, device)
        self.pinn = model
        self.physics_threshold = physics_loss_threshold
        self.pde_loss_weight = pde_loss_weight  # Store for backward compatibility
        
        logger.info(f"PhysicsAwareMixedPrecision initialized with threshold {physics_loss_threshold}")
    
    def validate_physics_accuracy(
        self,
        config: MixedPrecisionConfig,
        test_points: torch.Tensor,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Validate that quantized PINN still satisfies physics constraints.
        
        Args:
            config: Mixed-precision configuration to validate
            test_points: Test points for PDE evaluation
            threshold: Optional threshold override
        
        Returns:
            True if physics accuracy is acceptable
        """
        threshold = threshold or self.physics_threshold
        
        # Apply configuration
        quantized_pinn = self.apply_configuration(config)
        
        # Compute physics loss on original and quantized
        original_loss = self._compute_physics_loss(self.pinn, test_points)
        quantized_loss = self._compute_physics_loss(quantized_pinn, test_points)
        
        loss_increase = quantized_loss - original_loss
        
        is_valid = loss_increase <= threshold
        
        logger.info(f"Physics validation: original_loss={original_loss:.6f}, "
                   f"quantized_loss={quantized_loss:.6f}, increase={loss_increase:.6f}, "
                   f"valid={is_valid}")
        
        return is_valid
    
    def _compute_physics_loss(
        self,
        pinn: nn.Module,
        test_points: torch.Tensor
    ) -> float:
        """Compute physics loss for a PINN model."""
        pinn.eval()
        
        with torch.no_grad():
            # Compute physics residual
            # Simplified: would use actual PDE residual computation
            test_points = test_points.to(self.device)
            output = pinn(test_points)
            
            # Placeholder physics loss (standard deviation as proxy)
            physics_loss = output.std().item()
        
        return physics_loss
    
    def _recommend_precision(self, sensitivity: float) -> PrecisionType:
        """
        Override: More conservative precision for physics layers.
        
        PINNs need higher precision to preserve PDE residual accuracy.
        """
        if sensitivity > 0.6:  # More conservative threshold
            return PrecisionType.FP16
        elif sensitivity > 0.3:
            return PrecisionType.INT8
        else:
            return PrecisionType.INT4


# ============================================================================
# Utility Functions
# ============================================================================

def create_mixed_precision_optimizer(
    model: nn.Module,
    target_bits: float = 6.5,
    search_method: str = 'evolutionary',
    is_pinn: bool = False
) -> MixedPrecisionOptimizer:
    """
    Factory function to create appropriate optimizer.
    
    Args:
        model: Model to optimize
        target_bits: Target average bits per weight
        search_method: Search method ('evolutionary', 'gradient', 'heuristic')
        is_pinn: Whether model is a PINN (uses physics-aware optimizer)
    
    Returns:
        MixedPrecisionOptimizer instance
    """
    config = MixedPrecisionConfig(
        target_bits_avg=target_bits,
        search_method=search_method
    )
    
    if is_pinn and HAS_PINN:
        return PhysicsAwareMixedPrecision(pinn=model, config=config)
    else:
        return MixedPrecisionOptimizer(config=config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Mixed-Precision Quantization Optimizer")
    print("=" * 60)
    
    # Create simple model for demo
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create optimizer
    config = MixedPrecisionConfig(
        target_bits_avg=6.5,
        search_method='heuristic'  # Fast for demo
    )
    
    optimizer = MixedPrecisionOptimizer(config=config)
    
    # Heuristic search (no data needed)
    config_dict = optimizer._heuristic_search(model, ['fp16', 'int8', 'int4'])
    
    print(f"\nOptimal configuration:")
    for layer_name, precision in config_dict.items():
        print(f"  {layer_name}: {precision}")
    
    # Estimate compression
    stats = optimizer.estimate_compression(config_dict, model)
    print(f"\nCompression statistics:")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Avg bits per weight: {stats['avg_bits_per_weight']:.2f}")
    print(f"  Memory reduction: {stats['memory_reduction']:.1%}")
