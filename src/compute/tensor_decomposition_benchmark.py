"""
Tensor Decomposition Benchmarking Suite - Session 25
===================================================

Comprehensive benchmarking for tensor decomposition methods.
Evaluates compression-accuracy-speed trade-offs on real models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from src.compute.tensor_decomposition import (
    decompose_model,
    DecompositionConfig,
    compute_compression_ratio
)
from src.compute.tensor_decomposition_finetuning import (
    DecompositionFinetuner,
    FinetuneConfig
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    
    # Methods to benchmark
    methods: List[str] = field(default_factory=lambda: ["tucker", "cp", "tt"])
    
    # Rank configurations for each method
    tucker_ranks: List[List[int]] = field(default_factory=lambda: [
        [32, 64], [16, 32], [8, 16], [4, 8]
    ])
    cp_ranks: List[int] = field(default_factory=lambda: [64, 32, 16, 8])
    tt_ranks: List[List[int]] = field(default_factory=lambda: [
        [32, 64], [16, 32], [8, 16]
    ])
    
    # Fine-tuning settings
    finetune: bool = True
    finetune_epochs: int = 5
    
    # Evaluation settings
    num_inference_runs: int = 100
    warmup_runs: int = 10
    
    # Output settings
    save_results: bool = True
    results_dir: str = "benchmark_results"


@dataclass
class BenchmarkResult:
    """Results for a single configuration."""
    
    method: str
    ranks: List[int]
    
    # Model metrics
    original_params: int
    compressed_params: int
    compression_ratio: float
    
    # Accuracy metrics
    baseline_accuracy: float
    compressed_accuracy: float
    finetuned_accuracy: Optional[float] = None
    
    # Speed metrics
    baseline_time_ms: float = 0.0
    compressed_time_ms: float = 0.0
    speedup: float = 1.0
    
    # Memory metrics
    baseline_memory_mb: float = 0.0
    compressed_memory_mb: float = 0.0
    memory_reduction: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'ranks': self.ranks,
            'original_params': self.original_params,
            'compressed_params': self.compressed_params,
            'compression_ratio': self.compression_ratio,
            'baseline_accuracy': self.baseline_accuracy,
            'compressed_accuracy': self.compressed_accuracy,
            'finetuned_accuracy': self.finetuned_accuracy,
            'baseline_time_ms': self.baseline_time_ms,
            'compressed_time_ms': self.compressed_time_ms,
            'speedup': self.speedup,
            'baseline_memory_mb': self.baseline_memory_mb,
            'compressed_memory_mb': self.compressed_memory_mb,
            'memory_reduction': self.memory_reduction
        }


class TensorDecompositionBenchmark:
    """
    Benchmarking suite for tensor decomposition methods.
    
    Evaluates:
    - Compression ratios
    - Accuracy before/after fine-tuning
    - Inference speed
    - Memory usage
    - Pareto frontiers (compression vs accuracy)
    
    Example:
        >>> benchmark = TensorDecompositionBenchmark(config)
        >>> results = benchmark.benchmark_model(
        ...     model, train_loader, val_loader
        ... )
        >>> benchmark.print_summary(results)
        >>> benchmark.save_results(results, "resnet_benchmark.json")
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        
        # Create results directory
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def benchmark_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "model"
    ) -> List[BenchmarkResult]:
        """
        Benchmark a model with all configurations.
        
        Args:
            model: PyTorch model to benchmark
            train_loader: Training data loader (for fine-tuning)
            val_loader: Validation data loader (for evaluation)
            model_name: Name of the model (for reporting)
        
        Returns:
            List of benchmark results for each configuration
        """
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}\n")
        
        # Evaluate baseline
        print("📊 Evaluating baseline model...")
        baseline_acc = self._evaluate_accuracy(model, val_loader)
        baseline_time = self._measure_inference_time(model, val_loader)
        baseline_memory = self._estimate_memory(model)
        baseline_params = sum(p.numel() for p in model.parameters())
        
        print(f"   Accuracy: {baseline_acc:.2f}%")
        print(f"   Inference: {baseline_time:.2f}ms")
        print(f"   Memory: {baseline_memory:.2f}MB")
        print(f"   Params: {baseline_params:,}")
        
        results = []
        
        # Benchmark each method
        for method in self.config.methods:
            print(f"\n{'─'*70}")
            print(f"Method: {method.upper()}")
            print(f"{'─'*70}")
            
            # Get rank configurations for this method
            if method == "tucker":
                rank_configs = self.config.tucker_ranks
            elif method == "cp":
                rank_configs = [[r] for r in self.config.cp_ranks]
            elif method == "tt":
                rank_configs = self.config.tt_ranks
            else:
                print(f"⚠️  Unknown method: {method}")
                continue
            
            # Benchmark each rank configuration
            for ranks in rank_configs:
                try:
                    result = self._benchmark_configuration(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        method=method,
                        ranks=ranks,
                        baseline_acc=baseline_acc,
                        baseline_time=baseline_time,
                        baseline_memory=baseline_memory,
                        baseline_params=baseline_params
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ Error with {method} ranks={ranks}: {e}")
        
        return results
    
    def _benchmark_configuration(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        method: str,
        ranks: List[int],
        baseline_acc: float,
        baseline_time: float,
        baseline_memory: float,
        baseline_params: int
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        
        print(f"\n🔧 {method.upper()} ranks={ranks}")
        
        # Compress
        decomp_config = DecompositionConfig(
            method=method,
            ranks=ranks,
            auto_rank=False
        )
        
        # Create fresh copy of model
        compressed = self._copy_model(model)
        compressed = decompose_model(compressed, decomp_config)
        
        compressed_params = sum(p.numel() for p in compressed.parameters())
        compression_ratio = baseline_params / compressed_params
        
        # Evaluate compressed (before fine-tuning)
        compressed_acc = self._evaluate_accuracy(compressed, val_loader)
        compressed_time = self._measure_inference_time(compressed, val_loader)
        compressed_memory = self._estimate_memory(compressed)
        
        print(f"   Compression: {compression_ratio:.1f}x ({compressed_params:,} params)")
        print(f"   Accuracy (before FT): {compressed_acc:.2f}%")
        
        # Fine-tune if enabled
        finetuned_acc = None
        if self.config.finetune:
            print(f"   Fine-tuning ({self.config.finetune_epochs} epochs)...")
            
            finetune_config = FinetuneConfig(
                epochs=self.config.finetune_epochs,
                verbose=False,
                early_stopping=True,
                patience=3
            )
            finetuner = DecompositionFinetuner(finetune_config)
            criterion = nn.CrossEntropyLoss()
            
            compressed, metrics = finetuner.fine_tune(
                decomposed_model=compressed,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion
            )
            
            finetuned_acc = metrics['final_val_accuracy']
            print(f"   Accuracy (after FT): {finetuned_acc:.2f}%")
            print(f"   Recovered: {finetuned_acc - compressed_acc:+.2f}%")
        
        speedup = baseline_time / compressed_time
        memory_reduction = baseline_memory / compressed_memory
        
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Memory reduction: {memory_reduction:.2f}x")
        
        return BenchmarkResult(
            method=method,
            ranks=ranks,
            original_params=baseline_params,
            compressed_params=compressed_params,
            compression_ratio=compression_ratio,
            baseline_accuracy=baseline_acc,
            compressed_accuracy=compressed_acc,
            finetuned_accuracy=finetuned_acc,
            baseline_time_ms=baseline_time,
            compressed_time_ms=compressed_time,
            speedup=speedup,
            baseline_memory_mb=baseline_memory,
            compressed_memory_mb=compressed_memory,
            memory_reduction=memory_reduction
        )
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def _evaluate_accuracy(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()
        
        # Get a batch for testing
        data, _ = next(iter(data_loader))
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(data)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(self.config.num_inference_runs):
                start = time.perf_counter()
                _ = model(data)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        return sum(times) / len(times)
    
    def _estimate_memory(self, model: nn.Module) -> float:
        """Estimate model memory in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 (4 bytes per parameter)
        memory_bytes = total_params * 4
        memory_mb = memory_bytes / (1024 * 1024)
        return memory_mb
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary table."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}\n")
        
        # Header
        print(f"{'Method':<10} {'Ranks':<15} {'Comp':<8} {'Acc%':<8} {'FT Acc%':<8} {'Speed':<8}")
        print(f"{'-'*70}")
        
        # Results
        for r in results:
            ranks_str = str(r.ranks)
            ft_acc = f"{r.finetuned_accuracy:.2f}" if r.finetuned_accuracy else "N/A"
            
            print(
                f"{r.method:<10} "
                f"{ranks_str:<15} "
                f"{r.compression_ratio:<7.1f}x "
                f"{r.compressed_accuracy:<8.2f} "
                f"{ft_acc:<8} "
                f"{r.speedup:<7.2f}x"
            )
        
        print()
    
    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save results to JSON file."""
        output_path = Path(self.config.results_dir) / filename
        
        results_dict = {
            'results': [r.to_dict() for r in results],
            'config': {
                'methods': self.config.methods,
                'finetune': self.config.finetune,
                'finetune_epochs': self.config.finetune_epochs
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def find_pareto_frontier(
        self,
        results: List[BenchmarkResult],
        use_finetuned: bool = True
    ) -> List[BenchmarkResult]:
        """
        Find Pareto frontier: configurations that are not dominated.
        
        A configuration dominates another if it has:
        - Higher or equal accuracy
        - Higher or equal compression ratio
        And at least one is strictly better.
        
        Args:
            results: List of benchmark results
            use_finetuned: Use fine-tuned accuracy if available
        
        Returns:
            List of Pareto-optimal configurations
        """
        pareto = []
        
        for r1 in results:
            acc1 = r1.finetuned_accuracy if (use_finetuned and r1.finetuned_accuracy) else r1.compressed_accuracy
            
            is_dominated = False
            for r2 in results:
                if r1 is r2:
                    continue
                
                acc2 = r2.finetuned_accuracy if (use_finetuned and r2.finetuned_accuracy) else r2.compressed_accuracy
                
                # r2 dominates r1 if:
                # - r2 has >= accuracy AND >= compression
                # - At least one is strictly better
                if (acc2 >= acc1 and r2.compression_ratio >= r1.compression_ratio and
                    (acc2 > acc1 or r2.compression_ratio > r1.compression_ratio)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(r1)
        
        # Sort by compression ratio
        pareto.sort(key=lambda r: r.compression_ratio)
        
        return pareto
    
    def print_pareto_frontier(self, results: List[BenchmarkResult]):
        """Print Pareto-optimal configurations."""
        pareto = self.find_pareto_frontier(results)
        
        print(f"\n{'='*70}")
        print("PARETO FRONTIER (Non-dominated configurations)")
        print(f"{'='*70}\n")
        
        print(f"{'Method':<10} {'Ranks':<15} {'Compression':<12} {'Accuracy':<10}")
        print(f"{'-'*70}")
        
        for r in pareto:
            acc = r.finetuned_accuracy if r.finetuned_accuracy else r.compressed_accuracy
            ranks_str = str(r.ranks)
            
            print(
                f"{r.method:<10} "
                f"{ranks_str:<15} "
                f"{r.compression_ratio:<11.1f}x "
                f"{acc:<10.2f}%"
            )
        
        print()
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        model_name: str = "Model"
    ) -> str:
        """Generate detailed report."""
        report = []
        report.append("=" * 70)
        report.append(f"TENSOR DECOMPOSITION BENCHMARK REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 70)
        report.append("")
        
        # Best by metric
        best_compression = max(results, key=lambda r: r.compression_ratio)
        best_accuracy = max(
            results,
            key=lambda r: r.finetuned_accuracy if r.finetuned_accuracy else r.compressed_accuracy
        )
        best_speedup = max(results, key=lambda r: r.speedup)
        
        report.append("HIGHLIGHTS")
        report.append("-" * 70)
        report.append(f"Best Compression: {best_compression.method} {best_compression.ranks}")
        report.append(f"  {best_compression.compression_ratio:.1f}x compression")
        report.append("")
        report.append(f"Best Accuracy: {best_accuracy.method} {best_accuracy.ranks}")
        acc = best_accuracy.finetuned_accuracy if best_accuracy.finetuned_accuracy else best_accuracy.compressed_accuracy
        report.append(f"  {acc:.2f}% accuracy")
        report.append("")
        report.append(f"Best Speedup: {best_speedup.method} {best_speedup.ranks}")
        report.append(f"  {best_speedup.speedup:.2f}x faster")
        report.append("")
        
        # Pareto frontier
        pareto = self.find_pareto_frontier(results)
        report.append("PARETO FRONTIER")
        report.append("-" * 70)
        report.append(f"Found {len(pareto)} non-dominated configurations:")
        for r in pareto:
            acc = r.finetuned_accuracy if r.finetuned_accuracy else r.compressed_accuracy
            report.append(f"  {r.method} {r.ranks}: {r.compression_ratio:.1f}x, {acc:.2f}%")
        report.append("")
        
        return "\n".join(report)


def quick_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "model",
    save_results: bool = True
) -> List[BenchmarkResult]:
    """
    Quick benchmark with default settings.
    
    Args:
        model: Model to benchmark
        train_loader: Training data
        val_loader: Validation data
        model_name: Model name for reporting
        save_results: Whether to save results
    
    Returns:
        List of benchmark results
    """
    config = BenchmarkConfig(save_results=save_results)
    benchmark = TensorDecompositionBenchmark(config)
    
    results = benchmark.benchmark_model(
        model, train_loader, val_loader, model_name
    )
    
    benchmark.print_summary(results)
    benchmark.print_pareto_frontier(results)
    
    if save_results:
        benchmark.save_results(results, f"{model_name}_benchmark.json")
    
    return results
