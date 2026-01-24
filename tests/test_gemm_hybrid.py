"""
Test Suite for Hybrid GEMM Kernel (Task 1.1.4)

Comprehensive testing including:
- Correctness validation
- Accuracy analysis  
- Performance benchmarking
- Regression testing
- Stability analysis

Classes:
    HybridGEMMTester: Main test runner
    BenchmarkResults: Results container
"""

import numpy as np
import pyopencl as cl
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from src.opencl.hybrid_gemm import HybridGEMMExecutor, HybridGEMMConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    matrix_size: int
    algorithm: str
    iterations: int
    
    # Performance metrics
    time_ms: float
    std_ms: float
    gflops: float
    std_gflops: float
    
    # Accuracy metrics
    error_abs: float
    error_rel: float
    
    # Hardware metrics
    bandwidth_percent: float
    occupancy_percent: float
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Pretty print results."""
        return (
            f"Size={self.matrix_size:4d}  "
            f"Time={self.time_ms:.3f}ms (±{self.std_ms:.3f})  "
            f"GFLOPS={self.gflops:.1f} (±{self.std_gflops:.1f})  "
            f"Error={self.error_rel:.2e}  "
            f"BW={self.bandwidth_percent:.1f}%"
        )


class HybridGEMMTester:
    """
    Comprehensive tester for hybrid GEMM kernel.
    
    Tests correctness, accuracy, performance, and stability.
    """
    
    # Hardware specs for RX 590
    RX590_SPECS = {
        'peak_gflops': 6170.0,  # 6.17 TFLOPS
        'bandwidth_gb_s': 256.0,  # 256 GB/s
        'peak_power_w': 225.0,  # TDP
    }
    
    def __init__(self, executor: Optional[HybridGEMMExecutor] = None):
        """Initialize tester."""
        self.executor = executor or HybridGEMMExecutor()
        self.results = []
        
        logger.info("HybridGEMMTester initialized")
    
    def test_correctness(self, sizes: List[int] = [128, 256, 512, 1024]) -> bool:
        """
        Test computational correctness.
        
        Verifies that results match NumPy reference implementation.
        
        Args:
            sizes: Matrix sizes to test
        
        Returns:
            True if all tests pass
        """
        logger.info("=== Correctness Testing ===\n")
        
        all_pass = True
        max_error_threshold = 1e-4
        
        for size in sizes:
            # Generate random matrices
            np.random.seed(42)  # For reproducibility
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # Compute on GPU
            C_gpu = self.executor(A, B, alpha=1.0, beta=0.0)
            
            # Compute reference on CPU
            C_ref = A @ B
            
            # Compute error
            error_abs = np.linalg.norm(C_gpu - C_ref)
            error_rel = error_abs / np.linalg.norm(C_ref)
            
            # Check bounds
            passed = error_rel < max_error_threshold
            all_pass = all_pass and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(
                f"{status} n={size:4d}: "
                f"error_abs={error_abs:.2e}, error_rel={error_rel:.2e}"
            )
        
        logger.info()
        return all_pass
    
    def test_alpha_beta(self) -> bool:
        """Test with various alpha and beta values."""
        logger.info("=== Alpha/Beta Testing ===\n")
        
        size = 512
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        C_init = np.random.randn(size, size).astype(np.float32)
        
        test_cases = [
            (1.0, 0.0),
            (2.5, 0.0),
            (1.0, 1.0),
            (2.5, 0.5),
        ]
        
        all_pass = True
        
        for alpha, beta in test_cases:
            C_gpu = self.executor(A, B, C_init.copy(), alpha=alpha, beta=beta)
            C_ref = alpha * (A @ B) + beta * C_init
            
            error_rel = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
            passed = error_rel < 1e-4
            all_pass = all_pass and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(
                f"{status} alpha={alpha:.1f}, beta={beta:.1f}: "
                f"error={error_rel:.2e}"
            )
        
        logger.info()
        return all_pass
    
    def benchmark_kernel(
        self,
        size: int = 1024,
        iterations: int = 10,
        algorithm: str = "hybrid_float4_2x2"
    ) -> BenchmarkResults:
        """
        Benchmark kernel performance.
        
        Args:
            size: Matrix size (n×n)
            iterations: Number of iterations for timing
            algorithm: Algorithm name (for logging)
        
        Returns:
            BenchmarkResults object with metrics
        """
        # Prepare matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)
        
        # Warmup
        self.executor(A, B, C.copy(), alpha=1.0, beta=0.0)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.executor(A, B, C.copy(), alpha=1.0, beta=0.0)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        times = np.array(times)
        
        # Calculate metrics
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # GFLOPS = 2*n^3 / time_in_seconds
        gflops = (2 * size**3) / (mean_time / 1000) / 1e9
        std_gflops = gflops * (std_time / mean_time)
        
        # Verify accuracy
        C_gpu = self.executor(A, B, C.copy())
        C_ref = A @ B
        error_abs = np.linalg.norm(C_gpu - C_ref)
        error_rel = error_abs / np.linalg.norm(C_ref)
        
        # Estimate hardware metrics
        bandwidth_percent = self._estimate_bandwidth_percent(size, mean_time)
        occupancy_percent = self._estimate_occupancy_percent(size)
        
        result = BenchmarkResults(
            matrix_size=size,
            algorithm=algorithm,
            iterations=iterations,
            time_ms=mean_time,
            std_ms=std_time,
            gflops=gflops,
            std_gflops=std_gflops,
            error_abs=error_abs,
            error_rel=error_rel,
            bandwidth_percent=bandwidth_percent,
            occupancy_percent=occupancy_percent,
        )
        
        self.results.append(result)
        return result
    
    def benchmark_suite(
        self,
        sizes: List[int] = None,
        iterations: int = 10
    ) -> List[BenchmarkResults]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            sizes: Matrix sizes to benchmark
            iterations: Iterations per size
        
        Returns:
            List of BenchmarkResults
        """
        if sizes is None:
            sizes = [256, 512, 1024, 2048, 4096]
        
        logger.info("=== Comprehensive Benchmarking ===\n")
        logger.info(
            f"{'Size':>6} {'Time (ms)':>12} {'GFLOPS':>10} "
            f"{'Error':>12} {'Bandwidth %':>12} {'Stability %':>12}"
        )
        logger.info("-" * 80)
        
        results = []
        for size in sizes:
            result = self.benchmark_kernel(size, iterations)
            results.append(result)
            
            stability = (result.std_ms / result.time_ms) * 100 if result.time_ms > 0 else 0
            
            logger.info(
                f"{size:6d} "
                f"{result.time_ms:7.3f} ± {result.std_ms:4.3f}  "
                f"{result.gflops:8.1f}  "
                f"{result.error_rel:11.2e}  "
                f"{result.bandwidth_percent:11.1f}%  "
                f"{stability:11.1f}%"
            )
        
        logger.info()
        return results
    
    def test_stability(
        self,
        size: int = 1024,
        iterations: int = 100,
        tolerance: float = 0.01  # 1% variance allowed
    ) -> bool:
        """
        Test performance stability.
        
        Runs many iterations and checks variance is within tolerance.
        
        Args:
            size: Matrix size
            iterations: Number of iterations
            tolerance: Max allowed coefficient of variation
        
        Returns:
            True if stable
        """
        logger.info("=== Stability Testing ===\n")
        
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.executor(A, B)
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation
        
        passed = cv <= tolerance
        status = "✅ PASS" if passed else "❌ FAIL"
        
        logger.info(
            f"{status} Stability test (n={size}, {iterations} iterations):\n"
            f"  Mean: {mean_time:.3f} ms\n"
            f"  Std:  {std_time:.3f} ms\n"
            f"  CV:   {cv*100:.2f}% (threshold: {tolerance*100:.1f}%)\n"
        )
        
        return passed
    
    def test_regression(
        self,
        baseline_gflops: float = 542.0,
        min_improvement: float = 1.0  # Don't regress below baseline
    ) -> bool:
        """
        Test against baseline performance.
        
        Args:
            baseline_gflops: Baseline GFLOPS to compare against
            min_improvement: Minimum improvement ratio (1.0 = match baseline)
        
        Returns:
            True if no regression
        """
        logger.info("=== Regression Testing ===\n")
        
        size = 1024
        result = self.benchmark_kernel(size, iterations=20)
        
        ratio = result.gflops / baseline_gflops
        passed = ratio >= min_improvement
        
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            f"{status} Regression test (n={size}):\n"
            f"  Baseline:    {baseline_gflops:.1f} GFLOPS\n"
            f"  Achieved:    {result.gflops:.1f} GFLOPS\n"
            f"  Ratio:       {ratio:.2f}x\n"
        )
        
        return passed
    
    def generate_report(self, output_file: Optional[Path] = None):
        """
        Generate testing report.
        
        Args:
            output_file: Path to save JSON report
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware': self.RX590_SPECS,
            'results': [r.to_dict() for r in self.results],
            'summary': self._generate_summary(),
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def plot_results(self, output_file: Optional[Path] = None):
        """Generate performance plots."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        sizes = [r.matrix_size for r in self.results]
        gflops = [r.gflops for r in self.results]
        errors = [r.error_rel for r in self.results]
        bw = [r.bandwidth_percent for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # GFLOPS vs size
        axes[0, 0].plot(sizes, gflops, 'o-', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=self.RX590_SPECS['peak_gflops'], color='r', linestyle='--', label='Peak')
        axes[0, 0].set_xlabel('Matrix Size')
        axes[0, 0].set_ylabel('GFLOPS')
        axes[0, 0].set_title('Performance Scaling')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Error vs size
        axes[0, 1].semilogy(sizes, errors, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].axhline(y=1e-4, color='g', linestyle='--', label='Target')
        axes[0, 1].set_xlabel('Matrix Size')
        axes[0, 1].set_ylabel('Relative Error')
        axes[0, 1].set_title('Numerical Accuracy')
        axes[0, 1].grid(True, alpha=0.3, which='both')
        axes[0, 1].legend()
        
        # Bandwidth utilization
        axes[1, 0].bar(range(len(sizes)), bw, color='steelblue', alpha=0.7)
        axes[1, 0].axhline(y=100, color='r', linestyle='--', label='Peak')
        axes[1, 0].set_xticks(range(len(sizes)))
        axes[1, 0].set_xticklabels(sizes)
        axes[1, 0].set_ylabel('Utilization %')
        axes[1, 0].set_title('Memory Bandwidth Utilization')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].legend()
        
        # Efficiency (GFLOPS per Watt estimate)
        gflops_per_watt = [g / 30 for g in gflops]  # Rough estimate
        axes[1, 1].plot(sizes, gflops_per_watt, 's-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel('Matrix Size')
        axes[1, 1].set_ylabel('GFLOPS/W (estimated)')
        axes[1, 1].set_title('Power Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"Plots saved to {output_file}")
        
        return fig
    
    def _estimate_bandwidth_percent(self, size: int, time_ms: float) -> float:
        """Estimate memory bandwidth utilization."""
        # GEMM reads 2*n^2 + n^2 = 3n^2 elements (floats) = 12n^2 bytes
        # Plus write n^2 elements = 4n^2 bytes
        # Total: ~16n^2 bytes
        total_bytes = 16 * size**2
        time_s = time_ms / 1000
        bandwidth_used = total_bytes / time_s / 1e9  # GB/s
        
        percent = (bandwidth_used / self.RX590_SPECS['bandwidth_gb_s']) * 100
        return min(percent, 100)  # Cap at 100%
    
    def _estimate_occupancy_percent(self, size: int) -> float:
        """Estimate GPU occupancy based on problem size."""
        # Rough estimate based on workgroup count
        # RX 590: 36 CUs, max ~10 workgroups per CU
        max_workgroups = 36 * 10
        tile_size = self.executor.config.tile_size
        num_workgroups = ((size + tile_size - 1) // tile_size) ** 2
        
        occupancy = min((num_workgroups / max_workgroups) * 100, 100)
        return occupancy
    
    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        gflops_values = [r.gflops for r in self.results]
        
        return {
            'min_gflops': min(gflops_values),
            'max_gflops': max(gflops_values),
            'avg_gflops': np.mean(gflops_values),
            'target_gflops': 700,  # Phase 1 target
            'target_achieved': max(gflops_values) >= 700,
        }


def run_full_test_suite():
    """Execute complete test suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("HYBRID GEMM KERNEL - FULL TEST SUITE")
    logger.info("=" * 80 + "\n")
    
    # Create executor
    executor = HybridGEMMExecutor()
    tester = HybridGEMMTester(executor)
    
    # Run tests
    correctness_pass = tester.test_correctness([256, 512, 1024])
    alpha_beta_pass = tester.test_alpha_beta()
    stability_pass = tester.test_stability(iterations=100)
    regression_pass = tester.test_regression(baseline_gflops=542)
    
    # Run benchmarks
    results = tester.benchmark_suite(
        sizes=[256, 512, 1024, 2048, 4096],
        iterations=10
    )
    
    # Generate outputs
    report_file = Path("test_results/hybrid_gemm_report.json")
    plot_file = Path("test_results/hybrid_gemm_plots.png")
    
    report_file.parent.mkdir(exist_ok=True)
    
    tester.generate_report(report_file)
    tester.plot_results(plot_file)
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Correctness:  {'✅ PASS' if correctness_pass else '❌ FAIL'}")
    logger.info(f"Alpha/Beta:   {'✅ PASS' if alpha_beta_pass else '❌ FAIL'}")
    logger.info(f"Stability:    {'✅ PASS' if stability_pass else '❌ FAIL'}")
    logger.info(f"Regression:   {'✅ PASS' if regression_pass else '❌ FAIL'}")
    logger.info(f"\nTarget GFLOPS (n=1024): {max([r.gflops for r in results if r.matrix_size == 1024]):.1f}")
    logger.info(f"Target Phase 1:  700-800 GFLOPS")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_test_suite()
