#!/usr/bin/env python3
"""
GCN 4.0 Architecture-Aware GEMM Benchmark Suite
Tests Polaris 10 specific optimizations

Target: +5-10% performance improvement (285 → 300-315 GFLOPS)
Hardware: AMD Radeon RX 590 (Polaris 10)
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from opencl.gemm_gcn4_optimized import GCN4Config, GCN4GEMMExecutor
from opencl.gemm_vectorized import VectorizedGEMMExecutor
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCN4BenchmarkResult:
    """Results from a GCN 4.0 benchmark run."""

    def __init__(self, matrix_size: str, kernel_name: str, avg_time_ms: float,
                 min_time_ms: float, max_time_ms: float, std_time_ms: float,
                 gflops: float, accuracy_error: float = 0.0):
        self.matrix_size = matrix_size
        self.kernel_name = kernel_name
        self.avg_time_ms = avg_time_ms
        self.min_time_ms = min_time_ms
        self.max_time_ms = max_time_ms
        self.std_time_ms = std_time_ms
        self.gflops = gflops
        self.accuracy_error = accuracy_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'std_time_ms': self.std_time_ms,
            'gflops': self.gflops,
            'matrix_size': self.matrix_size,
            'kernel': self.kernel_name,
            'accuracy_error': self.accuracy_error
        }

class GCN4BenchmarkSuite:
    """Comprehensive benchmarking for GCN 4.0 GEMM."""

    def __init__(self, config: GCN4Config = None):
        self.config = config or GCN4Config()
        self.gcn4_executor = GCN4GEMMExecutor(self.config)
        self.vectorized_executor = VectorizedGEMMExecutor()  # For comparison
        self.results: List[GCN4BenchmarkResult] = []

        # Acceptance criteria for GCN 4.0 optimizations
        self.acceptance_criteria = {
            'min_improvement': 0.02,     # 2% minimum improvement over vectorized
            'max_error': 1e-4,          # Maintain accuracy
            'max_cv': 0.05             # Max coefficient of variation
        }

    def run_single_benchmark(self, size: int, num_runs: int = 5,
                           warmup: int = 3) -> Dict[str, GCN4BenchmarkResult]:
        """Run benchmark comparing GCN4 vs Vectorized kernels."""
        logger.info(f"Benchmarking GCN4 vs Vectorized, size {size}×{size}")

        # Create test matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Reference result (NumPy)
        C_ref = A @ B

        results = {}

        # Test GCN 4.0 optimized kernel
        logger.info("  Testing GCN 4.0 optimized kernel...")
        times_gcn4 = []
        errors_gcn4 = []

        # Warmup
        for _ in range(warmup):
            _ = self.gcn4_executor.gemm(A, B)

        # Benchmark runs
        for _ in range(num_runs):
            C_gcn4, exec_time = self.gcn4_executor.gemm(A, B)
            times_gcn4.append(exec_time)

            # Check accuracy
            error = np.max(np.abs(C_gcn4 - C_ref)) / np.max(np.abs(C_ref))
            errors_gcn4.append(error)

        # Statistics for GCN4
        avg_time_gcn4 = np.mean(times_gcn4)
        gflops_gcn4 = (2 * size**3) / (avg_time_gcn4 * 1e-3) / 1e9

        result_gcn4 = GCN4BenchmarkResult(
            matrix_size=f"{size}x{size}x{size}",
            kernel_name="gemm_gcn4_optimized",
            avg_time_ms=avg_time_gcn4,
            min_time_ms=np.min(times_gcn4),
            max_time_ms=np.max(times_gcn4),
            std_time_ms=np.std(times_gcn4),
            gflops=gflops_gcn4,
            accuracy_error=np.mean(errors_gcn4)
        )

        # Test vectorized kernel for comparison
        logger.info("  Testing SIMD vectorized kernel (baseline)...")
        times_vec = []
        errors_vec = []

        # Warmup
        for _ in range(warmup):
            _ = self.vectorized_executor.gemm(A, B)

        # Benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            C_vec = self.vectorized_executor.gemm(A, B)
            end_time = time.time()
            exec_time = (end_time - start_time) * 1000  # Convert to ms
            times_vec.append(exec_time)

            # Check accuracy
            error = np.max(np.abs(C_vec - C_ref)) / np.max(np.abs(C_ref))
            errors_vec.append(error)

        # Statistics for Vectorized
        avg_time_vec = np.mean(times_vec)
        gflops_vec = (2 * size**3) / (avg_time_vec * 1e-3) / 1e9

        result_vec = GCN4BenchmarkResult(
            matrix_size=f"{size}x{size}x{size}",
            kernel_name="gemm_vectorized_float4",
            avg_time_ms=avg_time_vec,
            min_time_ms=np.min(times_vec),
            max_time_ms=np.max(times_vec),
            std_time_ms=np.std(times_vec),
            gflops=gflops_vec,
            accuracy_error=np.mean(errors_vec)
        )

        results['gcn4'] = result_gcn4
        results['vectorized'] = result_vec

        # Calculate improvement
        improvement = (gflops_gcn4 - gflops_vec) / gflops_vec
        cv_gcn4 = np.std(times_gcn4) / avg_time_gcn4 if avg_time_gcn4 > 0 else float('inf')

        # Acceptance check
        passed = (improvement >= self.acceptance_criteria['min_improvement'] and
                 np.mean(errors_gcn4) <= self.acceptance_criteria['max_error'] and
                 cv_gcn4 <= self.acceptance_criteria['max_cv'])

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} | GCN4: {gflops_gcn4:.1f} GFLOPS | Vec: {gflops_vec:.1f} GFLOPS | Improvement: {improvement:+.1%}")

        self.results.extend([result_gcn4, result_vec])
        return results

    def run_full_suite(self, sizes: List[int] = None, num_runs: int = 5) -> Dict[str, Any]:
        """Run complete GCN 4.0 benchmark suite."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]

        logger.info("=" * 80)
        logger.info("GCN 4.0 ARCHITECTURE-AWARE GEMM BENCHMARK SUITE - PHASE 4, TECHNIQUE 3")
        logger.info("=" * 80)
        logger.info(f"Hardware: AMD Radeon RX 590 (Polaris 10)")
        logger.info(f"Target: +5-10% improvement (285 → 300-315 GFLOPS)")
        logger.info(f"Strategy: GCN 4.0 ISA-specific optimizations")
        logger.info("")

        self.results = []
        results_dict = {}

        for size in sizes:
            size_results = self.run_single_benchmark(size, num_runs)
            results_dict[f"{size}x{size}x{size}"] = {
                'gcn4': size_results['gcn4'].to_dict(),
                'vectorized': size_results['vectorized'].to_dict()
            }

        # Summary statistics
        gcn4_gflops = [r.gflops for r in self.results if r.kernel_name == 'gemm_gcn4_optimized']
        vec_gflops = [r.gflops for r in self.results if r.kernel_name == 'gemm_vectorized_float4']

        if gcn4_gflops and vec_gflops:
            peak_gcn4 = max(gcn4_gflops)
            peak_vec = max(vec_gflops)
            avg_improvement = np.mean([(g - v) / v for g, v in zip(gcn4_gflops, vec_gflops)])

            logger.info("")
            logger.info("SUMMARY STATISTICS:")
            logger.info(f"  GCN4 Peak Performance: {peak_gcn4:.1f} GFLOPS")
            logger.info(f"  Vectorized Peak Performance: {peak_vec:.1f} GFLOPS")
            logger.info(f"  Average Improvement: {avg_improvement:+.1%}")
            logger.info(f"  Target Achievement: {(peak_gcn4 / 300 - 1) * 100:+.1f}% vs 300 GFLOPS target")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"gcn4_optimized_benchmark_{timestamp}.json")

        benchmark_data = {
            "benchmark_info": {
                "phase": "Phase 4, Technique 3",
                "technique": "GCN 4.0 Architecture-Aware Optimization",
                "hardware": "AMD Radeon RX 590 (Polaris 10)",
                "timestamp": datetime.now().isoformat(),
                "objective": "+5-10% performance improvement with ISA-specific optimizations",
                "target_gflops": 315
            },
            "results": results_dict,
            "summary": {
                "peak_gcn4_gflops": peak_gcn4 if 'peak_gcn4' in locals() else 0,
                "peak_vectorized_gflops": peak_vec if 'peak_vec' in locals() else 0,
                "avg_improvement": avg_improvement if 'avg_improvement' in locals() else 0,
                "target_achievement": f"{(peak_gcn4/300-1)*100:.1f}%" if 'peak_gcn4' in locals() else "N/A",
                "acceptance_criteria": self.acceptance_criteria
            }
        }

        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(benchmark_data, f, indent=2, default=convert_numpy_types)

        logger.info(f"Results saved to: {results_file}")
        return benchmark_data

def main():
    """Main benchmark execution."""
    suite = GCN4BenchmarkSuite()
    summary = suite.run_full_suite()

    # Print final assessment
    if 'summary' in summary and 'peak_gcn4_gflops' in summary['summary']:
        peak_gcn4 = summary['summary']['peak_gcn4_gflops']
        target = 315
        improvement = (peak_gcn4 / 285 - 1) * 100

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL ASSESSMENT")
        logger.info("=" * 80)
        logger.info(f"Target: {target} GFLOPS (+10.5% over 285 GFLOPS baseline)")
        logger.info(f"Achieved: {peak_gcn4:.1f} GFLOPS ({improvement:+.1f}% vs baseline)")

        if peak_gcn4 >= target:
            logger.info("🎉 SUCCESS: GCN 4.0 optimization target achieved!")
        elif peak_gcn4 >= 300:
            logger.info("✅ GOOD: Solid progress toward target")
        else:
            logger.info("⚠️  MODERATE: Some improvement, needs refinement")
    else:
        logger.info("❌ ERROR: Could not calculate performance metrics")

    return summary

if __name__ == "__main__":
    main()