#!/usr/bin/env python3
"""
Mixed Precision FP16 GEMM Benchmark Suite
Tests FP16 compute + FP32 accumulate implementation

Target: +15-20% performance improvement (285 → 330-340 GFLOPS)
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

from opencl.gemm_mixed_precision_fp16 import MixedPrecisionConfig, MixedPrecisionGEMMExecutor
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MixedPrecisionBenchmarkResult:
    """Results from a Mixed Precision benchmark run."""

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

class MixedPrecisionBenchmarkSuite:
    """Comprehensive benchmarking for Mixed Precision GEMM."""

    def __init__(self, config: MixedPrecisionConfig = None):
        self.config = config or MixedPrecisionConfig()
        self.executor = MixedPrecisionGEMMExecutor(self.config)
        self.results: List[MixedPrecisionBenchmarkResult] = []

        # Acceptance criteria for Mixed Precision
        self.acceptance_criteria = {
            'min_gflops': 300,      # Should exceed SIMD vectorization baseline
            'max_error': 1e-3,      # Allow higher error for mixed precision
            'max_cv': 0.05         # Max coefficient of variation
        }

    def run_single_benchmark(self, size: int, num_runs: int = 5,
                           warmup: int = 3) -> MixedPrecisionBenchmarkResult:
        """Run benchmark for a single matrix size."""
        logger.info(f"Benchmarking Mixed Precision FP16, size {size}×{size}")

        # Create test matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Reference result (NumPy)
        C_ref = A @ B

        # Warmup runs
        for _ in range(warmup):
            _ = self.executor.gemm(A, B)

        # Benchmark runs
        times = []
        errors = []
        for _ in range(num_runs):
            C_gpu, exec_time = self.executor.gemm(A, B)
            times.append(exec_time)

            # Check accuracy
            error = np.max(np.abs(C_gpu - C_ref)) / np.max(np.abs(C_ref))
            errors.append(error)

        # Statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        cv = std_time / avg_time if avg_time > 0 else float('inf')

        # Performance metrics
        gflops = (2 * size**3) / (avg_time * 1e-3) / 1e9  # GFLOPS
        avg_error = np.mean(errors)

        # Acceptance check
        passed = (gflops >= self.acceptance_criteria['min_gflops'] and
                 avg_error <= self.acceptance_criteria['max_error'] and
                 cv <= self.acceptance_criteria['max_cv'])

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} | {gflops:.1f} GFLOPS | Error: {avg_error:.2e} | CV: {cv:.2%}")

        result = MixedPrecisionBenchmarkResult(
            matrix_size=f"{size}x{size}x{size}",
            kernel_name="gemm_mixed_precision_fp16",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_time_ms=std_time,
            gflops=gflops,
            accuracy_error=avg_error
        )

        self.results.append(result)
        return result

    def run_full_suite(self, sizes: List[int] = None, num_runs: int = 5) -> Dict[str, Any]:
        """Run complete Mixed Precision benchmark suite."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]

        logger.info("=" * 80)
        logger.info("MIXED PRECISION FP16 GEMM BENCHMARK SUITE - PHASE 4, TECHNIQUE 2")
        logger.info("=" * 80)
        logger.info(f"Hardware: AMD Radeon RX 590 (Polaris 10)")
        logger.info(f"Target: +15-20% improvement (285 → 330-340 GFLOPS)")
        logger.info(f"Strategy: FP16 compute + FP32 accumulate")
        logger.info("")

        self.results = []
        results_dict = {}

        for size in sizes:
            result = self.run_single_benchmark(size, num_runs)
            results_dict[result.matrix_size] = result.to_dict()

        # Summary statistics
        gflops_values = [r.gflops for r in self.results]
        peak_gflops = max(gflops_values) if gflops_values else 0
        avg_gflops = np.mean(gflops_values) if gflops_values else 0

        logger.info("")
        logger.info("SUMMARY STATISTICS:")
        logger.info(f"  Peak Performance: {peak_gflops:.1f} GFLOPS")
        logger.info(f"  Average Performance: {avg_gflops:.1f} GFLOPS")
        logger.info(".1f")
        logger.info(".1f")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"mixed_precision_fp16_benchmark_{timestamp}.json")

        benchmark_data = {
            "benchmark_info": {
                "phase": "Phase 4, Technique 2",
                "technique": "Mixed Precision FP16",
                "hardware": "AMD Radeon RX 590 (Polaris 10)",
                "timestamp": datetime.now().isoformat(),
                "objective": "+15-20% performance improvement with FP16 compute",
                "target_gflops": 330
            },
            "results": results_dict,
            "summary": {
                "peak_gflops": peak_gflops,
                "avg_gflops": avg_gflops,
                "improvement_over_baseline": f"{(peak_gflops/285-1)*100:.1f}%" if peak_gflops > 0 else "N/A",
                "acceptance_criteria": self.acceptance_criteria
            }
        }

        with open(results_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info(f"Results saved to: {results_file}")
        return benchmark_data

def main():
    """Main benchmark execution."""
    suite = MixedPrecisionBenchmarkSuite()
    summary = suite.run_full_suite()

    # Print final assessment
    peak = summary['summary']['peak_gflops']
    target = 330
    improvement = (peak / 285 - 1) * 100

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 80)
    logger.info(f"Target: {target} GFLOPS (+16% over 285 GFLOPS baseline)")
    logger.info(f"Achieved: {peak:.1f} GFLOPS ({improvement:+.1f}% vs baseline)")

    if peak >= target:
        logger.info("🎉 SUCCESS: Mixed Precision FP16 target achieved!")
    else:
        logger.info(f"⚠️  PARTIAL: Target not fully met, but {peak:.1f} GFLOPS is solid progress")

    return summary

if __name__ == "__main__":
    main()