#!/usr/bin/env python3

"""
Strassen GEMM Benchmark Suite

Phase 2, Technique 4: Advanced Algorithm Research
Evaluates theoretical O(n^2.807) vs practical GPU performance

Tests both complete and simple Strassen implementations
Compares against Phase 1 baseline and other techniques
"""

import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from opencl.gemm_strassen_wrapper import StrassenConfig, StrassenGEMMExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StrassenBenchmarkResult:
    """Results from a Strassen benchmark run."""
    kernel_variant: str
    matrix_size: int
    iterations: int
    time_mean_ms: float
    time_std_ms: float
    gflops: float
    relative_error: float
    cv_percent: float
    memory_mb: float
    passed: bool


class StrassenBenchmarkSuite:
    """Comprehensive benchmarking for Strassen GEMM."""

    # Phase 1 baseline for comparison
    PHASE1_BASELINE = {
        256: 720.0,
        512: 750.0,
        1024: 775.3,
        2048: 780.0,
        4096: 770.0
    }

    # Acceptance criteria for Strassen
    ACCEPTANCE_CRITERIA = {
        'complete': {
            'target_gflops': 400.0,  # Theoretical maximum
            'min_gflops': 300.0,     # Minimum acceptable
            'max_error': 1e-4,       # Strassen can have higher error
            'max_cv_percent': 10.0
        },
        'simple': {
            'target_gflops': 350.0,
            'min_gflops': 250.0,
            'max_error': 1e-3,
            'max_cv_percent': 10.0
        }
    }

    def __init__(self):
        self.results: List[StrassenBenchmarkResult] = []

    def run_single_benchmark(self,
                            variant: str,
                            size: int,
                            iterations: int = 10,
                            warmup: int = 3) -> StrassenBenchmarkResult:
        """Run benchmark for a single Strassen configuration."""

        logger.info(f"Benchmarking Strassen {variant} variant, size {size}×{size}")

        # Create executor
        config = StrassenConfig(kernel_variant=variant)
        executor = StrassenGEMMExecutor(config)

        # Generate test matrices
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Reference result
        logger.debug("Computing reference result...")
        C_ref = A @ B

        # Warmup
        logger.debug(f"Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            _ = executor.gemm(A, B)

        # Benchmark
        logger.debug(f"Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            C = executor.gemm(A, B)
            end = time.perf_counter()
            times.append(end - start)

            if (i + 1) % 3 == 0:
                logger.debug(f"  Iteration {i+1}/{iterations}")

        # Calculate metrics
        times = np.array(times)
        time_mean = np.mean(times)
        time_std = np.std(times)

        flops = 2 * size**3
        gflops = flops / time_mean / 1e9

        # Numerical accuracy (allow higher error for Strassen)
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)

        cv_percent = (time_std / time_mean) * 100
        memory_mb = (3 * size**2 * 4) / (1024**2)

        # Check criteria
        criteria = self.ACCEPTANCE_CRITERIA[variant]
        passed = (
            gflops >= criteria['min_gflops'] and
            error <= criteria['max_error'] and
            cv_percent <= criteria['max_cv_percent']
        )

        result = StrassenBenchmarkResult(
            kernel_variant=variant,
            matrix_size=size,
            iterations=iterations,
            time_mean_ms=time_mean * 1000,
            time_std_ms=time_std * 1000,
            gflops=gflops,
            relative_error=error,
            cv_percent=cv_percent,
            memory_mb=memory_mb,
            passed=passed
        )

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} | {gflops:.1f} GFLOPS | Error: {error:.2e} | CV: {cv_percent:.2f}%")

        return result

    def run_full_suite(self,
                      variants: List[str] = None,
                      sizes: List[int] = None) -> Dict:
        """Run complete Strassen benchmark suite."""

        if variants is None:
            variants = ['simple', 'complete']

        if sizes is None:
            sizes = [256, 512, 1024, 2048]

        logger.info("=" * 80)
        logger.info("STRASSEN GEMM BENCHMARK SUITE - PHASE 2, TECHNIQUE 4")
        logger.info("=" * 80)

        summary = {
            'results': [],
            'acceptance_status': {},
            'all_variants_passed': False
        }

        for variant in variants:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing Strassen variant: {variant.upper()}")
            logger.info(f"{'='*80}")

            variant_results = []
            variant_passed = True

            for size in sizes:
                result = self.run_single_benchmark(variant, size)
                variant_results.append(result)
                variant_passed &= result.passed

            # Calculate summary statistics
            gflops_values = [r.gflops for r in variant_results]
            peak_result = max(variant_results, key=lambda r: r.gflops)
            avg_gflops = np.mean(gflops_values)

            # Improvement vs Phase 1
            phase1_avg = np.mean([self.PHASE1_BASELINE.get(r.matrix_size, 775.0)
                                 for r in variant_results])
            improvement = (avg_gflops - phase1_avg) / phase1_avg * 100

            criteria = self.ACCEPTANCE_CRITERIA[variant]
            variant_summary = {
                'variant': variant,
                'peak_gflops': peak_result.gflops,
                'peak_size': peak_result.matrix_size,
                'avg_gflops': avg_gflops,
                'improvement_vs_phase1': improvement,
                'target_gflops': criteria['target_gflops'],
                'min_gflops': criteria['min_gflops'],
                'all_passed': variant_passed,
                'details': [asdict(r) for r in variant_results]
            }

            summary['results'].append(variant_summary)
            summary['acceptance_status'][variant] = variant_passed

        summary['all_variants_passed'] = all(summary['acceptance_status'].values())

        return summary

    def print_summary(self, summary: Dict):
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print("STRASSEN BENCHMARK SUMMARY")
        print("=" * 80)

        for variant_result in summary['results']:
            variant = variant_result['variant']
            status = "✅ PASS" if variant_result['all_passed'] else "❌ FAIL"

            print(f"\nStrassen {variant.upper()} Variant {status}")
            print("-" * 80)
            print(f"Peak Performance:     {variant_result['peak_gflops']:.1f} GFLOPS "
                  f"(at {variant_result['peak_size']}×{variant_result['peak_size']})")
            print(f"Average Performance:  {variant_result['avg_gflops']:.1f} GFLOPS")
            print(f"Improvement vs Phase 1: {variant_result['improvement_vs_phase1']:+.1f}%")
            print(f"Target:               {variant_result['target_gflops']:.1f} GFLOPS")
            print(f"Minimum Required:     {variant_result['min_gflops']:.1f} GFLOPS")

            print(f"\nDetailed Results:")
            print(f"{'Size':<10} {'GFLOPS':<12} {'Error':<12} {'CV%':<8} {'Status'}")
            print("-" * 60)
            for detail in variant_result['details']:
                status_icon = "✅" if detail['passed'] else "❌"
                print(f"{detail['matrix_size']:<10} "
                      f"{detail['gflops']:<12.1f} "
                      f"{detail['relative_error']:<12.2e} "
                      f"{detail['cv_percent']:<8.2f} "
                      f"{status_icon}")

        print("\n" + "=" * 80)
        overall_status = "✅ ALL TESTS PASSED" if summary['all_variants_passed'] else "❌ SOME TESTS FAILED"
        print(f"OVERALL STATUS: {overall_status}")
        print("=" * 80)


def main():
    """Main benchmark function."""
    suite = StrassenBenchmarkSuite()
    summary = suite.run_full_suite()

    # Save results
    results_file = Path(__file__).parent / 'results' / 'strassen_benchmark_results.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Print summary
    suite.print_summary(summary)

    return summary


if __name__ == "__main__":
    main()