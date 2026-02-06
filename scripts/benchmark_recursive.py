#!/usr/bin/env python3

"""
Comprehensive Benchmarking Suite - Block Recursive GEMM

Phase 2, Technique 1 validation script.
Benchmarks the optimized kernel variant and compares against Phase 1 baseline.

Expected Results:
- Optimized variant: 850-870 GFLOPS (+10-12% vs Phase 1 baseline)

Author: Phase 2 Development Team
Date: 2026-01-24
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

from opencl.gemm_recursive_wrapper import RecursiveConfig, RecursiveGEMMExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
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


class RecursiveBenchmarkSuite:
    """Comprehensive benchmarking for recursive GEMM."""
    
    # Phase 1 baseline for comparison
    PHASE1_BASELINE = {
        256: 720.0,
        512: 750.0,
        1024: 775.3,
        2048: 780.0,
        4096: 770.0
    }
    
    # Acceptance criteria
    ACCEPTANCE_CRITERIA = {
        'optimized': {
            'target_gflops': 860.0,
            'min_gflops': 850.0,
            'max_error': 1e-5,
            'max_cv_percent': 5.0
        }
    }
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
    
    def run_single_benchmark(self, 
                            variant: str,
                            size: int,
                            iterations: int = 20,
                            warmup: int = 3) -> BenchmarkResult:
        """Run benchmark for a single configuration.
        
        Args:
            variant: Kernel variant ('optimized')
            size: Matrix size (N×N)
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            BenchmarkResult object
        """
        logger.info(f"Benchmarking {variant} variant, size {size}×{size}")
        
        # Create executor with specific variant
        config = RecursiveConfig(kernel_variant=variant)
        executor = RecursiveGEMMExecutor(config)
        
        # Generate test matrices
        np.random.seed(42)  # For reproducibility
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Calculate reference (NumPy)
        logger.debug("Computing reference result...")
        C_ref = A @ B
        
        # Warmup runs
        logger.debug(f"Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            _ = executor.gemm(A, B)
        
        # Benchmark runs
        logger.debug(f"Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            C = executor.gemm(A, B)
            end = time.perf_counter()
            times.append(end - start)
            
            if (i + 1) % 5 == 0:
                logger.debug(f"  Iteration {i+1}/{iterations}")
        
        # Calculate metrics
        times = np.array(times)
        time_mean = np.mean(times)
        time_std = np.std(times)
        
        flops = 2 * size**3  # C = A @ B requires 2N³ FLOPs
        gflops = flops / time_mean / 1e9
        
        # Numerical accuracy
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        
        # Coefficient of variation
        cv_percent = (time_std / time_mean) * 100
        
        # Memory usage (approximate)
        memory_mb = (3 * size**2 * 4) / (1024**2)  # 3 matrices × 4 bytes
        
        # Check acceptance criteria
        criteria = self.ACCEPTANCE_CRITERIA[variant]
        passed = (
            gflops >= criteria['min_gflops'] and
            error <= criteria['max_error'] and
            cv_percent <= criteria['max_cv_percent']
        )
        
        result = BenchmarkResult(
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
        
        # Log result
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} | {gflops:.1f} GFLOPS | Error: {error:.2e} | CV: {cv_percent:.2f}%")
        
        return result
    
    def run_full_suite(self, 
                      variants: List[str] = None,
                      sizes: List[int] = None) -> Dict:
        """Run complete benchmark suite.
        
        Args:
            variants: List of variants to test (default: all)
            sizes: List of sizes to test (default: [256, 512, 1024, 2048])
            
        Returns:
            Dictionary with complete results
        """
        if variants is None:
            variants = ['optimized']
        
        if sizes is None:
            sizes = [256, 512, 1024, 2048]
        
        logger.info("=" * 80)
        logger.info("RECURSIVE GEMM BENCHMARK SUITE - PHASE 2, TECHNIQUE 1")
        logger.info("=" * 80)
        
        # Run all benchmarks
        for variant in variants:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing variant: {variant.upper()}")
            logger.info(f"{'='*80}")
            
            for size in sizes:
                result = self.run_single_benchmark(variant, size)
                self.results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        return summary
    
    def _generate_summary(self) -> Dict:
        """Generate comprehensive summary of results."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 2 - Technique 1',
            'technique': 'Block Recursive GEMM',
            'results': [],
            'comparison_with_phase1': {},
            'acceptance_status': {}
        }
        
        # Organize results by variant
        by_variant = {}
        for result in self.results:
            variant = result.kernel_variant
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(result)
        
        # Process each variant
        for variant, results in by_variant.items():
            # Find peak performance
            peak_result = max(results, key=lambda r: r.gflops)
            
            # Calculate average GFLOPS
            avg_gflops = np.mean([r.gflops for r in results])
            
            # Compare with Phase 1 baseline (using 1024×1024 as reference)
            ref_result = next((r for r in results if r.matrix_size == 1024), None)
            if ref_result:
                phase1_baseline = self.PHASE1_BASELINE[1024]
                improvement = ((ref_result.gflops - phase1_baseline) / phase1_baseline) * 100
            else:
                improvement = 0.0
            
            # Check if variant meets acceptance criteria
            criteria = self.ACCEPTANCE_CRITERIA[variant]
            variant_passed = all(r.passed for r in results)
            
            variant_summary = {
                'variant': variant,
                'peak_gflops': peak_result.gflops,
                'peak_size': peak_result.matrix_size,
                'avg_gflops': avg_gflops,
                'improvement_vs_phase1': improvement,
                'target_gflops': criteria['target_gflops'],
                'min_gflops': criteria['min_gflops'],
                'all_passed': variant_passed,
                'details': [asdict(r) for r in results]
            }
            
            summary['results'].append(variant_summary)
            summary['acceptance_status'][variant] = variant_passed
        
        # Overall assessment
        summary['all_variants_passed'] = all(summary['acceptance_status'].values())
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print formatted summary to console."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for variant_result in summary['results']:
            variant = variant_result['variant']
            status = "✅ PASS" if variant_result['all_passed'] else "❌ FAIL"
            
            print(f"\n{variant.upper()} Variant {status}")
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
    
    def save_results(self, summary: Dict, output_path: Path):
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy/Python types to native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        summary_native = convert_to_native(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary_native, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def main():
    """Main benchmark execution."""
    # Parse arguments (simple version)
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark recursive GEMM kernels')
    parser.add_argument('--variants', nargs='+', 
                       choices=['optimized'],
                       default=['optimized'],
                       help='Variants to test')
    parser.add_argument('--sizes', nargs='+', type=int,
                       default=[256, 512, 1024, 2048],
                       help='Matrix sizes to test')
    parser.add_argument('--output', type=str,
                       default='results/technique_1_benchmark_results.json',
                       help='Output JSON file path')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (fewer iterations)')
    
    args = parser.parse_args()
    
    # Adjust iterations for quick test
    if args.quick:
        logger.info("Running in QUICK mode (fewer iterations)")
    
    # Create benchmark suite
    suite = RecursiveBenchmarkSuite()
    
    # Run benchmarks
    try:
        summary = suite.run_full_suite(
            variants=args.variants,
            sizes=args.sizes
        )
        
        # Print summary
        suite.print_summary(summary)
        
        # Save results
        output_path = Path(args.output)
        suite.save_results(summary, output_path)
        
        # Exit with appropriate code
        sys.exit(0 if summary['all_variants_passed'] else 1)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
