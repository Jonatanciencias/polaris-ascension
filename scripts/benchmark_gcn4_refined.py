#!/usr/bin/env python3
"""
GCN 4.0 Refined Architecture-Aware GEMM Benchmark Suite
Tests Polaris 10 refined optimizations for consistent 300+ GFLOPS

Target: 300-315 GFLOPS (+5-10% desde 285 GFLOPS)
Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)
Refinements: Workgroup size optimization, advanced LDS banking, memory prefetching
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

from opencl.gemm_gcn4_refined import GCN4RefinedConfig, GCN4RefinedGEMMExecutor
from opencl.gemm_vectorized import VectorizedGEMMExecutor
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCN4RefinedBenchmarkResult:
    """Results from a GCN 4.0 refined benchmark run."""

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

class GCN4RefinedBenchmarkSuite:
    """Comprehensive benchmarking for GCN 4.0 refined GEMM."""

    def __init__(self, config: GCN4RefinedConfig = None):
        self.config = config or GCN4RefinedConfig()
        self.gcn4_refined_executor = GCN4RefinedGEMMExecutor(self.config)
        self.vectorized_executor = VectorizedGEMMExecutor()  # For comparison
        self.results: List[GCN4RefinedBenchmarkResult] = []

        # Acceptance criteria for GCN 4.0 refined optimizations
        self.acceptance_criteria = {
            'target_gflops': 300.0,      # Target 300+ GFLOPS
            'min_improvement': 0.05,     # 5% minimum improvement over vectorized
            'max_error': 1e-4,          # Maintain accuracy
            'max_cv': 0.05             # Max coefficient of variation
        }

    def run_single_benchmark(self, size: int, num_runs: int = 5,
                           warmup: int = 3) -> Dict[str, GCN4RefinedBenchmarkResult]:
        """Run benchmark comparing GCN4 refined vs Vectorized kernels."""
        logger.info(f"Benchmarking GCN4 refined vs Vectorized, size {size}×{size}")

        # Create test matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Reference result (NumPy)
        C_ref = A @ B

        results = {}

        # Test GCN 4.0 refined kernel
        logger.info("  Testing GCN 4.0 refined kernel...")
        times_gcn4 = []
        errors_gcn4 = []

        # Warmup
        for _ in range(warmup):
            _ = self.gcn4_refined_executor.gemm(A, B)

        # Benchmark runs
        for _ in range(num_runs):
            C_gcn4, exec_time = self.gcn4_refined_executor.gemm(A, B)
            times_gcn4.append(exec_time)

            # Check accuracy
            error = np.max(np.abs(C_gcn4 - C_ref)) / np.max(np.abs(C_ref))
            errors_gcn4.append(error)

        # Statistics for GCN4 refined
        avg_time_gcn4 = np.mean(times_gcn4)
        gflops_gcn4 = (2 * size**3) / (avg_time_gcn4 * 1e-3) / 1e9

        result_gcn4 = GCN4RefinedBenchmarkResult(
            matrix_size=f"{size}x{size}x{size}",
            kernel_name="gemm_gcn4_refined",
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

        result_vec = GCN4RefinedBenchmarkResult(
            matrix_size=f"{size}x{size}x{size}",
            kernel_name="gemm_vectorized_float4",
            avg_time_ms=avg_time_vec,
            min_time_ms=np.min(times_vec),
            max_time_ms=np.max(times_vec),
            std_time_ms=np.std(times_vec),
            gflops=gflops_vec,
            accuracy_error=np.mean(errors_vec)
        )

        results['gcn4_refined'] = result_gcn4
        results['vectorized'] = result_vec

        return results

    def run_comprehensive_benchmark(self, sizes: List[int] = None,
                                  num_runs: int = 5) -> Dict[str, Any]:
        """Run complete GCN 4.0 refined benchmark suite."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]  # Focus on sizes where GCN4 showed issues

        logger.info("GCN 4.0 REFINED GEMM BENCHMARK SUITE - PHASE 4")
        logger.info(f"Target: {self.acceptance_criteria['target_gflops']} GFLOPS")
        logger.info(f"Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)")
        logger.info(f"Refinements: Workgroup 32×8, Advanced LDS banking, Memory prefetching")

        all_results = {}

        for size in sizes:
            try:
                results = self.run_single_benchmark(size, num_runs)
                all_results[f"{size}x{size}x{size}"] = {
                    'gcn4_refined': results['gcn4_refined'].to_dict(),
                    'vectorized': results['vectorized'].to_dict()
                }

                # Log immediate results
                gcn4_gflops = results['gcn4_refined'].gflops
                vec_gflops = results['vectorized'].gflops
                improvement = (gcn4_gflops - vec_gflops) / vec_gflops * 100

                logger.info(f"Size {size}×{size}: GCN4 {gcn4_gflops:.1f} GFLOPS "
                          f"vs Vectorized {vec_gflops:.1f} GFLOPS "
                          f"({improvement:+.1f}%)")

            except Exception as e:
                logger.error(f"Failed to benchmark size {size}: {e}")
                continue

        return all_results

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights."""
        analysis = {
            'summary': {},
            'recommendations': [],
            'performance_analysis': {},
            'accuracy_analysis': {}
        }

        # Extract GCN4 refined results
        gcn4_results = {}
        vec_results = {}

        for size_key, size_results in results.items():
            if 'gcn4_refined' in size_results:
                gcn4_results[size_key] = size_results['gcn4_refined']
            if 'vectorized' in size_results:
                vec_results[size_key] = size_results['vectorized']

        if not gcn4_results:
            analysis['summary']['status'] = 'FAILED'
            analysis['summary']['message'] = 'No GCN4 refined results available'
            return analysis

        # Performance analysis
        gcn4_gflops_values = [r['gflops'] for r in gcn4_results.values()]
        vec_gflops_values = [r['gflops'] for r in vec_results.values()]

        analysis['performance_analysis'] = {
            'gcn4_avg_gflops': np.mean(gcn4_gflops_values),
            'gcn4_min_gflops': np.min(gcn4_gflops_values),
            'gcn4_max_gflops': np.max(gcn4_gflops_values),
            'vec_avg_gflops': np.mean(vec_gflops_values),
            'improvement_percent': (np.mean(gcn4_gflops_values) - np.mean(vec_gflops_values)) /
                                 np.mean(vec_gflops_values) * 100,
            'target_achieved': np.mean(gcn4_gflops_values) >= self.acceptance_criteria['target_gflops']
        }

        # Accuracy analysis
        gcn4_errors = [r['accuracy_error'] for r in gcn4_results.values()]
        analysis['accuracy_analysis'] = {
            'gcn4_max_error': np.max(gcn4_errors),
            'gcn4_avg_error': np.mean(gcn4_errors),
            'accuracy_acceptable': np.max(gcn4_errors) <= self.acceptance_criteria['max_error']
        }

        # Overall assessment
        target_met = analysis['performance_analysis']['target_achieved']
        accuracy_ok = analysis['accuracy_analysis']['accuracy_acceptable']
        improvement_ok = analysis['performance_analysis']['improvement_percent'] >= self.acceptance_criteria['min_improvement'] * 100

        if target_met and accuracy_ok:
            analysis['summary']['status'] = 'SUCCESS'
            analysis['summary']['message'] = (f"GCN 4.0 refined achieved {analysis['performance_analysis']['gcn4_avg_gflops']:.1f} GFLOPS "
                                            f"(target: {self.acceptance_criteria['target_gflops']} GFLOPS)")
        elif improvement_ok and accuracy_ok:
            analysis['summary']['status'] = 'PARTIAL_SUCCESS'
            analysis['summary']['message'] = (f"GCN 4.0 refined shows {analysis['performance_analysis']['improvement_percent']:.1f}% improvement "
                                            f"but did not reach {self.acceptance_criteria['target_gflops']} GFLOPS target")
        else:
            analysis['summary']['status'] = 'FAILED'
            analysis['summary']['message'] = 'GCN 4.0 refined optimizations did not meet acceptance criteria'

        # Recommendations
        if not target_met:
            analysis['recommendations'].append(
                f"Further optimization needed to reach {self.acceptance_criteria['target_gflops']} GFLOPS target"
            )

        if not accuracy_ok:
            analysis['recommendations'].append(
                "Accuracy issues detected - review kernel implementation"
            )

        return analysis

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any],
                    filename: str = None) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gcn4_refined_benchmark_{timestamp}.json"

        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        filepath = results_dir / filename

        output_data = {
            'benchmark_info': {
                'suite': 'GCN 4.0 Refined GEMM Benchmark',
                'target': f"{self.acceptance_criteria['target_gflops']} GFLOPS",
                'hardware': 'AMD Radeon RX 590 (Polaris 10, GCN 4.0)',
                'refinements': 'Workgroup 32×8, Advanced LDS banking, Memory prefetching',
                'timestamp': datetime.now().isoformat()
            },
            'results': self._convert_numpy_types(results),
            'analysis': self._convert_numpy_types(analysis)
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return str(filepath)

    def print_summary(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Print human-readable benchmark summary."""
        print("\n" + "="*80)
        print("GCN 4.0 REFINED GEMM BENCHMARK SUMMARY")
        print("="*80)

        print(f"\nTarget: {self.acceptance_criteria['target_gflops']} GFLOPS")
        print(f"Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)")
        print("Refinements: Workgroup 32×8, Advanced LDS banking, Memory prefetching")

        print(f"\nStatus: {analysis['summary']['status']}")
        print(f"Message: {analysis['summary']['message']}")

        perf = analysis['performance_analysis']
        print("\nPerformance Analysis:")
        print(f"  GCN4 Avg GFLOPS: {perf['gcn4_avg_gflops']:.1f}")
        print(f"  GCN4 Min GFLOPS: {perf['gcn4_min_gflops']:.1f}")
        print(f"  GCN4 Max GFLOPS: {perf['gcn4_max_gflops']:.1f}")
        print(f"  Vectorized Avg GFLOPS: {perf['vec_avg_gflops']:.1f}")
        print(f"  Improvement: {perf['improvement_percent']:.1f}%")

        acc = analysis['accuracy_analysis']
        print("\nAccuracy Analysis:")
        print(f"  GCN4 Max Error: {acc['gcn4_max_error']:.2e}")
        print(f"  GCN4 Avg Error: {acc['gcn4_avg_error']:.2e}")
        print(f"  Accuracy Acceptable: {acc['accuracy_acceptable']}")

        if analysis['recommendations']:
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")

        print("\nDetailed Results by Matrix Size:")
        print("-" * 80)
        print("<12")
        print("-" * 80)

        for size_key, size_results in results.items():
            if 'gcn4_refined' in size_results:
                gcn4 = size_results['gcn4_refined']
                vec = size_results.get('vectorized', {})
                vec_gflops = vec.get('gflops', 0)
                improvement = (gcn4['gflops'] - vec_gflops) / vec_gflops * 100 if vec_gflops > 0 else 0

                print("<12"
                      "<8.1f"
                      "<8.1f"
                      "<+7.1f")

def main():
    """Main benchmark execution."""
    suite = GCN4RefinedBenchmarkSuite()

    # Run comprehensive benchmark
    results = suite.run_comprehensive_benchmark()

    # Analyze results
    analysis = suite.analyze_results(results)

    # Save and print results
    results_file = suite.save_results(results, analysis)
    suite.print_summary(results, analysis)

    return results_file

if __name__ == "__main__":
    main()