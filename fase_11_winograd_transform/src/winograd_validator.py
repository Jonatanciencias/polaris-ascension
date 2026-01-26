#!/usr/bin/env python3
"""
🧪 WINOGRAD TRANSFORM VALIDATION & TESTING
==========================================

Script de validación completa para la implementación de Winograd transforms.
Verifica la corrección numérica, mide el rendimiento y compara con métodos
tradicionales de multiplicación matricial.

Para Radeon RX 580 - Fase 11: Winograd Transform Integration

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
import json
from typing import Dict, Any, List
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from winograd_transform import WinogradTransform, WinogradMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WinogradValidator:
    """Comprehensive validator for Winograd transform implementation"""

    def __init__(self):
        self.winograd = WinogradTransform()
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def validate_correctness(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Validate numerical correctness of Winograd implementation

        Args:
            sizes: Matrix sizes to test

        Returns:
            Validation results
        """
        if sizes is None:
            sizes = [64, 128, 256, 512]

        results = {
            'sizes': sizes,
            'max_errors': [],
            'mean_errors': [],
            'std_errors': [],
            'success_rate': 0.0
        }

        logger.info("🔍 Validating Winograd numerical correctness")

        successful_tests = 0

        for size in sizes:
            logger.info(f"Testing correctness for {size}x{size} matrices")

            # Generate test matrices with known properties
            np.random.seed(42)  # For reproducible results
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            try:
                # Compute using Winograd
                C_winograd, _ = self.winograd.winograd_gemm(A, B)

                # Compute reference using NumPy
                C_numpy = A @ B

                # Calculate errors
                errors = np.abs(C_winograd - C_numpy)
                max_error = np.max(errors)
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                results['max_errors'].append(float(max_error))
                results['mean_errors'].append(float(mean_error))
                results['std_errors'].append(float(std_error))

                logger.info(f"Size {size}x{size}: Max error = {max_error:.2e}, "
                           f"Mean error = {mean_error:.2e}")

                # Consider test successful if max error is reasonable
                if max_error < 1e-1:  # Allow some floating point error
                    successful_tests += 1
                else:
                    logger.warning(f"High error detected for size {size}x{size}")

            except Exception as e:
                logger.error(f"Failed to test size {size}x{size}: {e}")
                results['max_errors'].append(float('inf'))
                results['mean_errors'].append(float('inf'))
                results['std_errors'].append(float('inf'))

        results['success_rate'] = successful_tests / len(sizes) * 100
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        return results

    def benchmark_performance(self, sizes: List[int] = None, iterations: int = 3) -> Dict[str, Any]:
        """
        Comprehensive performance benchmarking

        Args:
            sizes: Matrix sizes to benchmark
            iterations: Number of iterations for averaging

        Returns:
            Benchmark results
        """
        if sizes is None:
            sizes = [512, 1024, 2048]

        results = {
            'sizes': sizes,
            'winograd_gflops': [],
            'numpy_gflops': [],
            'speedup_factors': [],
            'operations_saved_percent': [],
            'kernel_times_ms': [],
            'total_times_ms': []
        }

        logger.info("⚡ Running performance benchmarks")

        for size in sizes:
            logger.info(f"Benchmarking {size}x{size} matrices")

            # Generate test matrices
            np.random.seed(42)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Benchmark Winograd (multiple iterations)
            winograd_times = []
            winograd_gflops = []

            for i in range(iterations):
                try:
                    C_winograd, metrics = self.winograd.winograd_gemm(A, B)
                    winograd_times.append(metrics.total_time_ms)
                    winograd_gflops.append(metrics.gflops)
                except Exception as e:
                    logger.error(f"Winograd iteration {i+1} failed: {e}")
                    winograd_times.append(float('inf'))
                    winograd_gflops.append(0.0)

            # Benchmark NumPy
            numpy_times = []
            numpy_gflops = []

            for i in range(iterations):
                start_time = time.time()
                C_numpy = A @ B
                numpy_time = (time.time() - start_time) * 1000
                numpy_times.append(numpy_time)

                operations = 2 * size * size * size
                numpy_gflops.append(operations / (numpy_time * 1e6))

            # Calculate averages
            avg_winograd_gflops = np.mean([g for g in winograd_gflops if g > 0])
            avg_numpy_gflops = np.mean(numpy_gflops)
            avg_winograd_time = np.mean([t for t in winograd_times if t < float('inf')])
            avg_numpy_time = np.mean(numpy_times)

            # Calculate speedup
            speedup = avg_numpy_time / avg_winograd_time if avg_winograd_time > 0 else 0.0

            # Get operations saved from last Winograd run
            try:
                _, last_metrics = self.winograd.winograd_gemm(A, B)
                ops_saved = last_metrics.operations_saved_percent
            except:
                ops_saved = 0.0

            results['winograd_gflops'].append(float(avg_winograd_gflops))
            results['numpy_gflops'].append(float(avg_numpy_gflops))
            results['speedup_factors'].append(float(speedup))
            results['operations_saved_percent'].append(float(ops_saved))
            results['kernel_times_ms'].append(float(avg_winograd_time))
            results['total_times_ms'].append(float(avg_winograd_time))

            logger.info(f"Size {size}x{size}: Winograd = {avg_winograd_gflops:.2f} GFLOPS, "
                       f"NumPy = {avg_numpy_gflops:.2f} GFLOPS, "
                       f"Speedup = {speedup:.2f}x, "
                       f"Ops saved = {ops_saved:.1f}%")

        return results

    def compare_with_baseline(self, baseline_gflops: float = 758.51) -> Dict[str, Any]:
        """
        Compare Winograd performance with project baseline

        Args:
            baseline_gflops: Baseline performance from OpenCL kernels

        Returns:
            Comparison results
        """
        logger.info("📊 Comparing with project baseline")

        # Run benchmark for representative sizes
        benchmark_results = self.benchmark_performance(sizes=[1024, 2048])

        comparison = {
            'baseline_gflops': baseline_gflops,
            'winograd_max_gflops': max(benchmark_results['winograd_gflops']),
            'improvement_over_baseline_percent': 0.0,
            'meets_1000_gflops_target': False,
            'recommendation': ""
        }

        max_winograd = comparison['winograd_max_gflops']
        improvement = (max_winograd / baseline_gflops - 1) * 100
        comparison['improvement_over_baseline_percent'] = improvement

        if max_winograd >= 1000.0:
            comparison['meets_1000_gflops_target'] = True
            comparison['recommendation'] = "✅ EXCELLENT: Winograd transforms achieve 1000+ GFLOPS target"
        elif max_winograd >= baseline_gflops * 1.1:
            comparison['recommendation'] = "✅ GOOD: Winograd provides meaningful performance improvement"
        elif max_winograd >= baseline_gflops * 0.9:
            comparison['recommendation'] = "⚠️ MODERATE: Winograd maintains baseline performance"
        else:
            comparison['recommendation'] = "❌ POOR: Winograd underperforms compared to baseline"

        logger.info(f"Baseline: {baseline_gflops:.2f} GFLOPS")
        logger.info(f"Winograd Max: {max_winograd:.2f} GFLOPS")
        logger.info(f"Improvement: {improvement:.1f}%")
        logger.info(f"Recommendation: {comparison['recommendation']}")

        return comparison

    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite

        Returns:
            Complete validation results
        """
        logger.info("🚀 Starting complete Winograd validation suite")

        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 11: Winograd Transform Integration',
            'correctness_validation': {},
            'performance_benchmark': {},
            'baseline_comparison': {},
            'overall_assessment': {}
        }

        # 1. Correctness validation
        logger.info("Step 1: Numerical correctness validation")
        results['correctness_validation'] = self.validate_correctness()

        # 2. Performance benchmarking
        logger.info("Step 2: Performance benchmarking")
        results['performance_benchmark'] = self.benchmark_performance()

        # 3. Baseline comparison
        logger.info("Step 3: Baseline performance comparison")
        results['baseline_comparison'] = self.compare_with_baseline()

        # 4. Overall assessment
        correctness_rate = results['correctness_validation']['success_rate']
        max_performance = max(results['performance_benchmark']['winograd_gflops'])
        meets_target = results['baseline_comparison']['meets_1000_gflops_target']

        if correctness_rate >= 90.0 and meets_target:
            assessment = "✅ ACCEPTED: Winograd transforms provide excellent performance and accuracy"
            recommendation = "Integrate into production pipeline"
        elif correctness_rate >= 80.0 and max_performance >= 758.51 * 1.05:
            assessment = "✅ ACCEPTED: Winograd transforms provide good performance with acceptable accuracy"
            recommendation = "Integrate with accuracy monitoring"
        elif correctness_rate >= 70.0:
            assessment = "⚠️ CONDITIONAL: Winograd shows promise but needs accuracy improvements"
            recommendation = "Continue development and testing"
        else:
            assessment = "❌ REJECTED: Winograd transforms do not meet accuracy or performance requirements"
            recommendation = "Move to next optimization technique"

        results['overall_assessment'] = {
            'assessment': assessment,
            'recommendation': recommendation,
            'correctness_rate': correctness_rate,
            'max_performance_gflops': max_performance,
            'meets_1000_gflops_target': meets_target
        }

        # Save results
        results_file = self.results_dir / "winograd_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📄 Results saved to {results_file}")
        logger.info("🎯 Overall Assessment:")
        logger.info(f"   {assessment}")
        logger.info(f"   Recommendation: {recommendation}")

        return results

def main():
    """Main validation function"""
    logger.info("🧪 Winograd Transform Validation Suite")
    logger.info("=" * 50)

    validator = WinogradValidator()

    try:
        results = validator.run_full_validation()

        # Print summary
        print("\n" + "="*60)
        print("🎯 WINOGRAD TRANSFORM VALIDATION SUMMARY")
        print("="*60)
        print(f"Correctness Rate: {results['correctness_validation']['success_rate']:.1f}%")
        print(f"Max Performance: {max(results['performance_benchmark']['winograd_gflops']):.2f} GFLOPS")
        print(f"Assessment: {results['overall_assessment']['assessment']}")
        print(f"Recommendation: {results['overall_assessment']['recommendation']}")
        print("="*60)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()