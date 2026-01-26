#!/usr/bin/env python3
"""
🧪 MIXED PRECISION VALIDATION & TESTING
========================================

Suite de validación completa para optimizaciones de precisión mixta.
Verifica la corrección numérica, mide el rendimiento y compara con
métodos tradicionales de precisión completa.

Para Radeon RX 580 - Fase 12: Mixed Precision Optimizations

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

from mixed_precision_engine import MixedPrecisionEngine, PrecisionMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecisionValidator:
    """Comprehensive validator for mixed precision implementations"""

    def __init__(self):
        self.engine = MixedPrecisionEngine()
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def validate_accuracy(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Validate numerical accuracy of mixed precision implementations

        Args:
            sizes: Matrix sizes to test

        Returns:
            Accuracy validation results
        """
        if sizes is None:
            sizes = [128, 256, 512, 1024]

        results = {
            'sizes': sizes,
            'fp32_max_errors': [],
            'mixed_max_errors': [],
            'mixed_accuracy_loss_percent': [],
            'fp16_max_errors': [],
            'success_rate_fp32': 0.0,
            'success_rate_mixed': 0.0,
            'success_rate_fp16': 0.0
        }

        logger.info("🔍 Validating precision accuracy")

        for size in sizes:
            logger.info(f"Testing accuracy for {size}x{size} matrices")

            # Generate test matrices with known properties
            np.random.seed(42)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Reference computation (NumPy)
            C_ref = A @ B

            try:
                # Test FP32 implementation
                C_fp32, _ = self.engine.gemm_fp32(A, B)
                fp32_error = np.max(np.abs(C_fp32 - C_ref))
                results['fp32_max_errors'].append(float(fp32_error))

                # Test mixed precision
                C_mixed, _ = self.engine.gemm_mixed_precision(A, B)
                mixed_error = np.max(np.abs(C_mixed - C_ref))
                accuracy_loss = (mixed_error / np.max(np.abs(C_ref))) * 100
                results['mixed_max_errors'].append(float(mixed_error))
                results['mixed_accuracy_loss_percent'].append(float(accuracy_loss))

                # Test FP16 if available
                if self.engine.config.use_fp16:
                    try:
                        C_fp16, _ = self.engine.gemm_fp16(A, B)
                        fp16_error = np.max(np.abs(C_fp16 - C_ref))
                        results['fp16_max_errors'].append(float(fp16_error))
                    except Exception as e:
                        logger.warning(f"FP16 test failed: {e}")
                        results['fp16_max_errors'].append(float('inf'))
                else:
                    results['fp16_max_errors'].append(float('inf'))

                logger.info(f"Size {size}x{size}: FP32 error = {fp32_error:.2e}, "
                           f"Mixed error = {mixed_error:.2e} ({accuracy_loss:.3f}%)")

            except Exception as e:
                logger.error(f"Accuracy test failed for size {size}x{size}: {e}")
                results['fp32_max_errors'].append(float('inf'))
                results['mixed_max_errors'].append(float('inf'))
                results['mixed_accuracy_loss_percent'].append(100.0)
                results['fp16_max_errors'].append(float('inf'))

        # Calculate success rates (error < 1e-1 considered acceptable)
        results['success_rate_fp32'] = sum(1 for e in results['fp32_max_errors'] if e < 1e-1) / len(sizes) * 100
        results['success_rate_mixed'] = sum(1 for e in results['mixed_max_errors'] if e < 1e-1) / len(sizes) * 100
        results['success_rate_fp16'] = sum(1 for e in results['fp16_max_errors'] if e < 1e-1 and e != float('inf')) / len(sizes) * 100

        logger.info(f"Success rate FP32: {results['success_rate_fp32']:.1f}%")
        logger.info(f"Success rate Mixed: {results['success_rate_mixed']:.1f}%")
        logger.info(f"Success rate FP16: {results['success_rate_fp16']:.1f}%")
        return results

    def benchmark_performance(self, sizes: List[int] = None, iterations: int = 3) -> Dict[str, Any]:
        """
        Comprehensive performance benchmarking

        Args:
            sizes: Matrix sizes to benchmark
            iterations: Number of iterations for averaging

        Returns:
            Performance benchmark results
        """
        if sizes is None:
            sizes = [512, 1024, 1536]

        results = {
            'sizes': sizes,
            'fp32_gflops': [],
            'mixed_gflops': [],
            'fp16_gflops': [],
            'speedup_mixed_vs_fp32': [],
            'speedup_fp16_vs_fp32': [],
            'accuracy_loss_percent': [],
            'memory_efficiency': [],
            'kernel_times_ms': []
        }

        logger.info("⚡ Running performance benchmarks")

        for size in sizes:
            logger.info(f"Benchmarking {size}x{size} matrices")

            # Generate test matrices
            np.random.seed(42)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Benchmark FP32 (baseline)
            fp32_times = []
            for i in range(iterations):
                try:
                    _, time_ms = self.engine.gemm_fp32(A, B)
                    fp32_times.append(time_ms)
                except Exception as e:
                    logger.error(f"FP32 iteration {i+1} failed: {e}")
                    fp32_times.append(float('inf'))

            avg_fp32_time = np.mean([t for t in fp32_times if t < float('inf')])
            operations = 2 * size * size * size
            fp32_gflops = operations / (avg_fp32_time * 1e-3 * 1e9) if avg_fp32_time < float('inf') else 0.0
            results['fp32_gflops'].append(float(fp32_gflops))

            # Benchmark mixed precision
            mixed_times = []
            accuracy_losses = []
            for i in range(iterations):
                try:
                    _, metrics = self.engine.gemm_mixed_precision(A, B)
                    mixed_times.append(metrics.kernel_time_ms)
                    accuracy_losses.append(metrics.accuracy_loss_percent)
                except Exception as e:
                    logger.error(f"Mixed precision iteration {i+1} failed: {e}")
                    mixed_times.append(float('inf'))
                    accuracy_losses.append(100.0)

            avg_mixed_time = np.mean([t for t in mixed_times if t < float('inf')])
            mixed_gflops = operations / (avg_mixed_time * 1e-3 * 1e9) if avg_mixed_time < float('inf') else 0.0
            avg_accuracy_loss = np.mean(accuracy_losses)

            results['mixed_gflops'].append(float(mixed_gflops))
            results['accuracy_loss_percent'].append(float(avg_accuracy_loss))
            results['kernel_times_ms'].append(float(avg_mixed_time))

            # Calculate speedups
            speedup_mixed = mixed_gflops / fp32_gflops if fp32_gflops > 0 else 0.0
            results['speedup_mixed_vs_fp32'].append(float(speedup_mixed))

            # Benchmark FP16 if available
            if self.engine.config.use_fp16:
                fp16_times = []
                for i in range(iterations):
                    try:
                        _, time_ms = self.engine.gemm_fp16(A, B)
                        fp16_times.append(time_ms)
                    except Exception as e:
                        logger.error(f"FP16 iteration {i+1} failed: {e}")
                        fp16_times.append(float('inf'))

                avg_fp16_time = np.mean([t for t in fp16_times if t < float('inf')])
                fp16_gflops = operations / (avg_fp16_time * 1e-3 * 1e9) if avg_fp16_time < float('inf') else 0.0
                speedup_fp16 = fp16_gflops / fp32_gflops if fp32_gflops > 0 else 0.0

                results['fp16_gflops'].append(float(fp16_gflops))
                results['speedup_fp16_vs_fp32'].append(float(speedup_fp16))
            else:
                results['fp16_gflops'].append(0.0)
                results['speedup_fp16_vs_fp32'].append(0.0)

            # Memory efficiency estimate
            memory_efficiency = 1.5 if mixed_gflops > fp32_gflops * 1.2 else 1.0
            results['memory_efficiency'].append(float(memory_efficiency))

            logger.info(f"Size {size}x{size}: FP32={fp32_gflops:.2f}, "
                       f"Mixed={mixed_gflops:.2f}, "
                       ".2f"
                       ".2f"
                       ".3f")

        return results

    def compare_with_baseline(self, baseline_gflops: float = 758.51) -> Dict[str, Any]:
        """
        Compare mixed precision performance with project baseline

        Args:
            baseline_gflops: Baseline performance from OpenCL kernels

        Returns:
            Comparison results
        """
        logger.info("📊 Comparing with project baseline")

        # Run benchmark for representative sizes
        benchmark_results = self.benchmark_performance(sizes=[1024, 1536])

        comparison = {
            'baseline_gflops': baseline_gflops,
            'mixed_max_gflops': max(benchmark_results['mixed_gflops']),
            'fp16_max_gflops': max(benchmark_results['fp16_gflops']) if benchmark_results['fp16_gflops'] else 0.0,
            'improvement_over_baseline_percent': 0.0,
            'meets_1000_gflops_target': False,
            'average_accuracy_loss': np.mean(benchmark_results['accuracy_loss_percent']),
            'recommendation': ""
        }

        max_mixed = comparison['mixed_max_gflops']
        max_fp16 = comparison['fp16_max_gflops']
        improvement = (max_mixed / baseline_gflops - 1) * 100
        comparison['improvement_over_baseline_percent'] = improvement

        avg_accuracy_loss = comparison['average_accuracy_loss']

        if max_mixed >= 1000.0 and avg_accuracy_loss < 5.0:
            comparison['meets_1000_gflops_target'] = True
            comparison['recommendation'] = "✅ EXCELLENT: Mixed precision achieves 1000+ GFLOPS with acceptable accuracy"
        elif max_mixed >= 1000.0 and avg_accuracy_loss < 10.0:
            comparison['recommendation'] = "⚠️ GOOD: Achieves 1000+ GFLOPS but with higher accuracy loss - needs tuning"
        elif max_mixed >= baseline_gflops * 1.2 and avg_accuracy_loss < 5.0:
            comparison['recommendation'] = "✅ GOOD: Significant improvement with good accuracy"
        elif max_mixed >= baseline_gflops * 1.1:
            comparison['recommendation'] = "⚠️ MODERATE: Modest improvement - needs accuracy optimization"
        else:
            comparison['recommendation'] = "❌ POOR: Insufficient performance gain"

        logger.info(f"Baseline: {baseline_gflops:.2f} GFLOPS")
        logger.info(f"Mixed Precision Max: {max_mixed:.2f} GFLOPS")
        logger.info(f"FP16 Max: {max_fp16:.2f} GFLOPS")
        logger.info(f"Improvement: {improvement:.1f}%")
        logger.info(f"Average Accuracy Loss: {avg_accuracy_loss:.2f}%")
        logger.info(f"Recommendation: {comparison['recommendation']}")

        return comparison

    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite

        Returns:
            Complete validation results
        """
        logger.info("🚀 Starting complete mixed precision validation suite")

        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 12: Mixed Precision Optimizations',
            'accuracy_validation': {},
            'performance_benchmark': {},
            'baseline_comparison': {},
            'overall_assessment': {}
        }

        # 1. Accuracy validation
        logger.info("Step 1: Accuracy validation")
        results['accuracy_validation'] = self.validate_accuracy()

        # 2. Performance benchmarking
        logger.info("Step 2: Performance benchmarking")
        results['performance_benchmark'] = self.benchmark_performance()

        # 3. Baseline comparison
        logger.info("Step 3: Baseline performance comparison")
        results['baseline_comparison'] = self.compare_with_baseline()

        # Analyze results
        accuracy_rate = results['accuracy_validation']['success_rate_mixed']
        max_performance = max(results['performance_benchmark']['mixed_gflops'])
        meets_target = results['baseline_comparison']['meets_1000_gflops_target']
        avg_accuracy_loss = results['baseline_comparison']['average_accuracy_loss']

        if accuracy_rate >= 90.0 and meets_target and avg_accuracy_loss < 5.0:
            assessment = "✅ ACCEPTED: Mixed precision achieves excellent performance and accuracy"
            recommendation = "Integrate into production pipeline with FP16 acceleration"
        elif accuracy_rate >= 80.0 and max_performance >= 758.51 * 1.2 and avg_accuracy_loss < 10.0:
            assessment = "✅ ACCEPTED: Mixed precision provides good performance gains"
            recommendation = "Integrate with accuracy monitoring and dynamic switching"
        elif accuracy_rate >= 70.0 and max_performance >= 758.51 * 1.1:
            assessment = "⚠️ CONDITIONAL: Mixed precision shows promise but needs accuracy improvements"
            recommendation = "Continue development with better error compensation"
        else:
            assessment = "❌ REJECTED: Mixed precision does not meet accuracy or performance requirements"
            recommendation = "Move to Phase 13 (GCN Architecture Tuning)"

        results['overall_assessment'] = {
            'assessment': assessment,
            'recommendation': recommendation,
            'accuracy_rate': accuracy_rate,
            'max_performance_gflops': max_performance,
            'meets_1000_gflops_target': meets_target,
            'average_accuracy_loss_percent': avg_accuracy_loss
        }

        # Save results
        results_file = self.results_dir / "mixed_precision_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📄 Results saved to {results_file}")
        logger.info("🎯 Overall Assessment:")
        logger.info(f"   {assessment}")
        logger.info(f"   Recommendation: {recommendation}")

        return results

def main():
    """Main validation function"""
    logger.info("🧪 Mixed Precision Validation Suite")
    logger.info("=" * 50)

    validator = PrecisionValidator()

    try:
        results = validator.run_full_validation()

        # Print summary
        print("\n" + "="*70)
        print("🎯 MIXED PRECISION VALIDATION SUMMARY")
        print("="*70)
        print(f"Accuracy Rate: {results['accuracy_validation']['success_rate_mixed']:.1f}%")
        print(f"Max Performance: {max(results['performance_benchmark']['mixed_gflops']):.2f} GFLOPS")
        print(f"Average Accuracy Loss: {results['baseline_comparison']['average_accuracy_loss']:.2f}%")
        print(f"Assessment: {results['overall_assessment']['assessment']}")
        print(f"Recommendation: {results['overall_assessment']['recommendation']}")
        print("="*70)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()