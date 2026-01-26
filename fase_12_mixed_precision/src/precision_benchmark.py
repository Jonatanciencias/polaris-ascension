#!/usr/bin/env python3
"""
📊 MIXED PRECISION BENCHMARKING SCRIPT
======================================

Benchmarking script específico para comparar optimizaciones de precisión mixta
con el baseline de 758.51 GFLOPS establecido en las fases anteriores.

Fase 12: Mixed Precision Optimizations
Objetivo: Alcanzar 1000+ GFLOPS en Radeon RX 580

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from mixed_precision_engine import MixedPrecisionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecisionBenchmarker:
    """Specialized benchmarker for mixed precision optimizations vs baseline"""

    def __init__(self):
        self.engine = MixedPrecisionEngine()
        self.baseline_gflops = 758.51  # From OpenCL kernel optimization
        self.target_gflops = 1000.0

    def benchmark_vs_baseline(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark mixed precision performance against established baseline

        Args:
            sizes: Matrix sizes to test

        Returns:
            Benchmark results
        """
        if sizes is None:
            sizes = [1024, 1536, 2048, 2560]  # Test larger sizes for peak performance

        results = {
            'sizes': sizes,
            'fp32_performance': [],
            'mixed_performance': [],
            'fp16_performance': [],
            'baseline_reference': self.baseline_gflops,
            'target_reference': self.target_gflops,
            'improvement_mixed_percent': [],
            'improvement_fp16_percent': [],
            'accuracy_loss_percent': [],
            'meets_target': [],
            'peak_performance': 0.0,
            'optimal_size': 0
        }

        logger.info("🚀 Benchmarking Mixed Precision vs Baseline (758.51 GFLOPS)")

        for size in sizes:
            logger.info(f"Testing {size}x{size} matrix multiplication")

            # Generate test matrices
            np.random.seed(42)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            try:
                # Benchmark FP32 baseline
                _, fp32_time = self.engine.gemm_fp32(A, B)
                operations = 2 * size * size * size
                fp32_gflops = operations / (fp32_time * 1e-3 * 1e9)
                results['fp32_performance'].append(fp32_gflops)

                # Benchmark mixed precision
                _, metrics = self.engine.gemm_mixed_precision(A, B)
                mixed_gflops = metrics.gflops_mixed
                results['mixed_performance'].append(mixed_gflops)
                results['accuracy_loss_percent'].append(metrics.accuracy_loss_percent)

                # Calculate improvement over baseline
                improvement_mixed = (mixed_gflops / self.baseline_gflops - 1) * 100
                results['improvement_mixed_percent'].append(improvement_mixed)

                # Check if meets target
                meets_target = mixed_gflops >= self.target_gflops
                results['meets_target'].append(meets_target)

                # Track peak performance
                if mixed_gflops > results['peak_performance']:
                    results['peak_performance'] = mixed_gflops
                    results['optimal_size'] = size

                # Benchmark FP16 if available
                if self.engine.config.use_fp16:
                    try:
                        _, fp16_time = self.engine.gemm_fp16(A, B)
                        fp16_gflops = operations / (fp16_time * 1e-3 * 1e9)
                        improvement_fp16 = (fp16_gflops / self.baseline_gflops - 1) * 100
                        results['fp16_performance'].append(fp16_gflops)
                        results['improvement_fp16_percent'].append(improvement_fp16)
                    except Exception as e:
                        logger.warning(f"FP16 benchmark failed: {e}")
                        results['fp16_performance'].append(0.0)
                        results['improvement_fp16_percent'].append(-100.0)
                else:
                    results['fp16_performance'].append(0.0)
                    results['improvement_fp16_percent'].append(-100.0)

                logger.info(f"Size {size}x{size}: Mixed={mixed_gflops:.2f} GFLOPS "
                           ".1f"
                           f"Target: {'✅ MET' if meets_target else '❌ NOT MET'}")

            except Exception as e:
                logger.error(f"Benchmark failed for size {size}x{size}: {e}")
                results['fp32_performance'].append(0.0)
                results['mixed_performance'].append(0.0)
                results['fp16_performance'].append(0.0)
                results['improvement_mixed_percent'].append(-100.0)
                results['improvement_fp16_percent'].append(-100.0)
                results['accuracy_loss_percent'].append(100.0)
                results['meets_target'].append(False)

        return results

    def run_extended_benchmark(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Run extended benchmark to find peak sustainable performance

        Args:
            duration_minutes: How long to run the benchmark

        Returns:
            Extended benchmark results
        """
        logger.info(f"🏃 Running extended {duration_minutes}-minute mixed precision benchmark")

        size = 1536  # Use large matrices for peak performance
        results = {
            'duration_minutes': duration_minutes,
            'matrix_size': size,
            'iterations': 0,
            'total_operations': 0,
            'average_mixed_gflops': 0.0,
            'average_fp32_gflops': 0.0,
            'peak_mixed_gflops': 0.0,
            'sustained_mixed_gflops': 0.0,
            'average_accuracy_loss': 0.0,
            'performance_stability': 0.0
        }

        # Generate test matrices
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        mixed_performances = []
        fp32_performances = []
        accuracy_losses = []
        iteration = 0

        while time.time() < end_time:
            try:
                # Benchmark mixed precision
                _, metrics = self.engine.gemm_mixed_precision(A, B)
                mixed_performances.append(metrics.gflops_mixed)
                accuracy_losses.append(metrics.accuracy_loss_percent)

                # Benchmark FP32 every 5 iterations for comparison
                if iteration % 5 == 0:
                    _, fp32_time = self.engine.gemm_fp32(A, B)
                    operations = 2 * size * size * size
                    fp32_gflops = operations / (fp32_time * 1e-3 * 1e9)
                    fp32_performances.append(fp32_gflops)

                iteration += 1

                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_mixed = np.mean(mixed_performances[-10:])  # Last 10 iterations
                    avg_accuracy = np.mean(accuracy_losses[-10:])
                    logger.info(f"Iteration {iteration}: {avg_mixed:.2f} GFLOPS mixed, "
                               ".2f")

            except Exception as e:
                logger.error(f"Extended benchmark iteration {iteration} failed: {e}")
                break

        # Calculate final statistics
        if mixed_performances:
            results['iterations'] = len(mixed_performances)
            results['total_operations'] = len(mixed_performances) * 2 * size * size * size
            results['average_mixed_gflops'] = np.mean(mixed_performances)
            results['average_fp32_gflops'] = np.mean(fp32_performances) if fp32_performances else 0.0
            results['peak_mixed_gflops'] = np.max(mixed_performances)
            results['sustained_mixed_gflops'] = np.percentile(mixed_performances, 10)  # 10th percentile for sustained
            results['average_accuracy_loss'] = np.mean(accuracy_losses)
            results['performance_stability'] = np.std(mixed_performances) / np.mean(mixed_performances) * 100

            logger.info("📊 Extended Mixed Precision Benchmark Results:")
            logger.info(f"   Iterations: {results['iterations']}")
            logger.info(f"   Average Mixed: {results['average_mixed_gflops']:.2f} GFLOPS")
            logger.info(f"   Average FP32: {results['average_fp32_gflops']:.2f} GFLOPS")
            logger.info(f"   Peak Mixed: {results['peak_mixed_gflops']:.2f} GFLOPS")
            logger.info(f"   Sustained Mixed (P10): {results['sustained_mixed_gflops']:.2f} GFLOPS")
            logger.info(f"   Average Accuracy Loss: {results['average_accuracy_loss']:.2f}%")
            logger.info(f"   Performance Stability: {results['performance_stability']:.1f}%")
        else:
            logger.error("Extended benchmark failed - no successful iterations")

        return results

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Complete performance report
        """
        logger.info("📋 Generating Mixed Precision Performance Report")

        # Run benchmarks
        baseline_comparison = self.benchmark_vs_baseline()
        extended_benchmark = self.run_extended_benchmark(duration_minutes=3)  # Shorter for testing

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 12: Mixed Precision Optimizations',
            'baseline_performance_gflops': self.baseline_gflops,
            'target_performance_gflops': self.target_gflops,
            'baseline_comparison': baseline_comparison,
            'extended_benchmark': extended_benchmark,
            'conclusions': {}
        }

        # Analyze results
        peak_performance = baseline_comparison['peak_performance']
        sustained_performance = extended_benchmark.get('sustained_mixed_gflops', 0.0)
        avg_accuracy_loss = extended_benchmark.get('average_accuracy_loss', 100.0)

        # Determine success criteria
        meets_peak_target = peak_performance >= self.target_gflops
        meets_sustained_target = sustained_performance >= self.target_gflops

        improvement = (peak_performance / self.baseline_gflops - 1) * 100

        if meets_sustained_target and avg_accuracy_loss < 5.0:
            conclusion = "🎉 BREAKTHROUGH: Mixed precision achieves 1000+ GFLOPS sustained with excellent accuracy!"
            recommendation = "✅ ACCEPT: Integrate mixed precision into production"
            status = "SUCCESS"
        elif meets_peak_target and avg_accuracy_loss < 10.0:
            conclusion = "⭐ PROMISING: Mixed precision achieves 1000+ GFLOPS peak performance"
            recommendation = "⚠️ CONDITIONAL: Optimize for sustained performance and accuracy"
            status = "PARTIAL_SUCCESS"
        elif peak_performance >= self.baseline_gflops * 1.3 and avg_accuracy_loss < 5.0:
            conclusion = "📈 SIGNIFICANT IMPROVEMENT: Mixed precision provides substantial gains"
            recommendation = "✅ ACCEPT: Good performance/accuracy balance"
            status = "SUCCESS"
        elif peak_performance >= self.baseline_gflops * 1.2 and avg_accuracy_loss < 15.0:
            conclusion = "📈 MODERATE IMPROVEMENT: Mixed precision shows meaningful gains"
            recommendation = "⚠️ CONDITIONAL: Needs accuracy optimization"
            status = "PARTIAL_SUCCESS"
        else:
            conclusion = "❌ INSUFFICIENT: Mixed precision does not meet performance targets"
            recommendation = "⏭️ NEXT: Move to Phase 13 (GCN Architecture Tuning)"
            status = "FAILURE"

        report['conclusions'] = {
            'status': status,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'peak_performance_gflops': peak_performance,
            'sustained_performance_gflops': sustained_performance,
            'improvement_over_baseline_percent': improvement,
            'average_accuracy_loss_percent': avg_accuracy_loss,
            'meets_1000_gflops_target': meets_peak_target,
            'meets_sustained_1000_gflops': meets_sustained_target
        }

        # Save report
        report_file = Path(__file__).parent / "results" / "mixed_precision_performance_report.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 Performance report saved to {report_file}")
        logger.info("🎯 Conclusions:")
        logger.info(f"   Status: {status}")
        logger.info(f"   {conclusion}")
        logger.info(f"   {recommendation}")

        return report

def main():
    """Main benchmarking function"""
    logger.info("📊 Mixed Precision Performance Benchmark")
    logger.info("=" * 55)

    benchmarker = PrecisionBenchmarker()

    try:
        report = benchmarker.generate_performance_report()

        # Print summary
        print("\n" + "="*70)
        print("🎯 MIXED PRECISION PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Baseline Performance: {benchmarker.baseline_gflops:.2f} GFLOPS")
        print(f"Target Performance: {benchmarker.target_gflops:.2f} GFLOPS")
        print(f"Peak Achieved: {report['conclusions']['peak_performance_gflops']:.2f} GFLOPS")
        print(f"Sustained Achieved: {report['conclusions']['sustained_performance_gflops']:.2f} GFLOPS")
        print(".1f"        print(f"Average Accuracy Loss: {report['conclusions']['average_accuracy_loss_percent']:.2f}%")
        print(f"Status: {report['conclusions']['status']}")
        print(f"Conclusion: {report['conclusions']['conclusion']}")
        print(f"Recommendation: {report['conclusions']['recommendation']}")
        print("="*70)

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()