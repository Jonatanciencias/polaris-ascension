#!/usr/bin/env python3
"""
📊 WINOGRAD TRANSFORM BENCHMARKING SCRIPT
=========================================

Benchmarking script específico para comparar Winograd transforms
con el baseline de 758.51 GFLOPS establecido en las fases anteriores.

Fase 11: Winograd Transform Integration
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

from winograd_transform import WinogradTransform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WinogradBenchmarker:
    """Specialized benchmarker for Winograd transforms vs baseline"""

    def __init__(self):
        self.winograd = WinogradTransform()
        self.baseline_gflops = 758.51  # From OpenCL kernel optimization
        self.target_gflops = 1000.0

    def benchmark_vs_baseline(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark Winograd performance against established baseline

        Args:
            sizes: Matrix sizes to test

        Returns:
            Benchmark results
        """
        if sizes is None:
            sizes = [1024, 1536, 2048, 2560]  # Test larger sizes for peak performance

        results = {
            'sizes': sizes,
            'winograd_performance': [],
            'baseline_reference': self.baseline_gflops,
            'target_reference': self.target_gflops,
            'improvement_percent': [],
            'meets_target': [],
            'peak_performance': 0.0,
            'optimal_size': 0
        }

        logger.info("🚀 Benchmarking Winograd vs Baseline (758.51 GFLOPS)")

        for size in sizes:
            logger.info(f"Testing {size}x{size} matrix multiplication")

            # Generate test matrices
            np.random.seed(42)  # Reproducible results
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            try:
                # Benchmark Winograd
                C, metrics = self.winograd.winograd_gemm(A, B)

                winograd_gflops = metrics.gflops
                results['winograd_performance'].append(winograd_gflops)

                # Calculate improvement over baseline
                improvement = (winograd_gflops / self.baseline_gflops - 1) * 100
                results['improvement_percent'].append(improvement)

                # Check if meets target
                meets_target = winograd_gflops >= self.target_gflops
                results['meets_target'].append(meets_target)

                # Track peak performance
                if winograd_gflops > results['peak_performance']:
                    results['peak_performance'] = winograd_gflops
                    results['optimal_size'] = size

                logger.info(f"Size {size}x{size}: {winograd_gflops:.2f} GFLOPS "
                           ".1f"
                           f"Target: {'✅ MET' if meets_target else '❌ NOT MET'}")

            except Exception as e:
                logger.error(f"Benchmark failed for size {size}x{size}: {e}")
                results['winograd_performance'].append(0.0)
                results['improvement_percent'].append(-100.0)
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
        logger.info(f"🏃 Running extended {duration_minutes}-minute benchmark")

        size = 2048  # Use large matrices for peak performance
        results = {
            'duration_minutes': duration_minutes,
            'matrix_size': size,
            'iterations': 0,
            'total_operations': 0,
            'average_gflops': 0.0,
            'peak_gflops': 0.0,
            'sustained_gflops': 0.0,
            'performance_stability': 0.0
        }

        # Generate test matrices
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        performances = []
        iteration = 0

        while time.time() < end_time:
            try:
                C, metrics = self.winograd.winograd_gemm(A, B)
                performances.append(metrics.gflops)
                iteration += 1

                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_perf = np.mean(performances[-10:])  # Last 10 iterations
                    logger.info(f"Iteration {iteration}: {avg_perf:.2f} GFLOPS "
                               ".1f")

            except Exception as e:
                logger.error(f"Extended benchmark iteration {iteration} failed: {e}")
                break

        # Calculate final statistics
        if performances:
            results['iterations'] = len(performances)
            results['total_operations'] = len(performances) * 2 * size * size * size
            results['average_gflops'] = np.mean(performances)
            results['peak_gflops'] = np.max(performances)
            results['sustained_gflops'] = np.percentile(performances, 10)  # 10th percentile for sustained
            results['performance_stability'] = np.std(performances) / np.mean(performances) * 100

            logger.info("📊 Extended Benchmark Results:")
            logger.info(f"   Iterations: {results['iterations']}")
            logger.info(f"   Average: {results['average_gflops']:.2f} GFLOPS")
            logger.info(f"   Peak: {results['peak_gflops']:.2f} GFLOPS")
            logger.info(f"   Sustained (P10): {results['sustained_gflops']:.2f} GFLOPS")
            logger.info(".1f"
        else:
            logger.error("Extended benchmark failed - no successful iterations")

        return results

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Complete performance report
        """
        logger.info("📋 Generating Winograd Performance Report")

        # Run benchmarks
        baseline_comparison = self.benchmark_vs_baseline()
        extended_benchmark = self.run_extended_benchmark(duration_minutes=2)  # Shorter for testing

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 11: Winograd Transform Integration',
            'baseline_performance_gflops': self.baseline_gflops,
            'target_performance_gflops': self.target_gflops,
            'baseline_comparison': baseline_comparison,
            'extended_benchmark': extended_benchmark,
            'conclusions': {}
        }

        # Analyze results
        peak_performance = baseline_comparison['peak_performance']
        sustained_performance = extended_benchmark.get('sustained_gflops', 0.0)

        # Determine success criteria
        meets_peak_target = peak_performance >= self.target_gflops
        meets_sustained_target = sustained_performance >= self.target_gflops

        improvement = (peak_performance / self.baseline_gflops - 1) * 100

        if meets_sustained_target:
            conclusion = "🎉 BREAKTHROUGH: Winograd transforms achieve 1000+ GFLOPS sustained!"
            recommendation = "✅ ACCEPT: Integrate Winograd transforms into production"
            status = "SUCCESS"
        elif meets_peak_target:
            conclusion = "⭐ PROMISING: Winograd achieves 1000+ GFLOPS peak performance"
            recommendation = "⚠️ CONDITIONAL: Optimize for sustained performance"
            status = "PARTIAL_SUCCESS"
        elif peak_performance >= self.baseline_gflops * 1.1:
            conclusion = "📈 IMPROVEMENT: Winograd provides meaningful performance gains"
            recommendation = "🔄 CONTINUE: Further optimize Winograd implementation"
            status = "IMPROVEMENT"
        else:
            conclusion = "❌ INSUFFICIENT: Winograd does not meet performance requirements"
            recommendation = "⏭️ NEXT: Move to Phase 12 (Mixed Precision Optimizations)"
            status = "FAILURE"

        report['conclusions'] = {
            'status': status,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'peak_performance_gflops': peak_performance,
            'sustained_performance_gflops': sustained_performance,
            'improvement_over_baseline_percent': improvement,
            'meets_1000_gflops_target': meets_peak_target,
            'meets_sustained_1000_gflops': meets_sustained_target
        }

        # Save report
        report_file = Path(__file__).parent / "results" / "winograd_performance_report.json"
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
    logger.info("📊 Winograd Transform Performance Benchmark")
    logger.info("=" * 55)

    benchmarker = WinogradBenchmarker()

    try:
        report = benchmarker.generate_performance_report()

        # Print summary
        print("\n" + "="*70)
        print("🎯 WINOGRAD PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Baseline Performance: {benchmarker.baseline_gflops:.2f} GFLOPS")
        print(f"Target Performance: {benchmarker.target_gflops:.2f} GFLOPS")
        print(f"Peak Achieved: {report['conclusions']['peak_performance_gflops']:.2f} GFLOPS")
        print(f"Sustained Achieved: {report['conclusions']['sustained_performance_gflops']:.2f} GFLOPS")
        print(".1f"        print(f"Status: {report['conclusions']['status']}")
        print(f"Conclusion: {report['conclusions']['conclusion']}")
        print(f"Recommendation: {report['conclusions']['recommendation']}")
        print("="*70)

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()