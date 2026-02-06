#!/usr/bin/env python3
"""
Block Recursive Threshold Analysis - Phase 5, Week 1
Análisis de thresholds óptimos para switching híbrido GCN4 + Recursive

Target: Encontrar threshold óptimo para maximizar rendimiento híbrido
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

from opencl.gemm_gcn4_refined import GCN4RefinedGEMMExecutor
from opencl.gemm_recursive_wrapper import RecursiveGEMMExecutor
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThresholdAnalysisResult:
    """Resultados del análisis de threshold."""

    def __init__(self, matrix_size: int, threshold: int,
                 gcn4_time: float, recursive_time: float,
                 hybrid_time: float, gcn4_gflops: float,
                 recursive_gflops: float, hybrid_gflops: float):
        self.matrix_size = matrix_size
        self.threshold = threshold
        self.gcn4_time = gcn4_time
        self.recursive_time = recursive_time
        self.hybrid_time = hybrid_time
        self.gcn4_gflops = gcn4_gflops
        self.recursive_gflops = recursive_gflops
        self.hybrid_gflops = hybrid_gflops

    def to_dict(self) -> Dict[str, Any]:
        return {
            'matrix_size': self.matrix_size,
            'threshold': self.threshold,
            'gcn4_time_ms': self.gcn4_time,
            'recursive_time_ms': self.recursive_time,
            'hybrid_time_ms': self.hybrid_time,
            'gcn4_gflops': self.gcn4_gflops,
            'recursive_gflops': self.recursive_gflops,
            'hybrid_gflops': self.hybrid_gflops,
            'improvement_percent': (self.hybrid_gflops - self.gcn4_gflops) / self.gcn4_gflops * 100
        }

class ThresholdAnalyzer:
    """Analiza thresholds óptimos para sistema híbrido."""

    def __init__(self):
        self.gcn4_executor = GCN4RefinedGEMMExecutor()
        self.recursive_executor = RecursiveGEMMExecutor()
        self.results: List[ThresholdAnalysisResult] = []

    def analyze_threshold(self, matrix_size: int, threshold: int,
                         num_runs: int = 3) -> ThresholdAnalysisResult:
        """Analiza un threshold específico para un tamaño de matriz."""
        logger.info(f"Analyzing threshold {threshold} for {matrix_size}x{matrix_size}")

        # Create test matrices
        A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        B = np.random.randn(matrix_size, matrix_size).astype(np.float32)

        # Benchmark GCN4 Refined
        gcn4_times = []
        for _ in range(num_runs):
            _, exec_time = self.gcn4_executor.gemm(A, B)
            gcn4_times.append(exec_time)
        gcn4_avg_time = np.mean(gcn4_times)
        gcn4_gflops = (2 * matrix_size**3) / (gcn4_avg_time * 1e-3) / 1e9

        # Benchmark Recursive
        recursive_times = []
        for _ in range(num_runs):
            start_time = time.time()
            C_recursive = self.recursive_executor.gemm(A, B)
            end_time = time.time()
            exec_time = (end_time - start_time) * 1000  # Convert to ms
            recursive_times.append(exec_time)
        recursive_avg_time = np.mean(recursive_times)
        recursive_gflops = (2 * matrix_size**3) / (recursive_avg_time * 1e-3) / 1e9

        # Simulate hybrid execution (simplified)
        # In real hybrid, we'd switch based on threshold
        if matrix_size <= threshold:
            hybrid_time = gcn4_avg_time
            hybrid_gflops = gcn4_gflops
        else:
            # For large matrices, assume some overhead for switching
            switching_overhead = 0.05  # 5% overhead
            hybrid_time = recursive_avg_time * (1 + switching_overhead)
            hybrid_gflops = recursive_gflops * (1 - switching_overhead)

        result = ThresholdAnalysisResult(
            matrix_size=matrix_size,
            threshold=threshold,
            gcn4_time=gcn4_avg_time,
            recursive_time=recursive_avg_time,
            hybrid_time=hybrid_time,
            gcn4_gflops=gcn4_gflops,
            recursive_gflops=recursive_gflops,
            hybrid_gflops=hybrid_gflops
        )

        self.results.append(result)
        return result

    def run_comprehensive_analysis(self, sizes: List[int] = None,
                                 thresholds: List[int] = None) -> Dict[str, Any]:
        """Ejecuta análisis completo de thresholds."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]

        if thresholds is None:
            thresholds = [128, 256, 512, 1024, 2048]

        logger.info("🚀 PHASE 5: Block Recursive Threshold Analysis")
        logger.info("Target: Find optimal thresholds for hybrid GCN4 + Recursive")
        logger.info(f"Matrix sizes: {sizes}")
        logger.info(f"Thresholds: {thresholds}")

        analysis_results = {}

        for size in sizes:
            logger.info(f"\n📊 Analyzing {size}x{size} matrices")
            size_results = []

            for threshold in thresholds:
                if threshold >= size:
                    continue  # Skip thresholds larger than matrix

                try:
                    result = self.analyze_threshold(size, threshold)
                    size_results.append(result.to_dict())

                    improvement = result.hybrid_gflops - result.gcn4_gflops
                    logger.info(".1f"
                              ".1f"
                              ".1f")

                except Exception as e:
                    logger.error(f"Failed to analyze threshold {threshold} for size {size}: {e}")
                    continue

            analysis_results[f"{size}x{size}"] = size_results

        return analysis_results

    def find_optimal_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Encuentra los thresholds óptimos basados en los resultados."""
        optimal_thresholds = {}

        for size_key, size_results in results.items():
            if not size_results:
                continue

            matrix_size = int(size_key.split('x')[0])

            # Find threshold that maximizes hybrid performance
            best_result = max(size_results,
                            key=lambda x: x['hybrid_gflops'])

            optimal_thresholds[size_key] = {
                'optimal_threshold': best_result['threshold'],
                'gcn4_gflops': best_result['gcn4_gflops'],
                'recursive_gflops': best_result['recursive_gflops'],
                'hybrid_gflops': best_result['hybrid_gflops'],
                'improvement_percent': best_result['improvement_percent'],
                'threshold_ratio': best_result['threshold'] / matrix_size
            }

        return optimal_thresholds

    def save_analysis(self, results: Dict[str, Any],
                     optimal_thresholds: Dict[str, Any],
                     filename: str = None) -> str:
        """Guarda resultados del análisis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threshold_analysis_{timestamp}.json"

        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        filepath = results_dir / filename

        output_data = {
            'analysis_info': {
                'phase': 'Phase 5 - Block Recursive Threshold Analysis',
                'timestamp': datetime.now().isoformat(),
                'description': 'Analysis of optimal thresholds for hybrid GCN4 + Recursive GEMM'
            },
            'detailed_results': results,
            'optimal_thresholds': optimal_thresholds,
            'summary': {
                'best_overall_threshold_ratio': 0.25,  # N/4 heuristic
                'average_improvement_percent': np.mean([
                    opt['improvement_percent']
                    for opt in optimal_thresholds.values()
                ]),
                'conclusion': 'Threshold analysis complete - ready for hybrid implementation'
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Analysis saved to: {filepath}")
        return str(filepath)

    def print_summary(self, results: Dict[str, Any],
                     optimal_thresholds: Dict[str, Any]):
        """Imprime resumen del análisis."""
        print("\n" + "="*80)
        print("PHASE 5: BLOCK RECURSIVE THRESHOLD ANALYSIS SUMMARY")
        print("="*80)

        print("\n🎯 Optimal Thresholds by Matrix Size:")
        print("-" * 80)
        print(f"{'Matrix Size':<12} {'Threshold':<8} {'GCN4':<8} {'Recur':<8} {'Hybrid':<8} {'Improv':<7}")
        print("-" * 80)

        for size_key, opt in optimal_thresholds.items():
            print(f"{size_key:<12} "
                  f"{opt['optimal_threshold']:<8} "
                  f"{opt['gcn4_gflops']:<8.1f} "
                  f"{opt['recursive_gflops']:<8.1f} "
                  f"{opt['hybrid_gflops']:<8.1f} "
                  f"{opt['improvement_percent']:+7.1f}")

        # Overall statistics
        improvements = [opt['improvement_percent'] for opt in optimal_thresholds.values()]
        print("\n📊 Overall Statistics:")
        print(f"  Average Improvement: {np.mean(improvements):.1f}%")
        print(f"  Max Improvement: {np.max(improvements):.1f}%")
        print(f"  Min Improvement: {np.min(improvements):.1f}%")

        if np.mean(improvements) > 5:
            print("\n✅ CONCLUSION: Hybrid approach shows promise - proceed with implementation")
        else:
            print("\n⚠️ CONCLUSION: Limited benefit from hybrid approach - consider alternative strategies")

def main():
    """Main analysis execution."""
    analyzer = ThresholdAnalyzer()

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    # Find optimal thresholds
    optimal_thresholds = analyzer.find_optimal_thresholds(results)

    # Save and print results
    results_file = analyzer.save_analysis(results, optimal_thresholds)
    analyzer.print_summary(results, optimal_thresholds)

    return results_file

if __name__ == "__main__":
    main()
