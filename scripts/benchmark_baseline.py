#!/usr/bin/env python3
"""
Task 1.1.2.3 - Performance Baseline Benchmark

Measures GFLOPS for different matrix sizes to establish baseline performance.

Benchmarks:
  - Sizes: 256, 512, 1024, 2048
  - Iterations: 10 per size
  - Metrics: Time, GFLOPS, error, stability
"""

import sys
import numpy as np
import logging
import time
import json
from pathlib import Path
from statistics import mean, stdev

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.opencl.hybrid_gemm import HybridGEMMExecutor


class BaselineBenchmark:
    """Baseline performance benchmarking for hybrid GEMM."""
    
    # Hardware specifications
    RX590_SPECS = {
        'peak_gflops': 6170,  # 2304 cores × 2.68 GHz × 1 FMA per core / 1e9
        'peak_bandwidth_gbs': 256,
        'cores': 2304,
        'clock_mhz': 2680
    }
    
    BASELINE_GFLOPS = 542  # Current achievement with float4
    
    def __init__(self):
        """Initialize benchmark."""
        logger.info("Initializing Baseline Benchmark...")
        self.executor = HybridGEMMExecutor()
        self.results = {
            'timestamp': time.time(),
            'hardware': self.RX590_SPECS,
            'baseline_gflops': self.BASELINE_GFLOPS,
            'benchmarks': {}
        }
    
    def benchmark_size(self, size, iterations=10):
        """Benchmark a specific matrix size."""
        logger.info(f"\nBenchmarking {size}×{size} matrix ({iterations} iterations)...")
        
        np.random.seed(42 + size)  # Reproducible randomness
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        times = []
        errors = []
        
        # Warmup
        _ = self.executor(A, B)
        
        # Benchmark iterations
        for i in range(iterations):
            start = time.perf_counter()
            C_gpu = self.executor(A, B)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)
            
            # Calculate error (every 3 iterations to save time)
            if i % 3 == 0:
                C_ref = A @ B
                error_rel = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
                errors.append(error_rel)
            
            if (i + 1) % 3 == 0:
                logger.info(f"  Progress: {i+1}/{iterations}")
        
        # Calculate statistics
        time_mean = mean(times)
        time_std = stdev(times) if len(times) > 1 else 0
        
        # GFLOPS calculation
        flops = 2 * size**3
        gflops = flops / (time_mean / 1000) / 1e9
        gflops_std = (time_std / time_mean) * gflops if time_mean > 0 else 0
        
        # Error statistics
        if errors:
            error_mean = mean(errors)
            error_max = max(errors)
        else:
            error_mean = 0
            error_max = 0
        
        # Stability metric
        cv = (time_std / time_mean) * 100 if time_mean > 0 else 0  # Coefficient of variation
        
        # Bandwidth and occupancy
        bandwidth_percent = self._estimate_bandwidth_percent(size, time_mean)
        occupancy_percent = self._estimate_occupancy_percent(size, gflops)
        
        # Speedup vs baseline
        speedup = gflops / self.BASELINE_GFLOPS
        
        result = {
            'size': size,
            'iterations': iterations,
            'time_ms': float(time_mean),
            'time_std_ms': float(time_std),
            'gflops': float(gflops),
            'gflops_std': float(gflops_std),
            'error_rel': float(error_mean),
            'error_rel_max': float(error_max),
            'stability_cv': float(cv),
            'bandwidth_util_percent': float(bandwidth_percent),
            'occupancy_percent': float(occupancy_percent),
            'speedup_vs_baseline': float(speedup)
        }
        
        return result
    
    def _estimate_bandwidth_percent(self, size, time_ms):
        """Estimate memory bandwidth utilization percent."""
        # Data moved: 3 matrices × size² elements × 4 bytes
        # Plus K iterations of tile loading
        data_moved_bytes = 3 * size * size * 4
        
        bandwidth_used_gbs = data_moved_bytes / (1024**3) / (time_ms / 1000)
        percent = (bandwidth_used_gbs / self.RX590_SPECS['peak_bandwidth_gbs']) * 100
        
        return min(percent, 100)
    
    def _estimate_occupancy_percent(self, size, gflops):
        """Estimate GPU occupancy based on GFLOPS."""
        # Theoretical: peak_gflops / 2 (accounting for memory dependencies)
        theoretical_max = self.RX590_SPECS['peak_gflops'] / 2
        percent = (gflops / theoretical_max) * 100
        
        return min(percent, 100)
    
    def run_benchmark_suite(self, sizes=None, iterations=10):
        """Run benchmark suite for multiple matrix sizes."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]
        
        logger.info("\n" + "="*100)
        logger.info("BASELINE PERFORMANCE BENCHMARK - TASK 1.1.2.3")
        logger.info("="*100)
        logger.info(f"Hardware: Radeon RX 590")
        logger.info(f"Peak GFLOPS: {self.RX590_SPECS['peak_gflops']}")
        logger.info(f"Baseline (current): {self.BASELINE_GFLOPS} GFLOPS")
        logger.info("="*100)
        
        # Header
        logger.info(f"\n{'Size':>6} {'Time (ms)':>14} {'GFLOPS':>12} {'Stability':>12} "
                   f"{'Error':>12} {'BW%':>8} {'Occ%':>8} {'Speedup':>10}")
        logger.info("-"*100)
        
        for size in sizes:
            try:
                result = self.benchmark_size(size, iterations)
                self.results['benchmarks'][size] = result
                
                logger.info(
                    f"{size:6d} {result['time_ms']:7.3f}±{result['time_std_ms']:5.3f} "
                    f"{result['gflops']:9.1f}±{result['gflops_std']:6.1f} "
                    f"{result['stability_cv']:10.2f}% "
                    f"{result['error_rel']:11.2e} "
                    f"{result['bandwidth_util_percent']:7.1f}% "
                    f"{result['occupancy_percent']:7.1f}% "
                    f"{result['speedup_vs_baseline']:9.2f}x"
                )
                
            except Exception as e:
                logger.error(f"Benchmark failed for size {size}: {e}", exc_info=True)
                self.results['benchmarks'][size] = {'error': str(e)}
        
        logger.info("-"*100)
        
        return self.results['benchmarks']
    
    def generate_summary(self):
        """Generate summary statistics."""
        logger.info("\n" + "="*100)
        logger.info("SUMMARY")
        logger.info("="*100)
        
        benchmarks = self.results['benchmarks']
        
        if not benchmarks:
            logger.warning("No valid benchmarks to summarize")
            return
        
        valid_results = [b for b in benchmarks.values() if 'gflops' in b]
        
        if not valid_results:
            logger.warning("No valid results to summarize")
            return
        
        gflops_values = [b['gflops'] for b in valid_results]
        
        logger.info(f"Benchmarks completed: {len(valid_results)}/{len(benchmarks)}")
        logger.info(f"Average GFLOPS: {mean(gflops_values):.1f}")
        logger.info(f"Min GFLOPS: {min(gflops_values):.1f}")
        logger.info(f"Max GFLOPS: {max(gflops_values):.1f}")
        
        # Compare with baseline
        avg_speedup = mean([b['speedup_vs_baseline'] for b in valid_results])
        logger.info(f"Average speedup vs baseline (542 GFLOPS): {avg_speedup:.2f}x")
        
        # Stability check
        stability_values = [b['stability_cv'] for b in valid_results]
        avg_stability = mean(stability_values)
        logger.info(f"Average stability (CV): {avg_stability:.2f}%")
        
        if avg_stability < 1.0:
            logger.info("✅ Stability: EXCELLENT (<1%)")
        elif avg_stability < 2.0:
            logger.info("⚠️  Stability: GOOD (1-2%)")
        else:
            logger.warning("❌ Stability: POOR (>2%)")
        
        # Error check
        error_values = [b['error_rel'] for b in valid_results if b.get('error_rel', 0) > 0]
        if error_values:
            max_error = max(error_values)
            logger.info(f"Maximum error: {max_error:.2e}")
            if max_error < 1e-4:
                logger.info("✅ Accuracy: EXCELLENT (<1e-4)")
            elif max_error < 1e-3:
                logger.info("⚠️  Accuracy: GOOD (<1e-3)")
            else:
                logger.warning(f"❌ Accuracy: POOR ({max_error:.2e})")
        
        # Performance targets
        logger.info("\nPerformance Targets:")
        logger.info(f"  Phase 1 Goal: 700-800 GFLOPS")
        logger.info(f"  Current avg: {mean(gflops_values):.1f} GFLOPS")
        
        if mean(gflops_values) >= 700:
            logger.info("  ✅ PHASE 1 TARGET ACHIEVED")
        elif mean(gflops_values) >= 600:
            logger.info("  ⚠️  Close to target, optimization needed")
        else:
            logger.warning("  ❌ Below target, significant optimization required")
        
        self.results['summary'] = {
            'benchmarks_completed': len(valid_results),
            'avg_gflops': float(mean(gflops_values)),
            'min_gflops': float(min(gflops_values)),
            'max_gflops': float(max(gflops_values)),
            'avg_speedup': float(avg_speedup),
            'avg_stability_cv': float(avg_stability),
            'max_error': float(max_error) if error_values else 0
        }
    
    def save_results(self, output_file='results/baseline_benchmark.json'):
        """Save results to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")


def main():
    """Main entry point."""
    benchmark = BaselineBenchmark()
    
    try:
        # Run benchmarks for standard sizes
        benchmark.run_benchmark_suite(
            sizes=[256, 512, 1024, 2048],
            iterations=10
        )
        
        # Generate summary
        benchmark.generate_summary()
        
        # Save results
        benchmark.save_results()
        
        logger.info("\n✅ BASELINE BENCHMARK COMPLETED")
        return 0
        
    except Exception as e:
        logger.error(f"\nCritical error: {e}", exc_info=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
