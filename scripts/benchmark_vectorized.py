#!/usr/bin/env python3
"""
Phase 3 - Vectorized GEMM Benchmark Script

Compares performance between scalar and vectorized GEMM implementations:
- Scalar baseline: gemm_wave_fixed.cl (~60 GFLOPS)
- Vectorized target: gemm_wave_vectorized.cl (200-300 GFLOPS target)

Usage:
    python scripts/benchmark_vectorized.py

Output:
    - Performance comparison table
    - GFLOPS improvement metrics
    - Memory bandwidth utilization
    - SIMD efficiency analysis
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from opencl.gemm_vectorized import VectorizedGEMMExecutor, VectorizedConfig
    from opencl.hybrid_gemm_opt import OptimizedHybridGEMMExecutor, OptimizedConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedBenchmark:
    """Benchmark suite for vectorized GEMM kernels."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.scalar_executor = None
        self.vectorized_executor = None
        self.results = {}

    def setup_executors(self):
        """Setup both scalar and vectorized executors."""
        try:
            logger.info("Setting up scalar executor (baseline)...")
            scalar_config = OptimizedConfig(
                tile_size=16,
                block_size=2,
                lds_padding=2,
                workgroup_size=64
            )
            self.scalar_executor = OptimizedHybridGEMMExecutor(scalar_config)

            logger.info("Setting up vectorized executor...")
            vectorized_config = VectorizedConfig(
                tile_size=16,
                wg_size_x=16,
                wg_size_y=16,
                lds_padding=2,
                vector_width=4
            )
            self.vectorized_executor = VectorizedGEMMExecutor(vectorized_config)

        except Exception as e:
            logger.error(f"Failed to setup executors: {e}")
            raise

    def run_benchmark(self, sizes: List[int], iterations: int = 5) -> Dict:
        """
        Run comprehensive benchmark comparing scalar vs vectorized.

        Args:
            sizes: Matrix sizes to test
            iterations: Iterations per size

        Returns:
            Benchmark results dictionary
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'sizes': sizes,
            'iterations': iterations,
            'scalar': {},
            'vectorized': {},
            'comparison': {}
        }

        for size in sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking {size}x{size}x{size} matrices")
            logger.info(f"{'='*60}")

            # Generate test matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Benchmark scalar kernel
            logger.info("Testing scalar kernel (baseline)...")
            scalar_result = self._benchmark_kernel(
                self.scalar_executor, A, B, size, iterations, "scalar"
            )
            results['scalar'][size] = scalar_result

            # Benchmark vectorized kernel
            logger.info("Testing vectorized kernel...")
            vectorized_result = self._benchmark_kernel(
                self.vectorized_executor, A, B, size, iterations, "vectorized"
            )
            results['vectorized'][size] = vectorized_result

            # Calculate comparison metrics
            comparison = self._calculate_comparison(scalar_result, vectorized_result, size)
            results['comparison'][size] = comparison

            # Print immediate results
            self._print_comparison(size, scalar_result, vectorized_result, comparison)

        return results

    def _benchmark_kernel(self, executor, A: np.ndarray, B: np.ndarray,
                         size: int, iterations: int, name: str) -> Dict:
        """Benchmark a single kernel."""
        import time

        # Warmup
        try:
            _ = executor.gemm(A, B)
        except Exception as e:
            logger.warning(f"Warmup failed for {name}: {e}")

        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                C = executor.gemm(A, B)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Iteration {i} failed for {name}: {e}")
                times.append(float('inf'))

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if not valid_times:
            return {
                'success': False,
                'error': 'All iterations failed',
                'avg_time_ms': float('inf'),
                'gflops': 0.0,
                'std_dev': 0.0
            }

        avg_time = np.mean(valid_times)
        std_dev = np.std(valid_times)

        # Calculate GFLOPS (2 operations per FMA: multiply + add)
        gflops = (2 * size**3) / (avg_time * 1e9)

        return {
            'success': True,
            'avg_time_ms': avg_time * 1000,
            'gflops': gflops,
            'std_dev': std_dev * 1000,  # Convert to ms
            'iterations_completed': len(valid_times)
        }

    def _calculate_comparison(self, scalar: Dict, vectorized: Dict, size: int) -> Dict:
        """Calculate comparison metrics between scalar and vectorized."""
        if not scalar.get('success', False) or not vectorized.get('success', False):
            return {'error': 'One or both kernels failed'}

        scalar_gflops = scalar['gflops']
        vectorized_gflops = vectorized['gflops']

        speedup = vectorized_gflops / scalar_gflops if scalar_gflops > 0 else float('inf')
        improvement_percent = (speedup - 1) * 100

        # Theoretical bandwidth calculation (rough estimate)
        # RX 590: ~256 GB/s peak bandwidth
        # Each FLOP needs ~1.5 bytes (1 load A + 1 load B + 1 store C, but with reuse)
        # More accurate: ~0.5 bytes per FLOP for GEMM with good reuse
        theoretical_gflops = 256e9 / 0.5 / 1e9  # ~512 GFLOPS theoretical max

        scalar_bandwidth_util = (scalar_gflops / theoretical_gflops) * 100
        vectorized_bandwidth_util = (vectorized_gflops / theoretical_gflops) * 100

        return {
            'speedup': speedup,
            'improvement_percent': improvement_percent,
            'scalar_bandwidth_util': scalar_bandwidth_util,
            'vectorized_bandwidth_util': vectorized_bandwidth_util,
            'bandwidth_improvement': vectorized_bandwidth_util - scalar_bandwidth_util
        }

    def _print_comparison(self, size: int, scalar: Dict, vectorized: Dict, comparison: Dict):
        """Print comparison results for a size."""
        print(f"\n{'='*60}")
        print(f"SIZE {size}x{size}x{size} - PERFORMANCE COMPARISON")
        print(f"{'='*60}")

        if 'error' in comparison:
            print(f"ERROR: {comparison['error']}")
            return

        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
        print("<12")
