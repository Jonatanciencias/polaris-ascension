#!/usr/bin/env python3
"""
Wave-Level Optimized GEMM Benchmark
Técnica 3: GCN 4.0 Polaris 10 Optimizations

Target: 900-950 GFLOPS (15-23% improvement over Phase 1 baseline)
"""

import os
import sys
import time
import numpy as np
import pyopencl as cl
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class WaveOptimizedGEMM:
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.kernels = {}

    def initialize_opencl(self):
        """Initialize OpenCL context and queue"""
        try:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            print("✓ OpenCL initialized for wave-optimized GEMM")
            return True
        except Exception as e:
            print(f"✗ OpenCL initialization failed: {e}")
            return False

    def load_kernel(self, kernel_name: str) -> bool:
        """Load the wave-optimized kernel"""
        try:
            kernel_path = os.path.join('src', 'opencl', 'kernels', 'gemm_wave_optimized.cl')
            with open(kernel_path, 'r') as f:
                source = f.read()

            program = cl.Program(self.ctx, source).build()

            if hasattr(program, kernel_name):
                self.kernels[kernel_name] = getattr(program, kernel_name)
                print(f"✓ Kernel {kernel_name} loaded successfully")
                return True
            else:
                print(f"✗ Kernel {kernel_name} not found")
                return False

        except Exception as e:
            print(f"✗ Failed to load kernel {kernel_name}: {e}")
            return False

    def benchmark_kernel(self, kernel_name: str, M: int, N: int, K: int,
                        num_runs: int = 5) -> Dict[str, float]:
        """Benchmark the wave-optimized kernel"""
        if kernel_name not in self.kernels:
            return {'error': f'Kernel {kernel_name} not loaded'}

        kernel = self.kernels[kernel_name]

        # Adjust dimensions for tile size
        TILE_SIZE = 32
        WG_SIZE_X, WG_SIZE_Y = 16, 16

        M_adj = ((M + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        N_adj = ((N + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

        # Create test data
        A = np.random.randn(M_adj, K).astype(np.float32)
        B = np.random.randn(K, N_adj).astype(np.float32)
        C = np.zeros((M_adj, N_adj), dtype=np.float32)

        # OpenCL buffers
        A_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

        # Set kernel arguments
        kernel.set_args(np.int32(M_adj), np.int32(N_adj), np.int32(K),
                       np.float32(1.0), np.float32(0.0), A_buf, B_buf, C_buf)

        # Workgroup configuration optimized for GCN 4.0
        global_size = (M_adj // TILE_SIZE * WG_SIZE_X, N_adj // TILE_SIZE * WG_SIZE_Y)
        local_size = (WG_SIZE_X, WG_SIZE_Y)

        # Warmup
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()

        # Benchmark runs
        times = []
        for _ in range(num_runs):
            self.queue.finish()
            start = time.time()

            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()

            end = time.time()
            times.append((end - start) * 1000)

        # Calculate metrics
        avg_time = np.mean(times)
        operations = 2 * M_adj * N_adj * K
        gflops = (operations / (avg_time / 1000)) / 1e9

        return {
            'avg_time_ms': avg_time,
            'gflops': gflops,
            'matrix_size': f'{M_adj}x{N_adj}x{K}',
            'tile_size': TILE_SIZE,
            'workgroup_size': f'{WG_SIZE_X}x{WG_SIZE_Y}',
            'occupancy': 'Max (256 threads)',
            'kernel': kernel_name
        }

def run_wave_optimization_benchmark():
    """Run comprehensive benchmark for wave-optimized GEMM"""
    print("🌊 WAVE-LEVEL GCN 4.0 OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print("Técnica 3: Polaris 10 Architecture-Specific Optimizations")
    print("Target: 900-950 GFLOPS (15-23% over Phase 1)")

    benchmark = WaveOptimizedGEMM()

    if not benchmark.initialize_opencl():
        return None

    if not benchmark.load_kernel('gemm_wave_optimized'):
        return None

    # Test sizes relevant for Polaris 10
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    results = {}

    for M, N, K in sizes:
        print(f"\n--- Testing {M}x{N}x{K} matrices ---")
        result = benchmark.benchmark_kernel('gemm_wave_optimized', M, N, K)
        results[f'{M}x{N}x{K}'] = result

        if 'error' not in result:
            print(f"    Performance: {result['gflops']:.2f} GFLOPS")
            print(f"    Workgroup: {result['workgroup_size']}")
            print(f"    Tile Size: {result['tile_size']}x{result['tile_size']}")
            print(f"    Occupancy: {result['occupancy']}")
        else:
            print(f"    Error: {result['error']}")

    return results

def compare_with_phase1(results: Dict) -> Dict:
    """Compare wave-optimized results with Phase 1 baseline"""
    print("\n" + "="*60)
    print("📊 COMPARISON WITH PHASE 1 BASELINE")
    print("="*60)

    # Phase 1 baseline results (from previous benchmarks)
    phase1_baseline = {
        '512x512x512': 300.0,    # Estimated from previous runs
        '1024x1024x1024': 775.0, # Actual Phase 1 result
        '2048x2048x2048': 1200.0 # Estimated peak
    }

    comparison = {}

    for size_key, result in results.items():
        if 'error' in result:
            continue

        wave_gflops = result['gflops']
        baseline_gflops = phase1_baseline.get(size_key, 0)

        if baseline_gflops > 0:
            speedup = wave_gflops / baseline_gflops
            improvement = (speedup - 1.0) * 100

            comparison[size_key] = {
                'wave_gflops': wave_gflops,
                'baseline_gflops': baseline_gflops,
                'speedup': speedup,
                'improvement_percent': improvement
            }

            print(f"\n{size_key}:")
            print(f"    Wave Optimized: {wave_gflops:.1f} GFLOPS")
            print(f"    Phase 1 Baseline: {baseline_gflops:.1f} GFLOPS")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Improvement: +{improvement:.1f}%")
    return comparison

def main():
    """Main benchmark execution"""
    results = run_wave_optimization_benchmark()

    if results:
        comparison = compare_with_phase1(results)

        # Save results
        output = {
            'benchmark_info': {
                'technique': 'Técnica 3: Wave-level GCN 4.0 Optimizations',
                'target': '900-950 GFLOPS',
                'hardware': 'AMD Radeon RX 590 (Polaris 10)',
                'optimizations': [
                    'Workgroup (16,16) = 256 threads max occupancy',
                    'Tile size 32x32 optimal LDS usage',
                    'Wave scheduling for 64-wavefront architecture',
                    'LDS bank conflict minimization'
                ],
                'timestamp': datetime.now().isoformat()
            },
            'results': results,
            'comparison': comparison
        }

        filename = f"wave_optimization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✓ Results saved to {filename}")

        # Summary
        print("\n" + "="*60)
        print("🎯 WAVE OPTIMIZATION SUMMARY")
        print("="*60)

        if comparison:
            avg_improvement = np.mean([comp['improvement_percent'] for comp in comparison.values()])
            print(f"Average Improvement: {avg_improvement:.1f}%")
            if avg_improvement >= 15:
                print("✅ TARGET ACHIEVED: 15%+ improvement over Phase 1")
                print("   Wave-level optimizations successful!")
            elif avg_improvement >= 5:
                print("⚠️ PARTIAL SUCCESS: Some improvement achieved")
                print("   Further optimizations needed")
            else:
                print("❌ TARGET NOT MET: No significant improvement")
                print("   Need different optimization approach")

        print("\n💡 NEXT STEPS:")
        print("   1. If successful: Proceed to Técnica 1+ (Block Recursive)")
        print("   2. If partial: Add async memory operations")
        print("   3. If failed: Focus on Sparse kernels for specific use cases")

if __name__ == '__main__':
    main()