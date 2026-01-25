#!/usr/bin/env python3
"""
Simple Benchmark for gemm_wave_fixed.cl
Tests the corrected collaborative loading logic
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class WaveFixedBenchmark:
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.kernels = {}

    def initialize_opencl(self):
        """Initialize OpenCL context and queue"""
        try:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            print("✓ OpenCL initialized for wave-fixed GEMM")
            return True
        except Exception as e:
            print(f"✗ OpenCL initialization failed: {e}")
            return False

    def load_kernel(self, kernel_file: str) -> bool:
        """Load the wave-fixed kernel"""
        try:
            kernel_path = os.path.join('src', 'opencl', 'kernels', kernel_file)
            with open(kernel_path, 'r') as f:
                source = f.read()

            program = cl.Program(self.ctx, source).build()

            # Load the kernel
            if hasattr(program, 'gemm_wave_optimized'):
                self.kernels['gemm_wave_optimized'] = getattr(program, 'gemm_wave_optimized')
                print(f"✓ Kernel gemm_wave_optimized loaded successfully")
                return True
            else:
                print(f"✗ Kernel gemm_wave_optimized not found in {kernel_file}")
                return False
        except Exception as e:
            print(f"✗ Kernel loading failed: {e}")
            return False

    def benchmark_kernel(self, kernel_name: str, M: int, N: int, K: int, iterations: int = 3) -> Dict:
        """Benchmark the kernel with numerical validation"""
        print(f"\n--- Testing {M}x{N}x{K} matrices ---")

        # Create test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_gpu = np.zeros((M, N), dtype=np.float32)

        # Reference computation
        C_ref = np.dot(A, B)

        # OpenCL buffers
        A_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, C_gpu.nbytes)

        # Kernel parameters
        TILE_SIZE = 16
        WG_SIZE_X = 16
        WG_SIZE_Y = 16
        alpha = 1.0
        beta = 0.0

        global_size = (M // TILE_SIZE * WG_SIZE_X, N // TILE_SIZE * WG_SIZE_Y)
        local_size = (WG_SIZE_X, WG_SIZE_Y)

        kernel = self.kernels[kernel_name]

        # Warmup
        kernel(self.queue, global_size, local_size, np.int32(M), np.int32(N), np.int32(K), np.float32(alpha), np.float32(beta), A_buf, B_buf, C_buf)

        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            kernel(self.queue, global_size, local_size, np.int32(M), np.int32(N), np.int32(K), np.float32(alpha), np.float32(beta), A_buf, B_buf, C_buf)
            self.queue.finish()
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        gflops = (2 * M * N * K) / (avg_time * 1e9)

        # Copy result back
        cl.enqueue_copy(self.queue, C_gpu, C_buf).wait()

        # Numerical validation
        error = np.max(np.abs(C_gpu - C_ref))
        max_error = np.max(np.abs(C_ref))
        relative_error = error / max_error if max_error > 0 else error

        print(f"  {kernel_name}: {gflops:.2f} GFLOPS")
        print(f"  Numerical error: {error:.2e} (relative: {relative_error:.2e})")

        return {
            'gflops': float(gflops),
            'error': float(error),
            'relative_error': float(relative_error),
            'time': float(avg_time)
        }

def main():
    benchmark = WaveFixedBenchmark()

    if not benchmark.initialize_opencl():
        return

    if not benchmark.load_kernel('gemm_wave_fixed.cl'):
        return

    # Test different sizes
    sizes = [64, 128, 256]
    results = {}

    for size in sizes:
        M = N = K = size
        result = benchmark.benchmark_kernel('gemm_wave_optimized', M, N, K, iterations=3)
        results[f"{size}x{size}x{size}"] = result

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wave_fixed_benchmark_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {filename}")

    # Summary
    print("\n=== WAVE FIXED KERNEL BENCHMARK SUMMARY ===")
    for size, result in results.items():
        status = "✓ PASS" if result['relative_error'] < 1e-4 else "✗ FAIL"
        print(f"  {size}: {result['gflops']:.2f} GFLOPS, Error: {result['relative_error']:.2e} {status}")

if __name__ == "__main__":
    main()