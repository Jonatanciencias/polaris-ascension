#!/usr/bin/env python3
"""
Sweet Spot Refinement Experiment
Phase 2.1 Extension - Systematic Search for Optimal Matrix Size

Objective: Test matrix sizes around 1400 to find if there's a better sweet spot
Current best: 778 GFLOPS @ 1400×1400 (tile20)

Test plan:
- Sizes: 1350, 1375, 1400, 1425, 1450
- Kernel: tile20 (gemm_tile20_production.cl)
- Validation: Correctness + performance
- Output: CSV + visual summary

Author: AMD RX 590 GEMM Optimization Project
Date: February 2026
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pyopencl as cl
except ImportError:
    print("❌ Error: PyOpenCL not installed")
    print("Install: pip install pyopencl")
    sys.exit(1)


class SweetSpotBenchmark:
    """Professional benchmark runner for sweet spot refinement"""
    
    def __init__(self):
        """Initialize OpenCL context and load tile20 kernel"""
        # Setup OpenCL
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Load tile20 kernel
        kernel_path = Path(__file__).parent.parent.parent / "src" / "kernels" / "gemm_tile20_production.cl"
        
        if not kernel_path.exists():
            raise FileNotFoundError(f"tile20 kernel not found: {kernel_path}")
        
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()
        
        self.program = cl.Program(self.ctx, kernel_code).build()
        self.kernel = self.program.gemm_tile20_optimized
        
        # Configuration
        self.tile_size = 20
        self.local_size = (10, 10)
        
        print(f"✓ Initialized: {self.device.name}")
        print(f"✓ Kernel: tile20 (10×10 workgroup, 20×20 tile)")
        print()
    
    def run_benchmark(self, M, N, K, num_warmup=3, num_runs=10):
        """
        Run professional benchmark for given matrix size
        
        Args:
            M, N, K: Matrix dimensions
            num_warmup: Warmup iterations (discard)
            num_runs: Measurement iterations
            
        Returns:
            dict with performance metrics
        """
        # Generate test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        
        # GPU buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)
        
        # Global work size
        global_size = (
            ((N + self.tile_size - 1) // self.tile_size) * self.local_size[0],
            ((M + self.tile_size - 1) // self.tile_size) * self.local_size[1]
        )
        
        # Warmup
        for _ in range(num_warmup):
            self.kernel(
                self.queue, global_size, self.local_size,
                np.int32(M), np.int32(N), np.int32(K),
                np.float32(1.0),
                a_buf, b_buf,
                np.float32(0.0),
                c_buf
            )
        self.queue.finish()
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            event = self.kernel(
                self.queue, global_size, self.local_size,
                np.int32(M), np.int32(N), np.int32(K),
                np.float32(1.0),
                a_buf, b_buf,
                np.float32(0.0),
                c_buf
            )
            event.wait()
            elapsed = (event.profile.end - event.profile.start) * 1e-9
            times.append(elapsed)
        
        # Get result for correctness check
        cl.enqueue_copy(self.queue, C, c_buf).wait()
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        
        gflops = (2.0 * M * N * K) / (avg_time * 1e9)
        gflops_best = (2.0 * M * N * K) / (min_time * 1e9)
        
        # Correctness check
        C_ref = A @ B
        max_error = np.max(np.abs(C - C_ref))
        relative_error = max_error / np.max(np.abs(C_ref))
        
        return {
            'M': M, 'N': N, 'K': K,
            'gflops_avg': gflops,
            'gflops_best': gflops_best,
            'time_avg_ms': avg_time * 1000,
            'time_std_ms': std_time * 1000,
            'max_error': max_error,
            'relative_error': relative_error,
            'correct': max_error < 0.001
        }


def main():
    """Run sweet spot refinement experiment"""
    
    print("=" * 70)
    print("SWEET SPOT REFINEMENT EXPERIMENT")
    print("=" * 70)
    print()
    print("Objective: Find optimal matrix size for tile20 kernel")
    print("Current best: 778 GFLOPS @ 1400×1400")
    print("Testing: 1350, 1375, 1400, 1425, 1450")
    print()
    
    # Initialize benchmark
    try:
        benchmark = SweetSpotBenchmark()
    except Exception as e:
        print(f"❌ Error initializing benchmark: {e}")
        return 1
    
    # Test sizes
    test_sizes = [1350, 1375, 1400, 1425, 1450]
    results = []
    
    print("=" * 70)
    print("RUNNING BENCHMARKS")
    print("=" * 70)
    print()
    
    for i, size in enumerate(test_sizes, 1):
        print(f"[{i}/{len(test_sizes)}] Testing {size}×{size}...", end=" ", flush=True)
        
        try:
            result = benchmark.run_benchmark(size, size, size, num_warmup=3, num_runs=10)
            results.append(result)
            
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {result['gflops_avg']:.1f} GFLOPS (error: {result['max_error']:.6f})")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append({
                'M': size, 'N': size, 'K': size,
                'gflops_avg': 0, 'gflops_best': 0,
                'max_error': float('inf'), 'correct': False,
                'error': str(e)
            })
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Find best
    valid_results = [r for r in results if r['correct']]
    
    if not valid_results:
        print("❌ No valid results! All tests failed correctness check.")
        return 1
    
    best = max(valid_results, key=lambda r: r['gflops_avg'])
    baseline_1400 = next((r for r in results if r['M'] == 1400), None)
    
    # Print table
    print(f"{'Size':>6} | {'GFLOPS (avg)':>12} | {'GFLOPS (best)':>13} | {'Error':>12} | Status")
    print("-" * 70)
    
    for r in results:
        if 'error' in r:
            print(f"{r['M']:>6} | {'FAILED':>12} | {'FAILED':>13} | {'N/A':>12} | ❌")
        else:
            marker = " 🏆" if r == best else " ⭐" if r['M'] == 1400 else ""
            status = "✅" if r['correct'] else "❌"
            print(f"{r['M']:>6} | {r['gflops_avg']:>12.1f} | {r['gflops_best']:>13.1f} | "
                  f"{r['max_error']:>12.6f} | {status}{marker}")
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    print(f"🏆 Best performance: {best['gflops_avg']:.1f} GFLOPS @ {best['M']}×{best['N']}")
    print(f"   Best run: {best['gflops_best']:.1f} GFLOPS")
    print(f"   Time: {best['time_avg_ms']:.2f} ± {best['time_std_ms']:.2f} ms")
    print(f"   Error: {best['max_error']:.6f} (relative: {best['relative_error']:.2e})")
    print()
    
    if baseline_1400:
        delta = best['gflops_avg'] - baseline_1400['gflops_avg']
        delta_pct = (delta / baseline_1400['gflops_avg']) * 100
        
        if best['M'] == 1400:
            print(f"✓ Baseline (1400×1400) confirmed as best sweet spot")
            print(f"  Performance: {baseline_1400['gflops_avg']:.1f} GFLOPS")
        elif delta > 0:
            print(f"🎯 NEW SWEET SPOT FOUND!")
            print(f"   {best['M']}×{best['N']}: {best['gflops_avg']:.1f} GFLOPS")
            print(f"   1400×1400: {baseline_1400['gflops_avg']:.1f} GFLOPS")
            print(f"   Improvement: +{delta:.1f} GFLOPS (+{delta_pct:.1f}%)")
        else:
            print(f"✓ Baseline (1400×1400) remains best sweet spot")
            print(f"   1400×1400: {baseline_1400['gflops_avg']:.1f} GFLOPS")
            print(f"   {best['M']}×{best['N']}: {best['gflops_avg']:.1f} GFLOPS ({delta:.1f} GFLOPS, {delta_pct:.1f}%)")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    
    if best['M'] == 1400:
        print("✅ Current sweet spot (1400×1400) is optimal")
        print("   No better size found in tested range")
        print("   Recommendation: Keep 1400 as sweet spot")
    else:
        improvement = ((best['gflops_avg'] - baseline_1400['gflops_avg']) / baseline_1400['gflops_avg']) * 100
        if improvement > 1.0:
            print(f"🎯 NEW OPTIMAL SIZE: {best['M']}×{best['N']}")
            print(f"   Improvement: +{improvement:.1f}% vs 1400×1400")
            print(f"   Recommendation: Update sweet spot to {best['M']}")
        else:
            print("✅ Marginal difference (< 1%) - keep 1400 for consistency")
            print(f"   1400: {baseline_1400['gflops_avg']:.1f} GFLOPS")
            print(f"   {best['M']}: {best['gflops_avg']:.1f} GFLOPS ({improvement:+.1f}%)")
    
    print()
    
    # Save results to CSV
    output_file = Path(__file__).parent / "sweet_spot_refinement_results.csv"
    with open(output_file, 'w') as f:
        f.write("size,gflops_avg,gflops_best,time_ms,std_ms,max_error,relative_error,correct\n")
        for r in results:
            if 'error' not in r:
                f.write(f"{r['M']},{r['gflops_avg']:.2f},{r['gflops_best']:.2f},"
                       f"{r['time_avg_ms']:.3f},{r['time_std_ms']:.3f},"
                       f"{r['max_error']:.6e},{r['relative_error']:.6e},{r['correct']}\n")
    
    print(f"📊 Results saved to: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
