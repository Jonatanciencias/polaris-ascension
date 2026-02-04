#!/usr/bin/env python3
"""
Profile Engine Overhead - Identify performance bottlenecks

Compares standalone kernel execution vs integrated engine
to identify where the 289 GFLOPS gap comes from (848 standalone vs 559 integrated)
"""

import sys
import os
import numpy as np
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    print("⚠️  PyOpenCL not available")
    sys.exit(1)


@contextmanager
def timer(name):
    """Context manager for timing operations"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"  ⏱️  {name}: {elapsed_ms:.2f} ms")


def test_standalone_performance(size=2048):
    """Test standalone kernel performance (minimal overhead)"""
    print(f"\n{'='*70}")
    print(f"🔬 STANDALONE PERFORMANCE TEST ({size}×{size})")
    print(f"{'='*70}\n")
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    print(f"Device: {device.name}")
    
    # Load kernel
    kernel_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'opencl', 'kernels', 
        'gemm_float4_clover.cl'
    )
    
    with timer("Kernel compilation"):
        with open(kernel_path, 'r') as f:
            source = f.read()
        program = cl.Program(context, source).build(options="-cl-fast-relaxed-math")
    
    # Create matrices
    M = N = K = size
    N_aligned = ((N + 3) // 4) * 4
    
    with timer("Matrix allocation"):
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N_aligned).astype(np.float32)
        C = np.zeros((M, N_aligned), dtype=np.float32)
    
    # Create buffers (minimal overhead)
    mf = cl.mem_flags
    with timer("Buffer creation"):
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(context, mf.WRITE_ONLY, size=C.nbytes)
    
    # Setup kernel
    kernel = program.gemm_float4_vec
    local_size = (16, 16)
    global_size = (
        ((M + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((N_aligned // 4 + local_size[1] - 1) // local_size[1]) * local_size[1]
    )
    
    kernel.set_args(
        np.int32(M), np.int32(N_aligned), np.int32(K),
        np.float32(1.0),
        A_buf, B_buf,
        np.float32(0.0),
        C_buf
    )
    
    # Warmup
    with timer("Warmup (5 iterations)"):
        for _ in range(5):
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
    
    # Benchmark
    iterations = 20
    with timer(f"Benchmark ({iterations} iterations)"):
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
            queue.finish()
            end = time.perf_counter()
            times.append(end - start)
    
    # Calculate GFLOPS
    times = np.array(times)
    avg_time = np.mean(times)
    min_time = np.min(times)
    ops = 2 * M * N_aligned * K
    
    gflops_avg = (ops / avg_time) / 1e9
    gflops_peak = (ops / min_time) / 1e9
    
    print(f"\n📊 Results:")
    print(f"  Average: {gflops_avg:.2f} GFLOPS")
    print(f"  Peak:    {gflops_peak:.2f} GFLOPS")
    print(f"  Time:    {avg_time*1000:.2f} ms (avg), {min_time*1000:.2f} ms (min)")
    
    return gflops_peak


def test_integrated_performance(size=2048):
    """Test integrated engine performance"""
    print(f"\n{'='*70}")
    print(f"🏭 INTEGRATED ENGINE TEST ({size}×{size})")
    print(f"{'='*70}\n")
    
    from optimization_engines.optimized_kernel_engine import OptimizedKernelEngine
    
    with timer("Engine initialization"):
        engine = OptimizedKernelEngine()
    
    # Create matrices
    M = N = K = size
    N_aligned = ((N + 3) // 4) * 4
    
    with timer("Matrix allocation"):
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N_aligned).astype(np.float32)
    
    # First call (includes buffer allocation)
    print("\n🔄 First call (cold):")
    with timer("  Total time"):
        result1 = engine.gemm(A, B)
    
    print(f"  GFLOPS: {result1.kernel_metrics.gflops:.2f}")
    print(f"  Kernel: {result1.kernel_metrics.kernel_name}")
    
    # Second call (buffers cached)
    print("\n🔥 Second call (warm):")
    with timer("  Total time"):
        result2 = engine.gemm(A, B)
    
    print(f"  GFLOPS: {result2.kernel_metrics.gflops:.2f}")
    
    # Multiple calls to get average
    print("\n📈 Average over 10 calls:")
    gflops_list = []
    with timer("  Total time"):
        for i in range(10):
            result = engine.gemm(A, B)
            gflops_list.append(result.kernel_metrics.gflops)
    
    avg_gflops = np.mean(gflops_list)
    max_gflops = np.max(gflops_list)
    min_gflops = np.min(gflops_list)
    
    print(f"  Average: {avg_gflops:.2f} GFLOPS")
    print(f"  Peak:    {max_gflops:.2f} GFLOPS")
    print(f"  Min:     {min_gflops:.2f} GFLOPS")
    print(f"  Std dev: {np.std(gflops_list):.2f} GFLOPS")
    
    return max_gflops


def analyze_overhead_components(size=2048):
    """Detailed breakdown of overhead components"""
    print(f"\n{'='*70}")
    print(f"🔍 OVERHEAD BREAKDOWN ANALYSIS")
    print(f"{'='*70}\n")
    
    from optimization_engines.optimized_kernel_engine import OptimizedKernelEngine
    
    engine = OptimizedKernelEngine()
    
    M = N = K = size
    N_aligned = ((N + 3) // 4) * 4
    
    # Create matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N_aligned).astype(np.float32)
    
    # Analyze engine.gemm() call breakdown
    print("Timing individual components:\n")
    
    # Component 1: Matrix validation and preparation
    with timer("1. Matrix validation & prep"):
        A_test = np.ascontiguousarray(A, dtype=np.float32)
        B_test = np.ascontiguousarray(B, dtype=np.float32)
        C_test = np.zeros((M, N_aligned), dtype=np.float32)
    
    # Component 2: Kernel selection
    with timer("2. Kernel selection"):
        from optimization_engines.optimized_kernel_engine import KernelType
        kernel_type = engine.select_best_kernel(M, N_aligned, K, None)
    
    print(f"  Selected: {kernel_type}")
    
    # Component 3: Buffer allocation (first time)
    with timer("3. Buffer allocation (first)"):
        result1 = engine.gemm(A, B)
    
    # Component 4: Buffer reuse (second time)
    with timer("4. Buffer reuse (cached)"):
        result2 = engine.gemm(A, B)
    
    print(f"\n📊 Performance comparison:")
    print(f"  First call:  {result1.kernel_metrics.gflops:.2f} GFLOPS")
    print(f"  Second call: {result2.kernel_metrics.gflops:.2f} GFLOPS")
    print(f"  Improvement: {(result2.kernel_metrics.gflops / result1.kernel_metrics.gflops - 1)*100:.1f}%")
    
    # Check transfer metrics
    print(f"\n📥 Transfer metrics (first call):")
    print(f"  H2D time: {result1.transfer_metrics.h2d_time_ms:.2f} ms")
    print(f"  D2H time: {result1.transfer_metrics.d2h_time_ms:.2f} ms")
    total_transfer1 = result1.transfer_metrics.h2d_time_ms + result1.transfer_metrics.d2h_time_ms
    print(f"  Total transfer: {total_transfer1:.2f} ms")
    
    print(f"\n📥 Transfer metrics (second call):")
    print(f"  H2D time: {result2.transfer_metrics.h2d_time_ms:.2f} ms")
    print(f"  D2H time: {result2.transfer_metrics.d2h_time_ms:.2f} ms")
    total_transfer2 = result2.transfer_metrics.h2d_time_ms + result2.transfer_metrics.d2h_time_ms
    print(f"  Total transfer: {total_transfer2:.2f} ms")
    
    # Kernel execution time
    print(f"\n⚡ Kernel execution:")
    print(f"  First:  {result1.kernel_metrics.exec_time_ms:.2f} ms")
    print(f"  Second: {result2.kernel_metrics.exec_time_ms:.2f} ms")
    
    # Total time breakdown
    print(f"\n📊 Time breakdown (first call):")
    kernel_pct = (result1.kernel_metrics.exec_time_ms / result1.total_time_ms) * 100
    transfer_pct = (total_transfer1 / result1.total_time_ms) * 100
    overhead_pct = 100 - kernel_pct - transfer_pct
    
    print(f"  Kernel:   {result1.kernel_metrics.exec_time_ms:.2f} ms ({kernel_pct:.1f}%)")
    print(f"  Transfer: {total_transfer1:.2f} ms ({transfer_pct:.1f}%)")
    print(f"  Overhead: {result1.total_time_ms - result1.kernel_metrics.exec_time_ms - total_transfer1:.2f} ms ({overhead_pct:.1f}%)")
    print(f"  Total:    {result1.total_time_ms:.2f} ms")


def main():
    """Main profiling routine"""
    print("="*70)
    print("🔬 ENGINE OVERHEAD PROFILING")
    print("="*70)
    print("\nGoal: Identify why integrated (559 GFLOPS) < standalone (848 GFLOPS)")
    print("Gap: 289 GFLOPS (34% performance loss)")
    
    # Test size
    size = 2048
    
    # Test standalone
    standalone_gflops = test_standalone_performance(size)
    
    # Test integrated
    integrated_gflops = test_integrated_performance(size)
    
    # Detailed overhead analysis
    analyze_overhead_components(size)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Standalone:  {standalone_gflops:.2f} GFLOPS 🏆")
    print(f"Integrated:  {integrated_gflops:.2f} GFLOPS")
    print(f"Gap:         {standalone_gflops - integrated_gflops:.2f} GFLOPS")
    print(f"Loss:        {(1 - integrated_gflops/standalone_gflops)*100:.1f}%")
    
    print(f"\n💡 Optimization opportunities:")
    if standalone_gflops - integrated_gflops > 200:
        print(f"  🔴 CRITICAL: >200 GFLOPS gap indicates major overhead")
        print(f"     - Check buffer pool efficiency")
        print(f"     - Reduce memory manager overhead")
        print(f"     - Optimize transfer patterns")
    elif standalone_gflops - integrated_gflops > 100:
        print(f"  🟡 MODERATE: 100-200 GFLOPS gap")
        print(f"     - Fine-tune buffer management")
        print(f"     - Reduce validation overhead")
    else:
        print(f"  🟢 MINOR: <100 GFLOPS gap - acceptable overhead")
    
    print(f"\n🎯 Target: Reduce gap to <100 GFLOPS")
    print(f"   → This would give us {integrated_gflops + (standalone_gflops - integrated_gflops - 100):.0f}+ GFLOPS integrated")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
