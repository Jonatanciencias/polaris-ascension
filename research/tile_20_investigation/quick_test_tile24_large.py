#!/usr/bin/env python3
"""
Quick Benchmark: tile24 @ 4096
Test if tile24 performs well on very large matrices before considering tile32

Author: AMD RX 590 GEMM Optimization Project
Date: February 2026
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pyopencl as cl
except ImportError:
    print("❌ Error: PyOpenCL not installed")
    sys.exit(1)


def benchmark_tile24_large():
    """Benchmark tile24 on 4096×4096 to see if tile32 is needed"""
    
    print("=" * 70)
    print("QUICK BENCHMARK: tile24 @ 4096")
    print("=" * 70)
    print()
    print("Purpose: Test if tile24 handles 4096×4096 well")
    print("Decision: If 800+ GFLOPS → tile32 not needed")
    print("          If < 750 GFLOPS → tile32 might help")
    print()
    
    # Setup OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print(f"✓ Device: {device.name}")
    
    # Load tile24 kernel
    kernel_path = Path(__file__).parent.parent.parent / "src" / "kernels" / "gemm_tile24_production.cl"
    
    if not kernel_path.exists():
        print(f"❌ Error: Kernel not found: {kernel_path}")
        return 1
    
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    program = cl.Program(ctx, kernel_code).build()
    kernel = program.gemm_tile24_vectorized
    
    print("✓ Kernel: tile24 loaded")
    print()
    
    # Test sizes
    test_sizes = [3072, 4096, 5120]  # Compare 4096 with neighbors
    
    print("=" * 70)
    print("BENCHMARKS")
    print("=" * 70)
    print()
    
    results = {}
    
    for size in test_sizes:
        print(f"Testing {size}×{size}...", end=" ", flush=True)
        
        # Generate matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)
        
        # GPU buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        
        # Workgroup config for tile24
        local_size = (12, 12)
        tile_size = 24
        global_size = (
            ((size + tile_size - 1) // tile_size) * local_size[0],
            ((size + tile_size - 1) // tile_size) * local_size[1]
        )
        
        # Warmup
        for _ in range(3):
            kernel(queue, global_size, local_size,
                   np.int32(size), np.int32(size), np.int32(size),
                   np.float32(1.0), a_buf, b_buf, np.float32(0.0), c_buf)
        queue.finish()
        
        # Benchmark
        times = []
        for _ in range(10):
            event = kernel(queue, global_size, local_size,
                          np.int32(size), np.int32(size), np.int32(size),
                          np.float32(1.0), a_buf, b_buf, np.float32(0.0), c_buf)
            event.wait()
            elapsed = (event.profile.end - event.profile.start) * 1e-9
            times.append(elapsed)
        
        # Get result
        cl.enqueue_copy(queue, C, c_buf).wait()
        
        # Check correctness
        C_ref = A @ B
        max_error = np.max(np.abs(C - C_ref))
        
        # Calculate performance
        avg_time = np.mean(times)
        gflops = (2.0 * size * size * size) / (avg_time * 1e9)
        
        results[size] = {
            'gflops': gflops,
            'time_ms': avg_time * 1000,
            'error': max_error,
            'correct': max_error < 0.001
        }
        
        status = "✅" if max_error < 0.001 else "❌"
        print(f"{status} {gflops:.1f} GFLOPS (error: {max_error:.6f})")
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze 4096 performance
    perf_3072 = results[3072]['gflops']
    perf_4096 = results[4096]['gflops']
    perf_5120 = results[5120]['gflops']
    
    print(f"tile24 Performance:")
    print(f"  3072: {perf_3072:.1f} GFLOPS")
    print(f"  4096: {perf_4096:.1f} GFLOPS")
    print(f"  5120: {perf_5120:.1f} GFLOPS")
    print()
    
    # Check alignment
    tiles_3072 = 3072 / 24  # 128 tiles (perfect)
    tiles_4096 = 4096 / 24  # 170.67 tiles (requires padding)
    tiles_5120 = 5120 / 24  # 213.33 tiles (requires padding)
    
    print(f"Tile Alignment:")
    print(f"  3072 / 24 = {tiles_3072:.1f} {'✅ Perfect' if tiles_3072 == int(tiles_3072) else '⚠️ Padding'}")
    print(f"  4096 / 24 = {tiles_4096:.1f} {'✅ Perfect' if tiles_4096 == int(tiles_4096) else '⚠️ Padding required'}")
    print(f"  5120 / 24 = {tiles_5120:.1f} {'✅ Perfect' if tiles_5120 == int(tiles_5120) else '⚠️ Padding'}")
    print()
    
    # tile32 alignment check
    tiles_4096_t32 = 4096 / 32  # 128 (perfect!)
    print(f"tile32 would have:")
    print(f"  4096 / 32 = {tiles_4096_t32:.0f} ✅ PERFECT alignment")
    print()
    
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()
    
    if perf_4096 >= 800:
        print("✅ SKIP tile32")
        print(f"   tile24 @ 4096: {perf_4096:.1f} GFLOPS (EXCELLENT)")
        print("   No need for tile32 - current performance is great")
        print()
        print("   Reason: Even with imperfect alignment (170.67 tiles),")
        print("   tile24 achieves 800+ GFLOPS. tile32's perfect alignment")
        print("   won't add much value, and register spilling risk is high.")
        
    elif perf_4096 >= 750:
        print("⚠️ MARGINAL CASE")
        print(f"   tile24 @ 4096: {perf_4096:.1f} GFLOPS (GOOD)")
        print("   tile32 might add 20-40 GFLOPS, but risky")
        print()
        print("   Your call:")
        print("   - If you need 4096+ frequently: Try tile32")
        print("   - If 4096 is rare: Skip tile32, not worth risk")
        
    else:
        print("🎯 WORTH TRYING tile32")
        print(f"   tile24 @ 4096: {perf_4096:.1f} GFLOPS (ROOM FOR IMPROVEMENT)")
        print("   tile32's perfect alignment could help significantly")
        print()
        print("   Expected improvement: +50-100 GFLOPS")
        print("   Risk: Register spilling (-50%)")
        print("   Time: 3-4 hours")
        print()
        print("   Proceed with: research/tile_20_investigation/implement_tile32.py")
    
    print()
    
    # Performance trend
    drop_3072_4096 = perf_3072 - perf_4096
    drop_pct = (drop_3072_4096 / perf_3072) * 100
    
    if drop_pct > 5:
        print(f"⚠️ Notable performance drop @ 4096: {drop_pct:.1f}%")
        print("   This suggests alignment/cache issues")
        print("   tile32 might address this")
    elif drop_pct > 0:
        print(f"✓ Small performance drop @ 4096: {drop_pct:.1f}%")
        print("   Within normal variation")
    else:
        print(f"✓ Performance stable or improving @ 4096")
        print("   tile24 handles large matrices well")
    
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(benchmark_tile24_large())
