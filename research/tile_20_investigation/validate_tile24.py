#!/usr/bin/env python3
"""
Tile=24 Validation & Benchmark - Phase 2.1 Step 2

Professional validation of tile24_vectorized kernel:
1. Correctness validation across multiple sizes
2. Performance benchmark vs tile20
3. Stability testing (multiple iterations)
4. Comparison analysis
"""

import numpy as np
import pyopencl as cl
import time
import json
from pathlib import Path
from typing import Dict, List

def load_kernel(kernel_name: str) -> str:
    """Load kernel source code"""
    kernel_path = Path(__file__).parent / "kernels" / f"{kernel_name}.cl"
    
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    
    with open(kernel_path, 'r') as f:
        return f.read()


def benchmark_kernel(
    ctx, queue, kernel_code: str, kernel_name: str,
    M: int, N: int, K: int, 
    tile_size: int, local_size: tuple,
    warmup: int = 3, iterations: int = 10
) -> Dict:
    """
    Benchmark GEMM kernel with comprehensive metrics
    
    Returns:
        dict with gflops, time stats, correctness error
    """
    # Compile kernel
    try:
        prg = cl.Program(ctx, kernel_code).build()
    except Exception as e:
        return {'error': f'Compilation failed: {e}', 'gflops_avg': 0.0}
    
    # Generate test matrices
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
    
    # Global and local sizes
    global_size = (
        ((N + tile_size - 1) // tile_size) * local_size[0],
        ((M + tile_size - 1) // tile_size) * local_size[1]
    )
    
    # Get kernel function
    kernel_func = getattr(prg, kernel_name)
    
    # Alpha and beta
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    
    # Warmup
    for _ in range(warmup):
        kernel_func(
            queue, global_size, local_size,
            np.int32(M), np.int32(N), np.int32(K),
            alpha, a_buf, b_buf, beta, c_buf
        )
    queue.finish()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        kernel_func(
            queue, global_size, local_size,
            np.int32(M), np.int32(N), np.int32(K),
            alpha, a_buf, b_buf, beta, c_buf
        )
        queue.finish()
        end = time.perf_counter()
        times.append(end - start)
    
    # Read result
    cl.enqueue_copy(queue, C, c_buf)
    
    # Correctness check
    C_ref = A @ B
    max_error = np.max(np.abs(C - C_ref))
    mean_error = np.mean(np.abs(C - C_ref))
    
    # Statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    gflops_avg = (flops / avg_time) / 1e9
    gflops_best = (flops / min_time) / 1e9
    gflops_worst = (flops / max_time) / 1e9
    
    return {
        'gflops_avg': float(gflops_avg),
        'gflops_best': float(gflops_best),
        'gflops_worst': float(gflops_worst),
        'time_avg_ms': float(avg_time * 1000),
        'time_std_ms': float(std_time * 1000),
        'time_min_ms': float(min_time * 1000),
        'time_max_ms': float(max_time * 1000),
        'max_error': float(max_error),
        'mean_error': float(mean_error),
        'iterations': iterations,
        'correctness': 'PASS' if max_error < 0.1 else 'FAIL'
    }


def main():
    print("=" * 80)
    print("TILE=24 VECTORIZED - PROFESSIONAL VALIDATION & BENCHMARK")
    print("=" * 80)
    print()
    print("Phase 2.1 Step 2: Validate and benchmark tile24 implementation")
    print("Expected: 850-900 GFLOPS @ 1400×1400")
    print()
    
    # Initialize OpenCL
    platforms = cl.get_platforms()
    
    print(f"Available platforms: {len(platforms)}")
    for i, p in enumerate(platforms):
        print(f"  [{i}] {p.name}")
    print()
    
    # Find platform
    platform = None
    for p in platforms:
        if 'amd' in p.name.lower() or 'mesa' in p.name.lower():
            platform = p
            break
    
    if platform is None and platforms:
        platform = platforms[0]
        print(f"⚠️ AMD platform not found, using: {platform.name}")
    
    if platform is None:
        print("❌ No OpenCL platform found!")
        return
    
    devices = platform.get_devices()
    if not devices:
        print("❌ No devices found!")
        return
    
    device = devices[0]
    print(f"Device: {device.name}")
    print(f"Platform: {platform.name}")
    print(f"Max work group size: {device.max_work_group_size}")
    print()
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Load kernels
    print("Loading kernels...")
    tile20_code = load_kernel("approach_2_v3_vectorized")
    tile24_code = load_kernel("tile24_vectorized")
    print("✓ tile20_vectorized loaded")
    print("✓ tile24_vectorized loaded")
    print()
    
    # Test sizes
    test_sizes = [
        512,   # Small
        768,   # Medium-small
        1024,  # Medium
        1280,  # Previous best
        1400,  # NEW best (from step 1)
        1536,  # Medium-large
        2048,  # Large
        3072,  # Very large
    ]
    
    results = {
        'tile20': [],
        'tile24': []
    }
    
    print("=" * 80)
    print("CORRECTNESS VALIDATION")
    print("=" * 80)
    print()
    
    # First, validate correctness on smaller size
    print("Testing correctness on 512×512...")
    
    tile20_test = benchmark_kernel(
        ctx, queue, tile20_code, "gemm_tile20_vectorized",
        512, 512, 512, tile_size=20, local_size=(10, 10),
        warmup=1, iterations=1
    )
    
    tile24_test = benchmark_kernel(
        ctx, queue, tile24_code, "gemm_tile24_vectorized",
        512, 512, 512, tile_size=24, local_size=(12, 12),
        warmup=1, iterations=1
    )
    
    print(f"tile20: error={tile20_test['max_error']:.6f} - {tile20_test['correctness']}")
    print(f"tile24: error={tile24_test['max_error']:.6f} - {tile24_test['correctness']}")
    print()
    
    if tile24_test['correctness'] != 'PASS':
        print("❌ tile24 FAILED correctness test!")
        print(f"   Max error: {tile24_test['max_error']}")
        print("   ABORTING benchmarks")
        return
    
    print("✅ tile24 PASSED correctness validation")
    print()
    
    # Performance benchmarks
    print("=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print()
    
    for size in test_sizes:
        print(f"Benchmarking {size}×{size}...")
        print()
        
        # tile20
        print("  tile20 (10×10 threads, tile=20)...")
        r20 = benchmark_kernel(
            ctx, queue, tile20_code, "gemm_tile20_vectorized",
            size, size, size, tile_size=20, local_size=(10, 10),
            warmup=3, iterations=10
        )
        results['tile20'].append({'size': size, **r20})
        
        if 'error' not in r20:
            print(f"    {r20['gflops_avg']:.1f} GFLOPS "
                  f"(best: {r20['gflops_best']:.1f}) "
                  f"[{r20['time_avg_ms']:.2f}ms ± {r20['time_std_ms']:.2f}ms] "
                  f"{r20['correctness']}")
        else:
            print(f"    ERROR: {r20['error']}")
        
        # tile24
        print("  tile24 (12×12 threads, tile=24)...")
        r24 = benchmark_kernel(
            ctx, queue, tile24_code, "gemm_tile24_vectorized",
            size, size, size, tile_size=24, local_size=(12, 12),
            warmup=3, iterations=10
        )
        results['tile24'].append({'size': size, **r24})
        
        if 'error' not in r24:
            print(f"    {r24['gflops_avg']:.1f} GFLOPS "
                  f"(best: {r24['gflops_best']:.1f}) "
                  f"[{r24['time_avg_ms']:.2f}ms ± {r24['time_std_ms']:.2f}ms] "
                  f"{r24['correctness']}")
        else:
            print(f"    ERROR: {r24['error']}")
        
        # Comparison
        if 'error' not in r20 and 'error' not in r24:
            delta = r24['gflops_avg'] - r20['gflops_avg']
            pct = (delta / r20['gflops_avg']) * 100
            symbol = "🏆" if delta > 0 else "⚠️"
            print(f"  {symbol} tile24 vs tile20: {delta:+.1f} GFLOPS ({pct:+.1f}%)")
        
        print()
    
    # Analysis
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Summary table
    print("PERFORMANCE COMPARISON TABLE:")
    print()
    print("Size   | tile20    | tile24    | Delta     | Winner")
    print("-------|-----------|-----------|-----------|--------")
    
    for i, size in enumerate(test_sizes):
        r20 = results['tile20'][i]
        r24 = results['tile24'][i]
        
        if 'error' not in r20 and 'error' not in r24:
            delta = r24['gflops_avg'] - r20['gflops_avg']
            pct = (delta / r20['gflops_avg']) * 100
            winner = "tile24 🏆" if delta > 0 else "tile20"
            
            print(f"{size:4d}   | {r20['gflops_avg']:7.1f}   | {r24['gflops_avg']:7.1f}   | "
                  f"{delta:+6.1f} ({pct:+5.1f}%) | {winner}")
        else:
            print(f"{size:4d}   | ERROR     | ERROR     | -         | -")
    
    print()
    
    # Statistics
    tile20_valid = [r for r in results['tile20'] if 'error' not in r]
    tile24_valid = [r for r in results['tile24'] if 'error' not in r]
    
    if tile20_valid and tile24_valid:
        tile20_avg = np.mean([r['gflops_avg'] for r in tile20_valid])
        tile24_avg = np.mean([r['gflops_avg'] for r in tile24_valid])
        
        tile20_max = max([r['gflops_avg'] for r in tile20_valid])
        tile24_max = max([r['gflops_avg'] for r in tile24_valid])
        
        tile20_best_size = max(tile20_valid, key=lambda x: x['gflops_avg'])
        tile24_best_size = max(tile24_valid, key=lambda x: x['gflops_avg'])
        
        print("SUMMARY STATISTICS:")
        print()
        print(f"tile20:")
        print(f"  Average: {tile20_avg:.1f} GFLOPS")
        print(f"  Peak: {tile20_max:.1f} GFLOPS @ {tile20_best_size['size']}×{tile20_best_size['size']}")
        print()
        print(f"tile24:")
        print(f"  Average: {tile24_avg:.1f} GFLOPS")
        print(f"  Peak: {tile24_max:.1f} GFLOPS @ {tile24_best_size['size']}×{tile24_best_size['size']}")
        print()
        
        overall_delta = tile24_avg - tile20_avg
        overall_pct = (overall_delta / tile20_avg) * 100
        
        print(f"Overall Improvement: {overall_delta:+.1f} GFLOPS ({overall_pct:+.1f}%)")
        print()
    
    # Save results
    output_file = Path(__file__).parent / "tile24_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device.name,
            'test_sizes': test_sizes,
            'results': results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Final recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    
    if tile24_valid:
        best_tile24 = max(tile24_valid, key=lambda x: x['gflops_avg'])
        
        if best_tile24['gflops_avg'] >= 850:
            print(f"✅ SUCCESS: tile24 achieved {best_tile24['gflops_avg']:.1f} GFLOPS @ {best_tile24['size']}×{best_tile24['size']}")
            print("   Phase 2 target (850 GFLOPS) ACHIEVED!")
            print()
            print("   Next steps:")
            print("   1. Integrate tile24 into adaptive_kernel_selector")
            print("   2. Retrain ML model with tile24 data")
            print("   3. Deploy to production")
        else:
            print(f"⚠️ tile24 peak: {best_tile24['gflops_avg']:.1f} GFLOPS @ {best_tile24['size']}×{best_tile24['size']}")
            print(f"   Gap to 850 target: {850 - best_tile24['gflops_avg']:.1f} GFLOPS")
            print()
            print("   Options:")
            print("   1. Use tile20 @ 1400 (819.7 GFLOPS)")
            print("   2. Continue to Phase 2.2 (FP16 mixed precision)")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
