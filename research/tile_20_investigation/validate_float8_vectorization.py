#!/usr/bin/env python3
"""
Validate float8 Vectorization Performance
Phase 2.1 Extension - Float8 Experiment

Compare float8 vs float4 vectorization:
- tile20_float8.cl (NEW)
- tile20_optimized.cl (current best @ float4)

Target: >= 950 GFLOPS @ 1400 to justify integration
If < 900 GFLOPS: discard (register pressure or bandwidth not bottleneck)
"""

import numpy as np
import pyopencl as cl
import time
from pathlib import Path
import json

def load_kernel(kernel_name: str) -> str:
    """Load OpenCL kernel source"""
    kernel_path = Path(__file__).parent / "kernels" / kernel_name
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    return kernel_path.read_text()

def benchmark_kernel(
    ctx, queue, kernel_source: str, kernel_name: str,
    M: int, N: int, K: int,
    local_size: tuple,
    iterations: int = 10,
    warmup: int = 2
) -> dict:
    """
    Benchmark a GEMM kernel
    
    Returns:
        dict with performance metrics
    """
    # Create program
    try:
        prg = cl.Program(ctx, kernel_source).build(options=["-cl-fast-relaxed-math"])
    except Exception as e:
        print(f"❌ Compilation failed for {kernel_name}")
        print(f"Error: {e}")
        return None
    
    # Prepare data
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
    
    # Get kernel
    kernel = getattr(prg, kernel_name)
    
    # Set kernel arguments
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    kernel.set_args(
        np.int32(M), np.int32(N), np.int32(K),
        alpha, a_buf, b_buf, beta, c_buf
    )
    
    # Calculate global size
    tile_size = 20
    global_size = (
        ((M + tile_size - 1) // tile_size) * local_size[0],
        ((N + tile_size - 1) // tile_size) * local_size[1]
    )
    
    print(f"  Global size: {global_size}, Local size: {local_size}")
    
    # Warmup
    for _ in range(warmup):
        try:
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
            queue.finish()
        except Exception as e:
            print(f"❌ Kernel execution failed during warmup")
            print(f"Error: {e}")
            return None
    
    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
        end = time.perf_counter()
        times.append(end - start)
    
    # Read result for correctness check
    C_gpu = np.empty_like(C)
    cl.enqueue_copy(queue, C_gpu, c_buf).wait()
    
    # Compute reference (for correctness)
    C_ref = A @ B
    max_error = np.max(np.abs(C_gpu - C_ref))
    mean_error = np.mean(np.abs(C_gpu - C_ref))
    
    # Calculate GFLOPS
    ops = 2 * M * N * K
    avg_time = np.mean(times)
    min_time = np.min(times)
    gflops_avg = (ops / avg_time) / 1e9
    gflops_peak = (ops / min_time) / 1e9
    
    return {
        'M': M, 'N': N, 'K': K,
        'kernel_name': kernel_name,
        'local_size': local_size,
        'global_size': global_size,
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'gflops_avg': gflops_avg,
        'gflops_peak': gflops_peak,
        'max_error': max_error,
        'mean_error': mean_error,
        'iterations': iterations
    }

def compare_kernels(ctx, queue, size: int):
    """Compare float4 vs float8 at given size"""
    
    M = N = K = size
    print(f"\n{'='*70}")
    print(f"SIZE: {size}×{size}")
    print('='*70)
    
    results = {}
    
    # 1. Benchmark float4 (current best)
    print("\n🔵 Testing float4 (tile20_optimized.cl)...")
    try:
        kernel_src = load_kernel("tile20_optimized.cl")
        result = benchmark_kernel(
            ctx, queue, kernel_src, "gemm_tile20_optimized",
            M, N, K, local_size=(10, 10), iterations=10
        )
        if result:
            print(f"  ✅ GFLOPS: {result['gflops_peak']:.1f} (peak), {result['gflops_avg']:.1f} (avg)")
            print(f"  ✅ Correctness: max_error={result['max_error']:.6f}")
            results['float4'] = result
        else:
            print(f"  ❌ Benchmark failed")
            results['float4'] = None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results['float4'] = None
    
    # 2. Benchmark float8 (new experiment)
    print("\n🟣 Testing float8 (tile20_float8.cl)...")
    try:
        kernel_src = load_kernel("tile20_float8.cl")
        result = benchmark_kernel(
            ctx, queue, kernel_src, "gemm_tile20_float8",
            M, N, K, local_size=(10, 10), iterations=10
        )
        if result:
            print(f"  ✅ GFLOPS: {result['gflops_peak']:.1f} (peak), {result['gflops_avg']:.1f} (avg)")
            print(f"  ✅ Correctness: max_error={result['max_error']:.6f}")
            results['float8'] = result
        else:
            print(f"  ❌ Benchmark failed")
            results['float8'] = None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results['float8'] = None
    
    # 3. Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print('='*70)
    
    if results['float4'] and results['float8']:
        f4_gflops = results['float4']['gflops_peak']
        f8_gflops = results['float8']['gflops_peak']
        delta = f8_gflops - f4_gflops
        delta_pct = (delta / f4_gflops) * 100
        
        print(f"float4: {f4_gflops:.1f} GFLOPS")
        print(f"float8: {f8_gflops:.1f} GFLOPS")
        print(f"Delta:  {delta:+.1f} GFLOPS ({delta_pct:+.1f}%)")
        
        if f8_gflops >= 950:
            print(f"\n✅ SUCCESS: float8 >= 950 GFLOPS → INTEGRATE!")
        elif f8_gflops > f4_gflops:
            print(f"\n⚠️  MARGINAL: float8 better but < 950 target")
        else:
            print(f"\n❌ FAILED: float8 worse than float4 → DISCARD")
        
        results['delta_gflops'] = delta
        results['delta_percent'] = delta_pct
        results['decision'] = 'integrate' if f8_gflops >= 950 else ('marginal' if f8_gflops > f4_gflops else 'discard')
    else:
        print("⚠️  Cannot compare - one or both kernels failed")
        results['decision'] = 'error'
    
    return results

def main():
    print("="*70)
    print("FLOAT8 VECTORIZATION VALIDATION")
    print("="*70)
    print()
    print("Goal: Test if float8 can reach 950+ GFLOPS")
    print("Current best: 866.9 GFLOPS with float4")
    print("Target improvement: +10-15% (950-1000 GFLOPS)")
    print()
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    platform = platforms[0]  # Clover
    devices = platform.get_devices()
    device = devices[0]
    
    print(f"Platform: {platform.name}")
    print(f"Device: {device.name}")
    print(f"Preferred vector width (float): {device.preferred_vector_width_float}")
    print(f"Native vector width (float): {device.native_vector_width_float}")
    print()
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Test key sizes
    test_sizes = [1400, 1280, 2048]  # Sweet spot, Phase2 best, large matrix
    
    all_results = {}
    
    for size in test_sizes:
        results = compare_kernels(ctx, queue, size)
        all_results[size] = results
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()
    
    print(f"{'Size':<10} {'float4':<12} {'float8':<12} {'Delta':<15} {'Decision':<10}")
    print("-"*70)
    
    for size in test_sizes:
        res = all_results[size]
        if 'float4' in res and 'float8' in res and res['float4'] and res['float8']:
            f4 = res['float4']['gflops_peak']
            f8 = res['float8']['gflops_peak']
            delta = res['delta_percent']
            decision = res['decision']
            print(f"{size:<10} {f4:<12.1f} {f8:<12.1f} {delta:+6.1f}%        {decision:<10}")
        else:
            print(f"{size:<10} {'ERROR':<12} {'ERROR':<12} {'N/A':<15} {'error':<10}")
    
    print()
    
    # Final decision
    best_size = 1400
    if best_size in all_results and 'decision' in all_results[best_size]:
        decision = all_results[best_size]['decision']
        if decision == 'integrate':
            print("🎉 RECOMMENDATION: INTEGRATE float8 kernel")
            print("   - Achieved >= 950 GFLOPS target")
            print("   - Replace float4 as primary kernel")
        elif decision == 'marginal':
            print("⚠️  RECOMMENDATION: MARGINAL - Manual review needed")
            print("   - float8 slightly better, but < 950 target")
            print("   - Consider ROI: is small gain worth complexity?")
        elif decision == 'discard':
            print("❌ RECOMMENDATION: DISCARD float8 kernel")
            print("   - No improvement over float4")
            print("   - Stick with 866.9 GFLOPS (float4)")
            print("   - Reason: likely register pressure or bandwidth not bottleneck")
        else:
            print("❌ RECOMMENDATION: ERROR - Review logs")
    
    # Save results
    output_file = Path(__file__).parent / "float8_validation_results.json"
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable
        json_results = {}
        for size, res in all_results.items():
            json_results[str(size)] = {
                'float4': res.get('float4'),
                'float8': res.get('float8'),
                'delta_gflops': res.get('delta_gflops'),
                'delta_percent': res.get('delta_percent'),
                'decision': res.get('decision')
            }
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
