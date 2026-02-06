#!/usr/bin/env python3
"""
FP16 Mixed Precision Validation & Benchmark - Phase 2.2

Professional validation framework:
1. Check FP16 support on device
2. Precision analysis (multiple error metrics)
3. Performance benchmark vs FP32
4. Use case recommendations
"""

import numpy as np
import pyopencl as cl
import time
import json
from pathlib import Path
from typing import Dict, Tuple

def check_fp16_support(device) -> bool:
    """Check if device supports cl_khr_fp16"""
    extensions = device.extensions.split()
    return 'cl_khr_fp16' in extensions


def load_kernel(kernel_name: str) -> str:
    """Load kernel source code"""
    kernel_path = Path(__file__).parent / "kernels" / f"{kernel_name}.cl"
    
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    
    with open(kernel_path, 'r') as f:
        return f.read()


def precision_analysis(C_fp16: np.ndarray, C_fp32: np.ndarray) -> Dict:
    """
    Comprehensive precision analysis
    
    Returns multiple error metrics to assess precision loss
    """
    # Absolute errors
    abs_error = np.abs(C_fp16 - C_fp32)
    max_abs_error = np.max(abs_error)
    mean_abs_error = np.mean(abs_error)
    
    # Relative errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = abs_error / (np.abs(C_fp32) + 1e-10)
    rel_error = np.where(np.isfinite(rel_error), rel_error, 0)
    
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)
    
    # Percentile errors
    p50_abs = np.percentile(abs_error, 50)
    p95_abs = np.percentile(abs_error, 95)
    p99_abs = np.percentile(abs_error, 99)
    
    p50_rel = np.percentile(rel_error, 50)
    p95_rel = np.percentile(rel_error, 95)
    p99_rel = np.percentile(rel_error, 99)
    
    # Frobenius norm
    frob_norm_diff = np.linalg.norm(C_fp16 - C_fp32, 'fro')
    frob_norm_ref = np.linalg.norm(C_fp32, 'fro')
    frob_rel_error = frob_norm_diff / (frob_norm_ref + 1e-10)
    
    return {
        'max_abs_error': float(max_abs_error),
        'mean_abs_error': float(mean_abs_error),
        'max_rel_error': float(max_rel_error),
        'mean_rel_error': float(mean_rel_error),
        'p50_abs_error': float(p50_abs),
        'p95_abs_error': float(p95_abs),
        'p99_abs_error': float(p99_abs),
        'p50_rel_error': float(p50_rel),
        'p95_rel_error': float(p95_rel),
        'p99_rel_error': float(p99_rel),
        'frobenius_rel_error': float(frob_rel_error)
    }


def benchmark_kernel(
    ctx, queue, kernel_code: str, kernel_name: str,
    M: int, N: int, K: int,
    warmup: int = 3, iterations: int = 10
) -> Tuple[Dict, np.ndarray]:
    """
    Benchmark GEMM kernel and return result matrix
    
    Returns:
        (performance_metrics, result_matrix)
    """
    # Compile kernel
    try:
        prg = cl.Program(ctx, kernel_code).build()
    except Exception as e:
        return {'error': f'Compilation failed: {e}'}, None
    
    # Generate test matrices (SAME seed for fair comparison)
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32) * 10  # Scale up for better FP16 range
    B = np.random.randn(K, N).astype(np.float32) * 10
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
    
    # Global and local sizes
    local_size = (10, 10)
    global_size = (
        ((N + 19) // 20) * local_size[0],
        ((M + 19) // 20) * local_size[1]
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
    
    # Statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    gflops_avg = (flops / avg_time) / 1e9
    gflops_best = (flops / min_time) / 1e9
    
    metrics = {
        'gflops_avg': float(gflops_avg),
        'gflops_best': float(gflops_best),
        'time_avg_ms': float(avg_time * 1000),
        'time_std_ms': float(std_time * 1000),
        'time_min_ms': float(min_time * 1000),
    }
    
    return metrics, C


def assess_precision_acceptability(precision_metrics: Dict) -> Dict:
    """
    Assess if precision loss is acceptable for various use cases
    
    Criteria:
    - Neural Networks: mean_rel_error < 1%, max_rel_error < 5%
    - Image Processing: mean_abs_error < 0.1, max_abs_error < 1.0
    - Graphics: p95_rel_error < 2%
    - Scientific: mean_rel_error < 0.01%, max_rel_error < 0.1%
    """
    mean_rel = precision_metrics['mean_rel_error']
    max_rel = precision_metrics['max_rel_error']
    mean_abs = precision_metrics['mean_abs_error']
    max_abs = precision_metrics['max_abs_error']
    p95_rel = precision_metrics['p95_rel_error']
    
    return {
        'neural_networks': {
            'acceptable': mean_rel < 0.01 and max_rel < 0.05,
            'reason': f"mean_rel={mean_rel:.4f} (req: <0.01), max_rel={max_rel:.4f} (req: <0.05)"
        },
        'image_processing': {
            'acceptable': mean_abs < 0.1 and max_abs < 1.0,
            'reason': f"mean_abs={mean_abs:.4f} (req: <0.1), max_abs={max_abs:.4f} (req: <1.0)"
        },
        'graphics_rendering': {
            'acceptable': p95_rel < 0.02,
            'reason': f"p95_rel={p95_rel:.4f} (req: <0.02)"
        },
        'scientific_computing': {
            'acceptable': mean_rel < 0.0001 and max_rel < 0.001,
            'reason': f"mean_rel={mean_rel:.6f} (req: <0.0001), max_rel={max_rel:.6f} (req: <0.001)"
        }
    }


def main():
    print("=" * 80)
    print("FP16 MIXED PRECISION - PROFESSIONAL VALIDATION & BENCHMARK")
    print("=" * 80)
    print()
    print("Phase 2.2 Moonshot: Achieve 1000+ GFLOPS via FP16")
    print("Target: tile20 FP32 866.9 GFLOPS → FP16 1200+ GFLOPS")
    print()
    
    # Initialize OpenCL
    platforms = cl.get_platforms()
    
    print(f"Available platforms: {len(platforms)}")
    for i, p in enumerate(platforms):
        print(f"  [{i}] {p.name}")
    print()
    
    # Find platform - Try rusticl first (better FP16 support), then others
    platform = None
    
    # Priority 1: rusticl (modern Mesa driver with better FP16 support)
    for p in platforms:
        if 'rusticl' in p.name.lower():
            platform = p
            print(f"✓ Using rusticl platform (best FP16 support)")
            break
    
    # Priority 2: AMD/Mesa
    if platform is None:
        for p in platforms:
            if 'amd' in p.name.lower() or 'mesa' in p.name.lower():
                platform = p
                break
    
    # Priority 3: Any available
    if platform is None and platforms:
        platform = platforms[0]
    
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
    print()
    
    # Check FP16 support
    print("Checking FP16 support...")
    if check_fp16_support(device):
        print("✅ FP16 (cl_khr_fp16) is SUPPORTED")
    else:
        print("❌ FP16 (cl_khr_fp16) is NOT supported")
        print("   Cannot proceed with FP16 benchmark")
        return
    
    print()
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Load kernels
    print("Loading kernels...")
    try:
        fp32_code = load_kernel("approach_2_v3_vectorized")
        fp16_code = load_kernel("tile20_fp16_mixed")
        print("✓ FP32 kernel loaded (tile20_vectorized)")
        print("✓ FP16 kernel loaded (tile20_fp16_mixed)")
    except Exception as e:
        print(f"❌ Error loading kernels: {e}")
        return
    
    print()
    
    # Test sizes
    test_sizes = [512, 1024, 1400, 2048]
    
    results = {
        'fp32': [],
        'fp16': [],
        'precision': []
    }
    
    print("=" * 80)
    print("PRECISION VALIDATION")
    print("=" * 80)
    print()
    
    for size in test_sizes:
        print(f"Testing {size}×{size}...")
        print()
        
        # FP32 benchmark
        print("  Benchmarking FP32...")
        metrics_fp32, C_fp32 = benchmark_kernel(
            ctx, queue, fp32_code, "gemm_tile20_vectorized",
            size, size, size, warmup=3, iterations=10
        )
        
        if 'error' in metrics_fp32:
            print(f"  ❌ FP32 ERROR: {metrics_fp32['error']}")
            continue
        
        print(f"    FP32: {metrics_fp32['gflops_avg']:.1f} GFLOPS "
              f"(best: {metrics_fp32['gflops_best']:.1f}) "
              f"[{metrics_fp32['time_avg_ms']:.2f}ms]")
        
        # FP16 benchmark
        print("  Benchmarking FP16 mixed precision...")
        metrics_fp16, C_fp16 = benchmark_kernel(
            ctx, queue, fp16_code, "gemm_tile20_fp16_mixed",
            size, size, size, warmup=3, iterations=10
        )
        
        if 'error' in metrics_fp16:
            print(f"  ❌ FP16 ERROR: {metrics_fp16['error']}")
            continue
        
        print(f"    FP16: {metrics_fp16['gflops_avg']:.1f} GFLOPS "
              f"(best: {metrics_fp16['gflops_best']:.1f}) "
              f"[{metrics_fp16['time_avg_ms']:.2f}ms]")
        
        # Speedup
        speedup = metrics_fp16['gflops_avg'] / metrics_fp32['gflops_avg']
        print(f"  🚀 Speedup: {speedup:.2f}× ({metrics_fp16['gflops_avg'] - metrics_fp32['gflops_avg']:+.1f} GFLOPS)")
        print()
        
        # Precision analysis
        print("  Analyzing precision...")
        precision = precision_analysis(C_fp16, C_fp32)
        
        print(f"    Max absolute error: {precision['max_abs_error']:.6f}")
        print(f"    Mean absolute error: {precision['mean_abs_error']:.6f}")
        print(f"    Max relative error: {precision['max_rel_error']:.4%}")
        print(f"    Mean relative error: {precision['mean_rel_error']:.4%}")
        print(f"    P95 relative error: {precision['p95_rel_error']:.4%}")
        print(f"    Frobenius relative error: {precision['frobenius_rel_error']:.4%}")
        print()
        
        # Save results
        results['fp32'].append({'size': size, **metrics_fp32})
        results['fp16'].append({'size': size, **metrics_fp16})
        results['precision'].append({'size': size, **precision})
    
    # Overall analysis
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Performance comparison table
    print("PERFORMANCE COMPARISON:")
    print()
    print("Size   | FP32      | FP16      | Speedup | Delta")
    print("-------|-----------|-----------|---------|----------")
    
    for i, size in enumerate(test_sizes):
        if i < len(results['fp32']) and i < len(results['fp16']):
            fp32 = results['fp32'][i]
            fp16 = results['fp16'][i]
            
            if 'gflops_avg' in fp32 and 'gflops_avg' in fp16:
                speedup = fp16['gflops_avg'] / fp32['gflops_avg']
                delta = fp16['gflops_avg'] - fp32['gflops_avg']
                symbol = "🏆" if speedup > 1.3 else "✅" if speedup > 1.0 else "⚠️"
                
                print(f"{size:4d}   | {fp32['gflops_avg']:7.1f}   | {fp16['gflops_avg']:7.1f}   | "
                      f"{speedup:5.2f}×  | {delta:+7.1f} {symbol}")
    
    print()
    
    # Precision summary
    print("PRECISION SUMMARY:")
    print()
    print("Size   | Mean Rel Err | Max Rel Err | P95 Rel Err | Frob Rel Err")
    print("-------|--------------|-------------|-------------|-------------")
    
    for i, size in enumerate(test_sizes):
        if i < len(results['precision']):
            prec = results['precision'][i]
            
            print(f"{size:4d}   | {prec['mean_rel_error']:11.4%} | {prec['max_rel_error']:10.4%} | "
                  f"{prec['p95_rel_error']:10.4%} | {prec['frobenius_rel_error']:11.4%}")
    
    print()
    
    # Use case assessment (using largest size as representative)
    if results['precision']:
        print("USE CASE ASSESSMENT (based on largest matrix):")
        print()
        
        largest_prec = results['precision'][-1]
        assessment = assess_precision_acceptability(largest_prec)
        
        for use_case, result in assessment.items():
            status = "✅ ACCEPTABLE" if result['acceptable'] else "❌ NOT ACCEPTABLE"
            print(f"{use_case.replace('_', ' ').title():20s}: {status}")
            print(f"  {result['reason']}")
            print()
    
    # Overall statistics
    if results['fp16']:
        fp16_avg = np.mean([r['gflops_avg'] for r in results['fp16'] if 'gflops_avg' in r])
        fp16_peak = max([r['gflops_avg'] for r in results['fp16'] if 'gflops_avg' in r])
        fp32_avg = np.mean([r['gflops_avg'] for r in results['fp32'] if 'gflops_avg' in r])
        fp32_peak = max([r['gflops_avg'] for r in results['fp32'] if 'gflops_avg' in r])
        
        overall_speedup = fp16_avg / fp32_avg
        
        print("OVERALL STATISTICS:")
        print()
        print(f"FP32 Average: {fp32_avg:.1f} GFLOPS")
        print(f"FP32 Peak: {fp32_peak:.1f} GFLOPS")
        print(f"FP16 Average: {fp16_avg:.1f} GFLOPS")
        print(f"FP16 Peak: {fp16_peak:.1f} GFLOPS")
        print(f"Overall Speedup: {overall_speedup:.2f}×")
        print()
    
    # Save results
    output_file = Path(__file__).parent / "fp16_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device.name,
            'fp16_supported': True,
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
    
    if results['fp16']:
        peak_fp16 = max([r.get('gflops_avg', 0) for r in results['fp16']])
        
        if peak_fp16 >= 1000:
            print(f"✅ SUCCESS: FP16 achieved {peak_fp16:.1f} GFLOPS!")
            print("   Moonshot target (1000+ GFLOPS) ACHIEVED!")
            print()
            print("   Recommended for:")
            print("   ✅ Neural Network Training/Inference")
            print("   ✅ Image Processing")
            print("   ✅ Graphics/Rendering")
            print()
            print("   Not recommended for:")
            print("   ❌ High-precision Scientific Computing")
            print("   ❌ Financial Calculations")
        else:
            print(f"⚠️ FP16 peak: {peak_fp16:.1f} GFLOPS")
            print(f"   Gap to 1000 target: {1000 - peak_fp16:.1f} GFLOPS")
            print()
            print("   FP16 may not provide sufficient speedup for this hardware")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
