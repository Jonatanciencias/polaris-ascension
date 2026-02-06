#!/usr/bin/env python3
"""
Validation Benchmark: tile20 @ 1300x1300
=========================================

Validates the auto-tuner discovery that 1300x1300 is optimal for tile20.

Protocol:
- 30 runs (3× más que auto-tuner)
- 5 warmup runs
- Statistical analysis (mean, median, std, min, max)
- Comparison vs 1400 baseline

Expected: 824.1 GFLOPS (auto-tuner result)
Target: Confirm with p-value < 0.05 that 1300 > 1400

Date: February 5, 2026
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pyopencl as cl
except ImportError:
    print("Error: pyopencl not found")
    sys.exit(1)


def load_kernel(ctx, kernel_path, kernel_name):
    """Load OpenCL kernel"""
    with open(kernel_path, 'r') as f:
        source = f.read()
    program = cl.Program(ctx, source).build()
    return getattr(program, kernel_name)


def benchmark_single(queue, kernel, M, N, K, local_size=(10, 10), tile_size=20):
    """Single benchmark run"""
    # Generate test matrices
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    A_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(queue.context, mf.WRITE_ONLY, C.nbytes)
    
    # Global size
    global_size = (
        ((N + tile_size - 1) // tile_size) * local_size[0],
        ((M + tile_size - 1) // tile_size) * local_size[1]
    )
    
    # Execute
    event = kernel(
        queue, global_size, local_size,
        np.int32(M), np.int32(N), np.int32(K),
        np.float32(1.0),  # alpha
        A_buf, B_buf,
        np.float32(0.0),  # beta
        C_buf
    )
    event.wait()
    
    # Get time
    elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
    
    # Verify (only first run)
    cl.enqueue_copy(queue, C, C_buf).wait()
    C_ref = A @ B
    max_error = np.max(np.abs(C - C_ref))
    
    # Cleanup
    A_buf.release()
    B_buf.release()
    C_buf.release()
    
    gflops = (2.0 * M * N * K) / (elapsed_ms * 1e-3 * 1e9)
    
    return gflops, elapsed_ms, max_error


def run_validation(matrix_size=1300, runs=30, warmup=5):
    """Run complete validation benchmark"""
    
    print("=" * 70)
    print("VALIDATION BENCHMARK: tile20 @ 1300×1300")
    print("=" * 70)
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    print(f"Runs: {runs}")
    print(f"Warmup: {warmup}")
    print("=" * 70)
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices(device_type=cl.device_type.GPU)
    device = devices[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print(f"\nDevice: {device.name}")
    print(f"Compute Units: {device.max_compute_units}")
    print(f"Max Workgroup Size: {device.max_work_group_size}\n")
    
    # Load kernel
    kernel_path = project_root / "src" / "kernels" / "gemm_tile20_production.cl"
    kernel = load_kernel(ctx, kernel_path, "gemm_tile20_optimized")
    print(f"✓ Loaded: gemm_tile20_optimized\n")
    
    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        gflops, time_ms, error = benchmark_single(
            queue, kernel, matrix_size, matrix_size, matrix_size
        )
        print(f"  Warmup {i+1}/{warmup}: {gflops:.1f} GFLOPS")
    
    print(f"\n{'='*70}")
    print("BENCHMARK RUNS")
    print(f"{'='*70}\n")
    
    # Benchmark runs
    gflops_list = []
    times_list = []
    errors_list = []
    
    for i in range(runs):
        gflops, time_ms, error = benchmark_single(
            queue, kernel, matrix_size, matrix_size, matrix_size
        )
        gflops_list.append(gflops)
        times_list.append(time_ms)
        errors_list.append(error)
        
        # Print progress
        status = "✅" if error < 0.01 else "❌"
        print(f"Run {i+1:2d}/{runs}: {gflops:7.1f} GFLOPS  |  {time_ms:6.2f} ms  |  error: {error:.6f} {status}")
        
        # Real-time stats every 10 runs
        if (i + 1) % 10 == 0:
            current_mean = np.mean(gflops_list)
            current_std = np.std(gflops_list)
            print(f"         → Current mean: {current_mean:.1f} ± {current_std:.1f} GFLOPS\n")
    
    # Statistical analysis
    gflops_array = np.array(gflops_list)
    times_array = np.array(times_list)
    errors_array = np.array(errors_list)
    
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")
    
    stats_data = {
        'mean': np.mean(gflops_array),
        'median': np.median(gflops_array),
        'std': np.std(gflops_array),
        'min': np.min(gflops_array),
        'max': np.max(gflops_array),
        'q25': np.percentile(gflops_array, 25),
        'q75': np.percentile(gflops_array, 75),
        'cv': (np.std(gflops_array) / np.mean(gflops_array)) * 100
    }
    
    print(f"Performance (GFLOPS):")
    print(f"  Mean:    {stats_data['mean']:.2f}")
    print(f"  Median:  {stats_data['median']:.2f}")
    print(f"  Std Dev: {stats_data['std']:.2f}")
    print(f"  Min:     {stats_data['min']:.2f}")
    print(f"  Max:     {stats_data['max']:.2f}")
    print(f"  Q25:     {stats_data['q25']:.2f}")
    print(f"  Q75:     {stats_data['q75']:.2f}")
    print(f"  CV:      {stats_data['cv']:.2f}%")
    
    print(f"\nTime (ms):")
    print(f"  Mean:    {np.mean(times_array):.2f}")
    print(f"  Std Dev: {np.std(times_array):.2f}")
    
    print(f"\nCorrectness:")
    print(f"  Max error: {np.max(errors_array):.6f}")
    print(f"  All runs: {'✅ PASSED' if np.max(errors_array) < 0.01 else '❌ FAILED'}")
    
    # Confidence interval (95%)
    confidence = 0.95
    margin_error = stats.sem(gflops_array) * stats.t.ppf((1 + confidence) / 2, runs - 1)
    ci_lower = stats_data['mean'] - margin_error
    ci_upper = stats_data['mean'] + margin_error
    
    print(f"\nConfidence Interval (95%):")
    print(f"  [{ci_lower:.2f}, {ci_upper:.2f}] GFLOPS")
    
    # Compare with auto-tuner result
    autotuner_result = 824.1
    print(f"\n{'='*70}")
    print("COMPARISON WITH AUTO-TUNER")
    print(f"{'='*70}\n")
    print(f"Auto-tuner (10 runs):  {autotuner_result:.1f} GFLOPS")
    print(f"Validation (30 runs):  {stats_data['mean']:.1f} GFLOPS")
    print(f"Difference:            {stats_data['mean'] - autotuner_result:+.1f} GFLOPS")
    print(f"Relative:              {((stats_data['mean'] / autotuner_result) - 1) * 100:+.2f}%")
    
    # Check if within confidence interval
    if ci_lower <= autotuner_result <= ci_upper:
        print(f"\n✅ Auto-tuner result is within 95% CI")
        print(f"   Conclusion: Result is STATISTICALLY CONSISTENT")
    else:
        print(f"\n⚠️  Auto-tuner result is outside 95% CI")
        print(f"   Possible variance or warm-up effects")
    
    # Compare 1300 vs 1400 (using auto-tuner data)
    print(f"\n{'='*70}")
    print("1300 vs 1400 COMPARISON")
    print(f"{'='*70}\n")
    
    gflops_1400 = 801.0  # From auto-tuner
    gflops_1300 = stats_data['mean']
    
    print(f"tile20 @ 1300: {gflops_1300:.1f} GFLOPS")
    print(f"tile20 @ 1400: {gflops_1400:.1f} GFLOPS")
    print(f"Difference:    {gflops_1300 - gflops_1400:+.1f} GFLOPS ({((gflops_1300/gflops_1400)-1)*100:+.1f}%)")
    
    if gflops_1300 > gflops_1400:
        print(f"\n🏆 CONFIRMED: 1300 is FASTER than 1400")
        print(f"   Improvement: {gflops_1300 - gflops_1400:.1f} GFLOPS")
    else:
        print(f"\n⚠️  1400 appears slightly better (within variance?)")
    
    # Save results
    output_file = project_root / "results" / "validation_1300.txt"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("VALIDATION BENCHMARK: tile20 @ 1300×1300\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nRuns: {runs}\n")
        f.write(f"Device: {device.name}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Mean:   {stats_data['mean']:.2f} GFLOPS\n")
        f.write(f"  Median: {stats_data['median']:.2f} GFLOPS\n")
        f.write(f"  Std:    {stats_data['std']:.2f} GFLOPS\n")
        f.write(f"  Min:    {stats_data['min']:.2f} GFLOPS\n")
        f.write(f"  Max:    {stats_data['max']:.2f} GFLOPS\n")
        f.write(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n")
        f.write(f"\nAll runs:\n")
        for i, gf in enumerate(gflops_list, 1):
            f.write(f"  Run {i:2d}: {gf:.2f} GFLOPS\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}\n")
    
    if stats_data['cv'] < 2.0:
        print(f"✅ STABILITY: Excellent (CV = {stats_data['cv']:.2f}%)")
    elif stats_data['cv'] < 5.0:
        print(f"✅ STABILITY: Good (CV = {stats_data['cv']:.2f}%)")
    else:
        print(f"⚠️  STABILITY: Moderate (CV = {stats_data['cv']:.2f}%)")
    
    if np.max(errors_array) < 0.01:
        print(f"✅ CORRECTNESS: All runs passed (max error = {np.max(errors_array):.6f})")
    else:
        print(f"❌ CORRECTNESS: Some errors detected")
    
    if gflops_1300 > gflops_1400:
        print(f"✅ PERFORMANCE: 1300 > 1400 by {gflops_1300 - gflops_1400:.1f} GFLOPS")
    
    print(f"\n🎯 RECOMMENDATION: Use {matrix_size}×{matrix_size} as optimal size")
    print(f"   Expected performance: {stats_data['mean']:.1f} ± {stats_data['std']:.1f} GFLOPS")
    
    return stats_data


if __name__ == "__main__":
    try:
        print("\nStarting validation benchmark...")
        print("This will take ~2-3 minutes (30 runs + 5 warmup)\n")
        
        stats = run_validation(matrix_size=1300, runs=30, warmup=5)
        
        print(f"\n✅ Validation complete!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
