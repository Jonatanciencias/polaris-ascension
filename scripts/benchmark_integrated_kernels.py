#!/usr/bin/env python3
"""
Comprehensive Benchmark: Integrated FLOAT4 Kernels
Phase 1 Extension - Opción B
"""

import sys
sys.path.insert(0, '/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580')

from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine, KernelType
import numpy as np
from pathlib import Path
import time

def benchmark_kernel(engine, M, N, K, kernel_type, warmup=5, iterations=20):
    """Benchmark a specific kernel with warmup"""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_ref = A @ B
    
    # Warmup
    for _ in range(warmup):
        engine.gemm(A, B, kernel_type=kernel_type)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        result = engine.gemm(A, B, kernel_type=kernel_type)
        times.append(result.kernel_metrics.exec_time_ms)
    
    # Calculate stats
    avg_time = np.mean(times)
    best_time = np.min(times)
    ops = 2 * M * N * K
    avg_gflops = ops / avg_time / 1e6
    peak_gflops = ops / best_time / 1e6
    
    # Verify correctness
    result = engine.gemm(A, B, kernel_type=kernel_type)
    max_error = np.max(np.abs(result.result - C_ref))
    is_correct = max_error < 0.01
    
    return {
        'kernel': kernel_type.name,
        'size': f"{M}×{N}×{K}",
        'avg_time_ms': avg_time,
        'best_time_ms': best_time,
        'avg_gflops': avg_gflops,
        'peak_gflops': peak_gflops,
        'max_error': max_error,
        'correct': is_correct
    }

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " Comprehensive Benchmark: Integrated FLOAT4 Kernels".center(68) + "║")
    print("║" + " Phase 1 Extension - Opción B".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    # Clean cache
    cache_dir = Path('.kernel_cache')
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("✅ Kernel cache cleaned")
    
    cache_dir2 = Path.home() / ".cache" / "radeon_rx580_kernels"
    if cache_dir2.exists():
        import shutil
        shutil.rmtree(cache_dir2)
        print("✅ Engine cache cleaned\n")
    
    # Initialize engine
    print("🔧 Initializing OptimizedKernelEngine...")
    engine = OptimizedKernelEngine(enable_profiling=True)
    print(f"✅ Engine initialized: {engine.device.name}\n")
    
    # Test configurations
    test_configs = [
        # (M, N, K, KernelType, Description)
        (128, 128, 128, KernelType.GEMM_FLOAT4_SMALL, "Small - High Occupancy"),
        (256, 256, 256, KernelType.GEMM_FLOAT4_SMALL, "Medium - Target Size"),
        (512, 512, 512, KernelType.GEMM_FLOAT4_SMALL, "Large - Small Tiles"),
        (512, 512, 512, KernelType.GEMM_FLOAT4_CLOVER, "Large - Medium Tiles"),
        (1024, 1024, 1024, KernelType.GEMM_FLOAT4_CLOVER, "Very Large - 16x16 Tiles"),
        (2048, 2048, 2048, KernelType.GEMM_GCN4_ULTRA, "Huge - GCN4 Ultra"),
    ]
    
    results = []
    for M, N, K, kernel_type, desc in test_configs:
        print(f"📊 Benchmarking {desc}")
        print(f"   Size: {M}×{N}×{K}, Kernel: {kernel_type.name}")
        
        try:
            result = benchmark_kernel(engine, M, N, K, kernel_type)
            results.append(result)
            
            status = "✅" if result['correct'] else "❌"
            print(f"   Avg: {result['avg_gflops']:.2f} GFLOPS ({result['avg_time_ms']:.3f} ms)")
            print(f"   Peak: {result['peak_gflops']:.2f} GFLOPS ({result['best_time_ms']:.3f} ms)")
            print(f"   Correctness: {status} (error: {result['max_error']:.6f})\n")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}\n")
    
    # Summary
    print("=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Size':<15} {'Kernel':<25} {'Avg GFLOPS':<12} {'Peak GFLOPS':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✅ PASS" if r['correct'] else "❌ FAIL"
        print(f"{r['size']:<15} {r['kernel']:<25} {r['avg_gflops']:>10.2f}  {r['peak_gflops']:>12.2f}  {status}")
    
    # Best results
    valid_results = [r for r in results if r['correct']]
    if valid_results:
        best_avg = max(valid_results, key=lambda x: x['avg_gflops'])
        best_peak = max(valid_results, key=lambda x: x['peak_gflops'])
        
        print(f"\n🏆 Best Average Performance:")
        print(f"   {best_avg['peak_gflops']:.2f} GFLOPS ({best_avg['kernel']} @ {best_avg['size']})")
        
        print(f"\n⚡ Best Peak Performance:")
        print(f"   {best_peak['peak_gflops']:.2f} GFLOPS ({best_peak['kernel']} @ {best_peak['size']})")
        
        # Compare with targets
        baseline = 150.96
        target = 200.0
        achieved = best_peak['peak_gflops']
        
        print(f"\n📈 Performance vs Targets:")
        print(f"   Baseline (GCN4_ULTRA):     {baseline:.2f} GFLOPS")
        print(f"   Phase 1 Target:            {target:.2f} GFLOPS")
        print(f"   Achieved (Integrated):     {achieved:.2f} GFLOPS")
        print(f"   Speedup vs Baseline:       {achieved/baseline:.2f}×")
        print(f"   vs Target:                 {(achieved/target)*100:.1f}%")
        
        if achieved >= target:
            print(f"\n   ✅ Phase 1 Target ACHIEVED with integrated kernels!")
            print(f"   🎉 Exceeded by {achieved - target:.2f} GFLOPS ({((achieved/target)-1)*100:.1f}%)")
        else:
            print(f"\n   ⚠️  Below target by {target - achieved:.2f} GFLOPS")
    
    print("\n✅ Benchmark complete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
