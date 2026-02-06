#!/usr/bin/env python3
"""
Benchmark de Kernels GCN4-Specific para AMD Radeon RX 580/590

Este script evalúa el rendimiento de los nuevos kernels optimizados
específicamente para la arquitectura GCN 4.0 (Polaris).

Meta: 3-5× mejora sobre el kernel básico
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization_engines.optimized_kernel_engine import (
    OptimizedKernelEngine, 
    KernelType
)


def benchmark_kernel(engine, kernel_type, M, N, K, warmup=3, iterations=10):
    """Benchmark un kernel específico."""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        try:
            result = engine.gemm(A, B, kernel_type=kernel_type)
        except Exception as e:
            return None, str(e)
    
    # Benchmark
    times = []
    gflops_list = []
    
    for _ in range(iterations):
        result = engine.gemm(A, B, kernel_type=kernel_type)
        times.append(result.kernel_metrics.exec_time_ms)
        gflops_list.append(result.kernel_metrics.gflops)
    
    # Verify correctness
    cpu_result = np.dot(A, B)
    rel_error = np.max(np.abs(result.result - cpu_result)) / np.max(np.abs(cpu_result))
    
    return {
        'avg_time_ms': np.mean(times),
        'min_time_ms': np.min(times),
        'avg_gflops': np.mean(gflops_list),
        'max_gflops': np.max(gflops_list),
        'rel_error': rel_error,
        'correct': rel_error < 1e-3
    }, None


def main():
    print("=" * 80)
    print("🚀 BENCHMARK: Kernels GCN4-Specific para Polaris (RX 580/590)")
    print("=" * 80)
    print()
    
    # Initialize engine
    print("Inicializando OptimizedKernelEngine...")
    engine = OptimizedKernelEngine(enable_profiling=True, enable_advanced_memory=True)
    print(f"GPU: {engine.device.name}")
    print(f"Compute Units: {engine.max_compute_units}")
    print(f"Max Work Group Size: {engine.max_work_group_size}")
    print()
    
    # Test matrix sizes
    test_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    # Kernels to test
    kernels_to_test = [
        (KernelType.GEMM_BASIC, "GEMM Basic (16x16 tiled)"),
        (KernelType.GEMM_GCN4_HIGH_OCCUPANCY, "GCN4 High Occupancy"),
        (KernelType.GEMM_GCN4_ULTRA, "GCN4 Ultra (8x8 blocking)"),
        (KernelType.GEMM_GCN4_STREAMING, "GCN4 Streaming"),
    ]
    
    results = {}
    
    for M, N, K in test_sizes:
        print(f"\n{'='*80}")
        print(f"📊 Tamaño de Matriz: {M}x{N}x{K}")
        print(f"   Operaciones: {2*M*N*K/1e9:.2f} GFLOPs")
        print(f"   Memoria: {(M*K + K*N + M*N)*4/1e6:.1f} MB")
        print("=" * 80)
        
        results[(M, N, K)] = {}
        baseline_gflops = None
        
        for kernel_type, kernel_name in kernels_to_test:
            try:
                # Skip GCN4_ULTRA for non-aligned sizes
                if kernel_type in (KernelType.GEMM_GCN4_ULTRA, KernelType.GEMM_GCN4_STREAMING):
                    if M % 64 != 0 or N % 64 != 0 or K % 16 != 0:
                        print(f"  ⏭️  {kernel_name}: Skipped (size not aligned)")
                        continue
                
                result, error = benchmark_kernel(engine, kernel_type, M, N, K)
                
                if error:
                    print(f"  ❌ {kernel_name}: Error - {error}")
                    continue
                
                results[(M, N, K)][kernel_name] = result
                
                # Store baseline for comparison
                if kernel_type == KernelType.GEMM_BASIC:
                    baseline_gflops = result['avg_gflops']
                
                # Calculate speedup
                speedup = ""
                if baseline_gflops and baseline_gflops > 0:
                    speedup_val = result['avg_gflops'] / baseline_gflops
                    speedup = f"(×{speedup_val:.2f})"
                
                # Print result
                status = "✅" if result['correct'] else "❌"
                print(f"  {status} {kernel_name:30} | "
                      f"{result['avg_gflops']:7.1f} GFLOPS {speedup:8} | "
                      f"{result['avg_time_ms']:7.2f} ms | "
                      f"err={result['rel_error']:.2e}")
                
            except Exception as e:
                print(f"  ❌ {kernel_name}: Exception - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 RESUMEN DE RENDIMIENTO")
    print("=" * 80)
    
    # Find best kernel for each size
    for (M, N, K), size_results in results.items():
        if not size_results:
            continue
            
        best_kernel = max(size_results.items(), key=lambda x: x[1]['avg_gflops'])
        baseline = size_results.get("GEMM Basic (16x16 tiled)", {}).get('avg_gflops', 0)
        
        if baseline > 0:
            speedup = best_kernel[1]['avg_gflops'] / baseline
            print(f"  {M}x{N}: {best_kernel[0]} - {best_kernel[1]['avg_gflops']:.1f} GFLOPS "
                  f"(×{speedup:.2f} vs baseline)")
    
    # Overall statistics
    print("\n" + "-" * 80)
    all_gcn4_gflops = []
    all_baseline_gflops = []
    
    for size_results in results.values():
        for name, result in size_results.items():
            if "GCN4" in name:
                all_gcn4_gflops.append(result['avg_gflops'])
            elif "Basic" in name:
                all_baseline_gflops.append(result['avg_gflops'])
    
    if all_gcn4_gflops and all_baseline_gflops:
        avg_gcn4 = np.mean(all_gcn4_gflops)
        max_gcn4 = np.max(all_gcn4_gflops)
        avg_baseline = np.mean(all_baseline_gflops)
        
        print(f"\n  📊 Estadísticas GCN4:")
        print(f"     Promedio: {avg_gcn4:.1f} GFLOPS")
        print(f"     Máximo:   {max_gcn4:.1f} GFLOPS")
        print(f"     Baseline: {avg_baseline:.1f} GFLOPS")
        print(f"     Mejora:   ×{avg_gcn4/avg_baseline:.2f} promedio, ×{max_gcn4/avg_baseline:.2f} máximo")
        
        # Check if we met the target
        target_met = avg_gcn4 / avg_baseline >= 3.0
        print(f"\n  🎯 Meta (3-5× mejora): {'✅ ALCANZADA' if target_met else '⚠️ EN PROGRESO'}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark completado")
    print("=" * 80)


if __name__ == "__main__":
    main()
