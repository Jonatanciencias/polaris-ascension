#!/usr/bin/env python3
"""
Sweet Spot Refinement - Phase 2.1 Step 1

Objetivo: Probar tamaños alrededor de 1280 para confirmar peak óptimo
Tamaños: 1200, 1250, 1280, 1320, 1350, 1400, 1450

Expected: 1280 es probablemente el peak, pero puede haber mejoras menores
Potencial: +10-15 GFLOPS si encontramos mejor sweet spot
"""

import numpy as np
import pyopencl as cl
import time
import json
from pathlib import Path

def benchmark_gemm(ctx, queue, kernel_code, M, N, K, warmup=3, iterations=10):
    """Benchmark GEMM kernel con estadísticas detalladas"""
    
    # Compile kernel
    prg = cl.Program(ctx, kernel_code).build()
    
    # Allocate matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    
    # Global and local sizes
    local_size = (10, 10)  # tile20 uses 10×10 workgroup
    global_size = (
        ((N + 19) // 20) * local_size[0],
        ((M + 19) // 20) * local_size[1]
    )
    
    # Warmup
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    
    for _ in range(warmup):
        prg.gemm_tile20_vectorized(
            queue, global_size, local_size,
            np.int32(M), np.int32(N), np.int32(K),
            alpha, a_buf, b_buf, beta, c_buf
        )
    queue.finish()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        prg.gemm_tile20_vectorized(
            queue, global_size, local_size,
            np.int32(M), np.int32(N), np.int32(K),
            alpha, a_buf, b_buf, beta, c_buf
        )
        queue.finish()
        end = time.perf_counter()
        times.append(end - start)
    
    # Correctness check
    cl.enqueue_copy(queue, C, c_buf)
    C_ref = A @ B
    max_error = np.max(np.abs(C - C_ref))
    
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
    
    return {
        'gflops_avg': gflops_avg,
        'gflops_best': gflops_best,
        'time_avg_ms': avg_time * 1000,
        'time_std_ms': std_time * 1000,
        'time_min_ms': min_time * 1000,
        'time_max_ms': max_time * 1000,
        'max_error': max_error,
        'iterations': iterations
    }


def load_tile20_kernel():
    """Load the best tile20 kernel (approach_2_v3_vectorized.cl)"""
    kernel_path = Path(__file__).parent / "kernels" / "approach_2_v3_vectorized.cl"
    
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    
    with open(kernel_path, 'r') as f:
        return f.read()


def main():
    print("=" * 80)
    print("SWEET SPOT REFINEMENT - Phase 2.1 Step 1")
    print("=" * 80)
    print()
    print("Objetivo: Encontrar el tamaño óptimo alrededor de 1280×1280")
    print("Kernel: tile20_vectorized (approach 2 v3)")
    print()
    
    # Initialize OpenCL
    platforms = cl.get_platforms()
    
    # Debug: print all platforms
    print(f"Available platforms: {len(platforms)}")
    for i, p in enumerate(platforms):
        print(f"  [{i}] {p.name}")
    print()
    
    # Try to find AMD or any available platform
    platform = None
    for p in platforms:
        if 'amd' in p.name.lower() or 'advanced micro devices' in p.name.lower() or 'mesa' in p.name.lower():
            platform = p
            break
    
    if platform is None and platforms:
        # Use first available platform
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
    print()
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Load kernel
    print("Loading tile20 v3 vectorized kernel...")
    kernel_code = load_tile20_kernel()
    print("✓ Kernel loaded")
    print()
    
    # Test sizes around 1280 (current best)
    test_sizes = [
        1200,  # -80 from current best
        1250,  # -30
        1280,  # Current best (745.6 GFLOPS)
        1320,  # +40
        1350,  # +70
        1400,  # +120
        1450,  # +170
    ]
    
    results = []
    
    print("=" * 80)
    print("BENCHMARKING")
    print("=" * 80)
    print()
    
    for size in test_sizes:
        print(f"Testing {size}×{size}...")
        
        try:
            result = benchmark_gemm(
                ctx, queue, kernel_code,
                M=size, N=size, K=size,
                warmup=3,
                iterations=10
            )
            
            result['size'] = size
            results.append(result)
            
            # Print result
            status = "✅" if result['max_error'] < 0.1 else "❌"
            print(f"  Size: {size:4d} | "
                  f"GFLOPS: {result['gflops_avg']:7.1f} ± {result['gflops_avg'] - result['gflops_best']:5.1f} | "
                  f"Time: {result['time_avg_ms']:6.2f}ms ± {result['time_std_ms']:5.2f}ms | "
                  f"Error: {result['max_error']:.6f} {status}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'size': size,
                'gflops_avg': 0.0,
                'error': str(e)
            })
        
        print()
    
    # Analysis
    print("=" * 80)
    print("ANÁLISIS DE RESULTADOS")
    print("=" * 80)
    print()
    
    # Find best size
    valid_results = [r for r in results if r['gflops_avg'] > 0]
    if not valid_results:
        print("❌ No valid results!")
        return
    
    best = max(valid_results, key=lambda x: x['gflops_avg'])
    worst = min(valid_results, key=lambda x: x['gflops_avg'])
    
    print(f"🏆 BEST: {best['size']}×{best['size']}")
    print(f"   Performance: {best['gflops_avg']:.1f} GFLOPS (best: {best['gflops_best']:.1f})")
    print(f"   Time: {best['time_avg_ms']:.2f}ms ± {best['time_std_ms']:.2f}ms")
    print(f"   Error: {best['max_error']:.6f}")
    print()
    
    print(f"📊 RANGE:")
    print(f"   Best:  {best['size']} @ {best['gflops_avg']:.1f} GFLOPS")
    print(f"   Worst: {worst['size']} @ {worst['gflops_avg']:.1f} GFLOPS")
    print(f"   Spread: {best['gflops_avg'] - worst['gflops_avg']:.1f} GFLOPS ({((best['gflops_avg'] - worst['gflops_avg']) / worst['gflops_avg'] * 100):.1f}%)")
    print()
    
    # Check if 1280 is still best
    result_1280 = next((r for r in valid_results if r['size'] == 1280), None)
    if result_1280:
        if best['size'] == 1280:
            print("✅ CONFIRMADO: 1280 sigue siendo el sweet spot óptimo")
        else:
            improvement = best['gflops_avg'] - result_1280['gflops_avg']
            print(f"🎯 NUEVO PEAK ENCONTRADO: {best['size']}")
            print(f"   Mejora sobre 1280: +{improvement:.1f} GFLOPS (+{(improvement / result_1280['gflops_avg'] * 100):.1f}%)")
    
    print()
    
    # Summary table
    print("TABLA COMPLETA:")
    print()
    print("Size   | Avg GFLOPS | Best GFLOPS | Time (ms) | Error     | vs 1280")
    print("-------|------------|-------------|-----------|-----------|----------")
    
    for r in valid_results:
        diff = ""
        if result_1280:
            delta = r['gflops_avg'] - result_1280['gflops_avg']
            diff = f"{delta:+6.1f}"
        
        marker = "🏆" if r['size'] == best['size'] else "  "
        print(f"{r['size']:4d} {marker} | {r['gflops_avg']:10.1f} | {r['gflops_best']:11.1f} | "
              f"{r['time_avg_ms']:9.2f} | {r['max_error']:9.6f} | {diff}")
    
    print()
    
    # Save results
    output_file = Path(__file__).parent / "sweet_spot_refinement_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'kernel': 'tile20_v3_vectorized',
            'test_sizes': test_sizes,
            'results': results,
            'best_size': best['size'],
            'best_gflops': best['gflops_avg'],
            'device': device.name
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Recommendation
    print("=" * 80)
    print("RECOMENDACIÓN")
    print("=" * 80)
    print()
    
    if best['size'] == 1280:
        print("✅ Mantener 1280×1280 como sweet spot")
        print("   No se encontró mejora significativa en otros tamaños")
        print("   Proceder a Step 2: Implementar tile=24 kernel")
    else:
        improvement = best['gflops_avg'] - result_1280['gflops_avg']
        if improvement > 10:
            print(f"🎯 USAR {best['size']}×{best['size']} como nuevo sweet spot")
            print(f"   Mejora significativa: +{improvement:.1f} GFLOPS")
            print("   Actualizar neural_predictor_dataset.json")
            print("   Proceder a Step 2: Implementar tile=24 kernel")
        else:
            print(f"⚠️ {best['size']} es ligeramente mejor (+{improvement:.1f} GFLOPS)")
            print("   Mejora marginal, MANTENER 1280 por simplicidad")
            print("   Proceder a Step 2: Implementar tile=24 kernel")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
