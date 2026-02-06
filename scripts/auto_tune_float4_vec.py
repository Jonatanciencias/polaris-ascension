#!/usr/bin/env python3
"""
Auto-Tuner for FLOAT4_VEC Kernel - Tile Size Optimization

Goal: Find optimal tile sizes to push from 566 GFLOPS to 600+ GFLOPS

Tuning parameters:
- TILE_SIZE: 8, 12, 16, 20, 24, 32
- LOCAL_SIZE: (8,8), (16,16), (16,8), (8,16), (32,8)
- UNROLL_FACTOR: 2, 4, 8, 16
"""

import sys
import os
import numpy as np
import time
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    print("⚠️  PyOpenCL not available")
    sys.exit(1)


def create_tuned_kernel(tile_size, local_m, local_n, unroll_k):
    """Generate optimized kernel with specific parameters"""
    
    kernel_source = f"""
// Auto-tuned FLOAT4_VEC kernel
// TILE_SIZE={tile_size}, LOCAL=({local_m},{local_n}), UNROLL={unroll_k}

#define TILE_SIZE {tile_size}
#define LOCAL_M {local_m}
#define LOCAL_N {local_n}
#define UNROLL_K {unroll_k}

__kernel void gemm_float4_vec_tuned(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {{
    // Local memory - cada work-item carga 1 float de A y 4 floats de B
    __local float As[TILE_SIZE * TILE_SIZE];
    __local float Bs[TILE_SIZE * TILE_SIZE * 4];  // 4× para 4 columnas
    
    // Work-item indices
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col_base = get_global_id(1) * 4;
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);
    
    // Accumulator
    float4 sum = (float4)(0.0f);
    
    // Number of tiles
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K in tiles
    for (int t = 0; t < num_tiles; t++) {{
        // Load tile from A
        const int a_row = group_row * TILE_SIZE + local_row;
        const int a_col = t * TILE_SIZE + local_col;
        
        if (a_row < M && a_col < K) {{
            As[local_row * TILE_SIZE + local_col] = A[a_row * K + a_col];
        }} else {{
            As[local_row * TILE_SIZE + local_col] = 0.0f;
        }}
        
        // Load tile from B (4 columnas por work-item)
        const int b_row = t * TILE_SIZE + local_row;
        const int b_col_base = group_col * TILE_SIZE * 4 + local_col * 4;
        const int lds_offset = local_row * TILE_SIZE * 4 + local_col * 4;
        
        if (b_row < K && b_col_base + 3 < N) {{
            // Vectorized load
            float4 b_vec = vload4(0, &B[b_row * N + b_col_base]);
            Bs[lds_offset + 0] = b_vec.x;
            Bs[lds_offset + 1] = b_vec.y;
            Bs[lds_offset + 2] = b_vec.z;
            Bs[lds_offset + 3] = b_vec.w;
        }} else if (b_row < K) {{
            // Boundary
            for (int i = 0; i < 4; i++) {{
                if (b_col_base + i < N) {{
                    Bs[lds_offset + i] = B[b_row * N + b_col_base + i];
                }} else {{
                    Bs[lds_offset + i] = 0.0f;
                }}
            }}
        }} else {{
            Bs[lds_offset + 0] = 0.0f;
            Bs[lds_offset + 1] = 0.0f;
            Bs[lds_offset + 2] = 0.0f;
            Bs[lds_offset + 3] = 0.0f;
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute with unrolling
        #pragma unroll {unroll_k}
        for (int k = 0; k < TILE_SIZE; k++) {{
            float a_val = As[local_row * TILE_SIZE + k];
            
            const int b_offset = k * TILE_SIZE * 4 + local_col * 4;
            float4 b_vec;
            b_vec.x = Bs[b_offset + 0];
            b_vec.y = Bs[b_offset + 1];
            b_vec.z = Bs[b_offset + 2];
            b_vec.w = Bs[b_offset + 3];
            
            sum = mad(a_val, b_vec, sum);
        }}
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
    // Write results
    if (global_row < M && global_col_base + 3 < N) {{
        const int c_idx = global_row * N + global_col_base;
        
        float4 c_vec;
        if (beta == 0.0f) {{
            c_vec = alpha * sum;
        }} else {{
            c_vec = mad(alpha, sum, beta * vload4(0, &C[c_idx]));
        }}
        
        vstore4(c_vec, 0, &C[c_idx]);
    }} else if (global_row < M) {{
        for (int i = 0; i < 4 && global_col_base + i < N; i++) {{
            const int c_idx = global_row * N + global_col_base + i;
            float val = (i == 0) ? sum.x : (i == 1) ? sum.y : (i == 2) ? sum.z : sum.w;
            
            if (beta == 0.0f) {{
                C[c_idx] = alpha * val;
            }} else {{
                C[c_idx] = mad(alpha, val, beta * C[c_idx]);
            }}
        }}
    }}
}}
"""
    return kernel_source


def benchmark_configuration(ctx, queue, device, tile_size, local_m, local_n, unroll_k, size=2048):
    """Benchmark a specific configuration"""
    
    # Generate kernel
    try:
        kernel_source = create_tuned_kernel(tile_size, local_m, local_n, unroll_k)
        program = cl.Program(ctx, kernel_source).build(options="-cl-fast-relaxed-math")
    except Exception as e:
        return None, f"Compilation failed: {e}"
    
    # Create matrices
    M = N = K = size
    N_aligned = ((N + 3) // 4) * 4
    
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N_aligned).astype(np.float32)
    C = np.zeros((M, N_aligned), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
    
    # Setup kernel
    kernel = program.gemm_float4_vec_tuned
    local_size = (local_m, local_n)
    global_size = (
        ((M + local_m - 1) // local_m) * local_m,
        ((N_aligned // 4 + local_n - 1) // local_n) * local_n
    )
    
    kernel.set_args(
        np.int32(M), np.int32(N_aligned), np.int32(K),
        np.float32(1.0),
        A_buf, B_buf,
        np.float32(0.0),
        C_buf
    )
    
    # Warmup
    try:
        for _ in range(3):
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
    except Exception as e:
        return None, f"Execution failed: {e}"
    
    # Benchmark
    iterations = 10
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
        end = time.perf_counter()
        times.append(end - start)
    
    # Calculate GFLOPS
    min_time = np.min(times)
    ops = 2 * M * N_aligned * K
    gflops = (ops / min_time) / 1e9
    
    # Verify correctness
    result = np.empty((M, N_aligned), dtype=np.float32)
    cl.enqueue_copy(queue, result, C_buf).wait()
    
    expected = A @ B
    max_error = np.max(np.abs(result - expected))
    
    if max_error > 0.1:
        return None, f"Correctness failed: error={max_error:.4f}"
    
    return gflops, "OK"


def auto_tune_kernel(size=2048):
    """Auto-tune kernel parameters"""
    
    print("="*70)
    print("🔬 FLOAT4_VEC AUTO-TUNER")
    print("="*70)
    print(f"\nMatrix size: {size}×{size}")
    print(f"Goal: Optimize tile sizes to reach 600+ GFLOPS\n")
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    device = devices[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    print(f"Device: {device.name}")
    print(f"Max work group size: {device.max_work_group_size}")
    print(f"Local mem size: {device.local_mem_size / 1024:.0f} KB")
    
    # Parameter space
    tile_sizes = [12, 16, 20, 24]
    local_configs = [(8, 8), (16, 16), (8, 16), (16, 8), (12, 12)]
    unroll_factors = [2, 4, 8]
    
    print(f"\nSearch space:")
    print(f"  Tile sizes: {tile_sizes}")
    print(f"  Local configs: {local_configs}")
    print(f"  Unroll factors: {unroll_factors}")
    print(f"  Total configs: {len(tile_sizes) * len(local_configs) * len(unroll_factors)}")
    
    print(f"\n{'='*70}")
    print("Starting auto-tuning...")
    print(f"{'='*70}\n")
    
    results = []
    best_gflops = 0
    best_config = None
    
    total_configs = len(list(product(tile_sizes, local_configs, unroll_factors)))
    current = 0
    
    for tile_size in tile_sizes:
        for local_m, local_n in local_configs:
            # Check work group size
            if local_m * local_n > device.max_work_group_size:
                continue
            
            # Check LDS requirements
            lds_required = (tile_size * tile_size + tile_size * tile_size * 4) * 4  # bytes
            if lds_required > device.local_mem_size:
                continue
            
            for unroll_k in unroll_factors:
                current += 1
                config_name = f"T{tile_size}_L{local_m}x{local_n}_U{unroll_k}"
                
                print(f"[{current}/{total_configs}] Testing {config_name}... ", end='', flush=True)
                
                gflops, status = benchmark_configuration(
                    ctx, queue, device, tile_size, local_m, local_n, unroll_k, size
                )
                
                if gflops is not None:
                    print(f"{gflops:.2f} GFLOPS ✅")
                    results.append({
                        'tile_size': tile_size,
                        'local_m': local_m,
                        'local_n': local_n,
                        'unroll_k': unroll_k,
                        'gflops': gflops,
                        'config_name': config_name
                    })
                    
                    if gflops > best_gflops:
                        best_gflops = gflops
                        best_config = (tile_size, local_m, local_n, unroll_k)
                        print(f"     🏆 NEW BEST!")
                else:
                    print(f"❌ {status}")
    
    # Sort results
    results.sort(key=lambda x: x['gflops'], reverse=True)
    
    # Display results
    print(f"\n{'='*70}")
    print("📊 AUTO-TUNING RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Rank':<6} {'Config':<20} {'GFLOPS':<12} {'vs Baseline'}")
    print("-"*70)
    
    baseline = 566  # Current best
    for i, result in enumerate(results[:10], 1):
        improvement = ((result['gflops'] / baseline) - 1) * 100
        marker = "🏆" if i == 1 else "  "
        print(f"{marker} #{i:<4} {result['config_name']:<20} {result['gflops']:<12.2f} {improvement:+.1f}%")
    
    if best_config:
        print(f"\n{'='*70}")
        print("🏆 BEST CONFIGURATION")
        print(f"{'='*70}\n")
        
        tile_size, local_m, local_n, unroll_k = best_config
        print(f"Tile size:     {tile_size}")
        print(f"Local size:    ({local_m}, {local_n})")
        print(f"Unroll factor: {unroll_k}")
        print(f"Performance:   {best_gflops:.2f} GFLOPS")
        print(f"Improvement:   {((best_gflops / baseline) - 1) * 100:+.1f}% vs baseline ({baseline} GFLOPS)")
        
        if best_gflops >= 600:
            print(f"\n🎉 TARGET ACHIEVED: {best_gflops:.2f} GFLOPS >= 600 GFLOPS!")
        else:
            print(f"\n📈 Progress: {best_gflops:.2f} / 600 GFLOPS ({best_gflops/600*100:.1f}%)")
    
    return results, best_config


def main():
    """Main auto-tuning routine"""
    results, best_config = auto_tune_kernel(size=2048)
    
    if best_config and results:
        print(f"\n{'='*70}")
        print("💾 RECOMMENDED ACTION")
        print(f"{'='*70}\n")
        
        tile_size, local_m, local_n, unroll_k = best_config
        best_gflops = results[0]['gflops']
        
        print("Update gemm_float4_clover.cl with optimal parameters:")
        print(f"\n#define CLOVER_TILE_16 {tile_size}   // Optimized tile size")
        print(f"local_size = ({local_m}, {local_n})   // In kernel config")
        print(f"#pragma unroll {unroll_k}           // In compute loop")
        
        print(f"\nExpected performance: {best_gflops:.2f} GFLOPS")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
