#!/usr/bin/env python3
"""
Diagnose REGISTER_TILED Kernel Issue
Task 1.1.4: Fix REGISTER_TILED for Clover
"""

import sys
sys.path.insert(0, '/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580')

import pyopencl as cl
import numpy as np
from pathlib import Path

def test_original_register_tiled():
    """Test the original REGISTER_TILED kernel"""
    print("="*70)
    print(" TEST 1: Original REGISTER_TILED Kernel")
    print("="*70)
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    gpu = None
    for dev in devices:
        if dev.type == cl.device_type.GPU:
            gpu = dev
            break
    
    if not gpu:
        print("❌ No GPU found")
        return False
    
    print(f"Device: {gpu.name}")
    
    context = cl.Context([gpu])
    queue = cl.CommandQueue(context)
    
    # Load kernel
    kernel_path = Path("src/opencl/kernels/gemm_rx580_optimized.cl")
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    # Try to compile with REGISTER_TILED
    build_options = (
        "-D TILE_SIZE=32 "
        "-D TILE_K=16 "
        "-D WPT_REG=8 "
        "-D LDS_PADDING=1 "
        "-cl-mad-enable "
        "-cl-fast-relaxed-math "
        "-cl-std=CL1.1"
    )
    
    print(f"Build options: {build_options}")
    print("Compiling...")
    
    try:
        program = cl.Program(context, kernel_code).build(options=build_options)
        print("✅ Compilation successful!")
        
        # Try to get the kernel
        kernel = program.gemm_register_tiled
        print("✅ Kernel accessible!")
        
        # Try simple execution
        M = N = K = 128
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        
        mf = cl.mem_flags
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
        
        kernel.set_args(np.int32(M), np.int32(N), np.int32(K), A_buf, B_buf, C_buf)
        
        # Work size calculation (4x4 workgroups, each thread does 8x8 work)
        local_size = (4, 4)  # 4x4 threads per workgroup
        global_size = (
            ((M + 31) // 32) * 4,  # 32 elements per workgroup in M
            ((N + 31) // 32) * 4   # 32 elements per workgroup in N
        )
        
        print(f"Work sizes: local={local_size}, global={global_size}")
        
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        event.wait()
        
        cl.enqueue_copy(queue, C, C_buf).wait()
        
        # Verify
        C_ref = A @ B
        max_error = np.max(np.abs(C - C_ref))
        print(f"Max error: {max_error:.6f}")
        
        if max_error < 0.1:
            print("✅ Original kernel works!")
            return True
        else:
            print(f"❌ Large error: {max_error}")
            return False
            
    except cl.RuntimeError as e:
        print(f"❌ Compilation error: {e}")
        return False
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clover_compatible_version():
    """Test a Clover-compatible version without restrict"""
    print("\n" + "="*70)
    print(" TEST 2: Clover-Compatible Version (No restrict)")
    print("="*70)
    
    # Create simplified kernel
    clover_kernel = """
// Clover-compatible REGISTER_TILED kernel
#define TILE_SIZE 32
#define TILE_K 16
#define WPT_REG 8
#define LDS_PADDING 1

__kernel void gemm_register_tiled_clover(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C
) {
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    
    __local float As[TILE_SIZE * (TILE_K + LDS_PADDING)];
    __local float Bs[TILE_K * (TILE_SIZE + LDS_PADDING)];
    
    // Simplified: 4x4 accumulator instead of 8x8
    float acc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    float regA[4];
    float regB[4];
    
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_base = tile * TILE_K;
        
        // Simple cooperative loading
        int linear_id = tidm * 4 + tidn;
        
        // Load A tile
        for (int load = 0; load < 4; load++) {
            int load_id = linear_id + load * 16;
            int a_row = load_id / TILE_K;
            int a_col = load_id % TILE_K;
            int global_a_row = gidm * TILE_SIZE + a_row;
            int global_a_col = k_base + a_col;
            
            if (global_a_row < M && global_a_col < K && a_row < TILE_SIZE) {
                As[a_row * (TILE_K + LDS_PADDING) + a_col] = A[global_a_row * K + global_a_col];
            } else if (a_row < TILE_SIZE) {
                As[a_row * (TILE_K + LDS_PADDING) + a_col] = 0.0f;
            }
        }
        
        // Load B tile
        for (int load = 0; load < 4; load++) {
            int load_id = linear_id + load * 16;
            int b_row = load_id / TILE_SIZE;
            int b_col = load_id % TILE_SIZE;
            int global_b_row = k_base + b_row;
            int global_b_col = gidn * TILE_SIZE + b_col;
            
            if (global_b_row < K && global_b_col < N && b_row < TILE_K) {
                Bs[b_row * (TILE_SIZE + LDS_PADDING) + b_col] = B[global_b_row * N + global_b_col];
            } else if (b_row < TILE_K) {
                Bs[b_row * (TILE_SIZE + LDS_PADDING) + b_col] = 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute
        for (int k = 0; k < TILE_K; k++) {
            // Load to registers
            for (int w = 0; w < 4; w++) {
                int row_idx = tidm * 4 + w;
                int col_idx = tidn * 4 + w;
                if (row_idx < TILE_SIZE) {
                    regA[w] = As[row_idx * (TILE_K + LDS_PADDING) + k];
                }
                if (col_idx < TILE_SIZE) {
                    regB[w] = Bs[k * (TILE_SIZE + LDS_PADDING) + col_idx];
                }
            }
            
            // MAD
            for (int wm = 0; wm < 4; wm++) {
                for (int wn = 0; wn < 4; wn++) {
                    acc[wm][wn] = mad(regA[wm], regB[wn], acc[wm][wn]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    const int base_m = gidm * TILE_SIZE + tidm * 4;
    const int base_n = gidn * TILE_SIZE + tidn * 4;
    
    for (int wm = 0; wm < 4; wm++) {
        for (int wn = 0; wn < 4; wn++) {
            if (base_m + wm < M && base_n + wn < N) {
                C[(base_m + wm) * N + base_n + wn] = acc[wm][wn];
            }
        }
    }
}
"""
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    gpu = None
    for dev in devices:
        if dev.type == cl.device_type.GPU:
            gpu = dev
            break
    
    context = cl.Context([gpu])
    queue = cl.CommandQueue(context)
    
    build_options = "-cl-mad-enable -cl-fast-relaxed-math -cl-std=CL1.1"
    
    print("Compiling Clover-compatible version...")
    try:
        program = cl.Program(context, clover_kernel).build(options=build_options)
        print("✅ Compilation successful!")
        
        kernel = program.gemm_register_tiled_clover
        
        # Test
        M = N = K = 256
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        
        mf = cl.mem_flags
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
        
        kernel.set_args(np.int32(M), np.int32(N), np.int32(K), A_buf, B_buf, C_buf)
        
        local_size = (8, 8)  # 8x8 threads (simplified from 4x4)
        global_size = (
            ((M + 31) // 32) * 8,
            ((N + 31) // 32) * 8
        )
        
        print(f"Work sizes: local={local_size}, global={global_size}")
        
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        event.wait()
        
        cl.enqueue_copy(queue, C, C_buf).wait()
        
        # Verify
        C_ref = A @ B
        max_error = np.max(np.abs(C - C_ref))
        print(f"Max error: {max_error:.6f}")
        
        if max_error < 0.1:
            print("✅ Clover-compatible version works!")
            
            # Benchmark
            import time
            iterations = 10
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
                event.wait()
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times) * 1000
            ops = 2 * M * N * K
            gflops = ops / avg_time / 1e6
            
            print(f"Performance: {gflops:.2f} GFLOPS @ {M}x{N}x{K}")
            return True
        else:
            print(f"❌ Large error: {max_error}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n🔬 REGISTER_TILED Kernel Diagnosis")
    print("Task 1.1.4: Fix REGISTER_TILED for Clover\n")
    
    result1 = test_original_register_tiled()
    result2 = test_clover_compatible_version()
    
    print("\n" + "="*70)
    print(" DIAGNOSIS SUMMARY")
    print("="*70)
    print(f"Original REGISTER_TILED: {'✅ WORKS' if result1 else '❌ FAILS'}")
    print(f"Clover-Compatible Version: {'✅ WORKS' if result2 else '❌ FAILS'}")
    
    if result2:
        print("\n✅ Solution found: Remove restrict, simplify WPT, use 8x8 workgroups")
        print("   Next step: Integrate into gemm_float4_clover.cl or new file")
    else:
        print("\n⚠️  Need further investigation")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
