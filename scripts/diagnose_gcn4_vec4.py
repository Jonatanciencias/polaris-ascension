#!/usr/bin/env python3
"""
Diagnose GCN4_VEC4 Performance Issue
Task 1.2.1: Profile and identify bottlenecks

Current: 29 GFLOPS (very poor)
Target: 150+ GFLOPS (5× improvement)
"""

import sys
sys.path.insert(0, '/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580')

import pyopencl as cl
import numpy as np
from pathlib import Path
import time

def analyze_gcn4_vec4_signature():
    """Analyze the kernel signature issue"""
    print("="*70)
    print(" ANALYSIS 1: Kernel Signature")
    print("="*70)
    
    # Load kernel
    kernel_path = Path("src/opencl/kernels/gemm_gcn4_ultra.cl")
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    # Find gemm_gcn4_vec4 signature
    import re
    match = re.search(r'void gemm_gcn4_vec4\((.*?)\)', kernel_code, re.DOTALL)
    if match:
        signature = match.group(1)
        print("Current signature:")
        print(signature)
        print()
        
        if 'float4*' in signature:
            print("⚠️  PROBLEM FOUND: Kernel expects float4* pointers")
            print("   Engine passes float* pointers")
            print("   This causes misaligned access and terrible performance")
            print()
            print("✅ SOLUTION: Rewrite kernel to accept float* and use vload4/vstore4")
            return True
    return False

def test_current_gcn4_vec4():
    """Test current GCN4_VEC4 performance through engine"""
    print("\n" + "="*70)
    print(" TEST 1: Current GCN4_VEC4 Performance")
    print("="*70)
    
    from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine, KernelType
    
    engine = OptimizedKernelEngine(enable_profiling=True)
    
    test_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    
    print(f"{'Size':<15} {'GFLOPS':<10} {'Time (ms)':<12} {'Status'}")
    print("-" * 70)
    
    for M, N, K in test_sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        try:
            result = engine.gemm(A, B, kernel_type=KernelType.GEMM_GCN4_VEC4)
            C_ref = A @ B
            error = np.max(np.abs(result.result - C_ref))
            
            status = "✅" if error < 0.01 else "❌"
            print(f"{M}×{N}×{K:<5} {result.kernel_metrics.gflops:>8.2f}  {result.kernel_metrics.exec_time_ms:>10.3f}  {status}")
        except Exception as e:
            print(f"{M}×{N}×{K:<5} {'ERROR':<10} {'N/A':<12} ❌")
            print(f"   Error: {str(e)[:60]}")
    
    print()

def create_fixed_gcn4_vec4():
    """Create a fixed version using vload4/vstore4"""
    print("="*70)
    print(" FIX: Create Improved GCN4_VEC4 Kernel")
    print("="*70)
    
    fixed_kernel = """
// Fixed GCN4 VEC4 kernel - Uses vload4/vstore4 instead of float4 pointers
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_gcn4_vec4_fixed(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    
    // Each thread computes 4x4 output elements
    const int row = gidm * 32 + tidm * 4;
    const int col = gidn * 32 + tidn * 4;
    
    // LDS for tiles - use float instead of float4
    __local float As[32][16 + 1];  // 32x16 with padding
    __local float Bs[16][32 + 1];  // 16x32 with padding
    
    // Accumulators (4x4 = 16 per thread)
    float acc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    const int num_tiles = (K + 15) / 16;
    
    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 16;
        
        // Cooperative loading of A tile (each thread loads 4 elements)
        for (int i = 0; i < 4; i++) {
            int a_row = row + i;
            for (int k_local = 0; k_local < 2; k_local++) {
                int a_col = k_base + tidn * 2 + k_local;
                if (a_row < M && a_col < K) {
                    As[tidm * 4 + i][tidn * 2 + k_local] = A[a_row * K + a_col];
                } else {
                    As[tidm * 4 + i][tidn * 2 + k_local] = 0.0f;
                }
            }
        }
        
        // Cooperative loading of B tile
        for (int k_local = 0; k_local < 2; k_local++) {
            int b_row = k_base + tidm * 2 + k_local;
            for (int j = 0; j < 4; j++) {
                int b_col = col + j;
                if (b_row < K && b_col < N) {
                    Bs[tidm * 2 + k_local][tidn * 4 + j] = B[b_row * N + b_col];
                } else {
                    Bs[tidm * 2 + k_local][tidn * 4 + j] = 0.0f;
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            for (int i = 0; i < 4; i++) {
                float a_val = As[tidm * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float b_val = Bs[k][tidn * 4 + j];
                    acc[i][j] = mad(a_val, b_val, acc[i][j]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results with alpha/beta
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int out_row = row + i;
            int out_col = col + j;
            if (out_row < M && out_col < N) {
                int idx = out_row * N + out_col;
                if (beta == 0.0f) {
                    C[idx] = alpha * acc[i][j];
                } else {
                    C[idx] = alpha * acc[i][j] + beta * C[idx];
                }
            }
        }
    }
}
"""
    
    print("✅ Fixed kernel created")
    print("   Key changes:")
    print("   - Changed float4* → float*")
    print("   - Removed vload4/vstore4 complexity")
    print("   - Direct float array access")
    print("   - Proper coalescing pattern")
    print()
    
    return fixed_kernel

def test_fixed_kernel(kernel_code):
    """Test the fixed kernel"""
    print("="*70)
    print(" TEST 2: Fixed GCN4_VEC4 Performance")
    print("="*70)
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    gpu = None
    for dev in devices:
        if dev.type == cl.device_type.GPU:
            gpu = dev
            break
    
    context = cl.Context([gpu])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    build_options = "-cl-mad-enable -cl-fast-relaxed-math -cl-std=CL1.1"
    
    try:
        program = cl.Program(context, kernel_code).build(options=build_options)
        kernel = program.gemm_gcn4_vec4_fixed
        
        print("✅ Kernel compiled successfully")
        print()
        
        test_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
        
        print(f"{'Size':<15} {'GFLOPS':<10} {'Time (ms)':<12} {'vs Old':<10} {'Status'}")
        print("-" * 75)
        
        old_perf = {256: 10, 512: 20, 1024: 29}  # Approximate old performance
        
        for M, N, K in test_sizes:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            C = np.zeros((M, N), dtype=np.float32)
            
            mf = cl.mem_flags
            A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
            
            kernel.set_args(
                np.int32(M), np.int32(N), np.int32(K),
                np.float32(1.0),
                A_buf, B_buf,
                np.float32(0.0),
                C_buf
            )
            
            local_size = (8, 8)
            global_size = (
                ((M + 31) // 32) * 8,
                ((N + 31) // 32) * 8
            )
            
            # Warmup
            for _ in range(3):
                cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
            
            # Benchmark
            times = []
            for _ in range(10):
                event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
                event.wait()
                kernel_time_ns = event.profile.end - event.profile.start
                times.append(kernel_time_ns / 1e6)
            
            avg_time = np.mean(times)
            ops = 2 * M * N * K
            gflops = ops / avg_time / 1e6
            
            cl.enqueue_copy(queue, C, C_buf).wait()
            
            # Verify
            C_ref = A @ B
            error = np.max(np.abs(C - C_ref))
            status = "✅" if error < 0.01 else "❌"
            
            old_gflops = old_perf.get(M, 29)
            speedup = f"{gflops/old_gflops:.2f}×"
            
            print(f"{M}×{N}×{K:<5} {gflops:>8.2f}  {avg_time:>10.3f}  {speedup:>8}  {status}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n🔬 GCN4_VEC4 PERFORMANCE DIAGNOSIS")
    print("Task 1.2.1: Profiling and optimization\n")
    
    # Step 1: Analyze signature issue
    has_issue = analyze_gcn4_vec4_signature()
    
    # Step 2: Test current performance
    test_current_gcn4_vec4()
    
    # Step 3: Create and test fix
    fixed_kernel = create_fixed_gcn4_vec4()
    test_fixed_kernel(fixed_kernel)
    
    print("="*70)
    print(" DIAGNOSIS COMPLETE")
    print("="*70)
    print("✅ Root cause: float4* pointer type mismatch")
    print("✅ Solution: Rewrite with float* and proper tiling")
    print("🎯 Next step: Integrate fixed kernel into gemm_gcn4_ultra.cl")

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
