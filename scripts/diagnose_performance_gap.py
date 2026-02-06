#!/usr/bin/env python3
"""
Diagnose Performance Gap: Standalone vs Engine Integration
Compare kernel execution parameters and identify differences
"""

import sys
sys.path.insert(0, '/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580')

import pyopencl as cl
import numpy as np
from pathlib import Path
from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine, KernelType

def test_standalone():
    """Test gemm_float4_small the same way test_float4_clover.py does it"""
    print("="*70)
    print(" STANDALONE TEST (Original Method)")
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
        return None
    
    context = cl.Context([gpu])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Load kernel
    kernel_path = Path("src/opencl/kernels/gemm_float4_clover.cl")
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    # Build with float4_small configuration
    build_options = "-DTILE_SIZE=8 -DTILE_K=8 -cl-mad-enable -cl-fast-relaxed-math -cl-std=CL1.1"
    program = cl.Program(context, kernel_code).build(options=build_options)
    kernel = program.gemm_float4_small
    
    # Test matrix
    M = N = K = 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    # Create buffers
    mf = cl.mem_flags
    A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
    
    # Set work sizes (8x8 local, match test script)
    local_size = (8, 8)
    global_size = (
        ((M + 7) // 8) * 8,
        ((N + 7) // 8) * 8
    )
    
    print(f"   Matrix Size: {M}×{N}×{K}")
    print(f"   Local Size: {local_size}")
    print(f"   Global Size: {global_size}")
    print(f"   Work Groups: {global_size[0]//local_size[0]} × {global_size[1]//local_size[1]} = {(global_size[0]//local_size[0]) * (global_size[1]//local_size[1])}")
    
    # Set arguments
    kernel.set_args(
        np.int32(M), np.int32(N), np.int32(K),
        np.float32(1.0),
        A_buf, B_buf,
        np.float32(0.0),
        C_buf
    )
    
    # Execute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    event.wait()
    
    # Get result
    cl.enqueue_copy(queue, C, C_buf).wait()
    
    # Calculate performance
    kernel_time_ns = event.profile.end - event.profile.start
    kernel_time_ms = kernel_time_ns / 1e6
    ops = 2 * M * N * K
    gflops = ops / kernel_time_ms / 1e6
    
    print(f"   Kernel Time: {kernel_time_ms:.3f} ms")
    print(f"   GFLOPS: {gflops:.2f}")
    
    # Verify correctness
    C_ref = A @ B
    max_error = np.max(np.abs(C - C_ref))
    print(f"   Max Error: {max_error:.6f}")
    print(f"   Correct: {'✅' if max_error < 0.01 else '❌'}\n")
    
    return {
        'method': 'standalone',
        'gflops': gflops,
        'time_ms': kernel_time_ms,
        'local_size': local_size,
        'global_size': global_size,
        'error': max_error
    }

def test_engine():
    """Test using OptimizedKernelEngine"""
    print("="*70)
    print(" ENGINE TEST (Integrated Method)")
    print("="*70)
    
    engine = OptimizedKernelEngine(enable_profiling=True)
    
    M = N = K = 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    print(f"   Matrix Size: {M}×{N}×{K}")
    
    # Force FLOAT4_SMALL kernel
    kernel_type = KernelType.GEMM_FLOAT4_SMALL
    print(f"   Kernel Type: {kernel_type.name}")
    
    # Get work sizes from engine
    global_size, local_size = engine._get_optimal_work_size(kernel_type, M, N)
    print(f"   Local Size: {local_size}")
    print(f"   Global Size: {global_size}")
    print(f"   Work Groups: {global_size[0]//local_size[0]} × {global_size[1]//local_size[1]} = {(global_size[0]//local_size[0]) * (global_size[1]//local_size[1])}")
    
    # Execute
    result = engine.gemm(A, B, kernel_type=kernel_type)
    
    print(f"   Kernel Time: {result.kernel_metrics.exec_time_ms:.3f} ms")
    print(f"   GFLOPS: {result.kernel_metrics.gflops:.2f}")
    
    # Verify correctness
    C_ref = A @ B
    max_error = np.max(np.abs(result.result - C_ref))
    print(f"   Max Error: {max_error:.6f}")
    print(f"   Correct: {'✅' if max_error < 0.01 else '❌'}\n")
    
    return {
        'method': 'engine',
        'gflops': result.kernel_metrics.gflops,
        'time_ms': result.kernel_metrics.exec_time_ms,
        'local_size': local_size,
        'global_size': global_size,
        'error': max_error
    }

def main():
    print("\n🔬 PERFORMANCE GAP DIAGNOSIS")
    print("Comparing Standalone vs Engine Integration\n")
    
    standalone_result = test_standalone()
    engine_result = test_engine()
    
    if standalone_result and engine_result:
        print("="*70)
        print(" COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<20} {'Standalone':<20} {'Engine':<20} {'Difference':<15}")
        print("-"*70)
        
        gflops_diff = standalone_result['gflops'] - engine_result['gflops']
        gflops_pct = (gflops_diff / standalone_result['gflops']) * 100
        print(f"{'GFLOPS':<20} {standalone_result['gflops']:>18.2f}  {engine_result['gflops']:>18.2f}  {gflops_pct:>13.1f}%")
        
        time_diff = engine_result['time_ms'] - standalone_result['time_ms']
        time_pct = (time_diff / standalone_result['time_ms']) * 100
        print(f"{'Time (ms)':<20} {standalone_result['time_ms']:>18.3f}  {engine_result['time_ms']:>18.3f}  {time_pct:>13.1f}%")
        
        print(f"\n{'Local Size':<20} {str(standalone_result['local_size']):<20} {str(engine_result['local_size']):<20}")
        print(f"{'Global Size':<20} {str(standalone_result['global_size']):<20} {str(engine_result['global_size']):<20}")
        
        if standalone_result['local_size'] != engine_result['local_size']:
            print("\n⚠️  LOCAL SIZE MISMATCH")
        if standalone_result['global_size'] != engine_result['global_size']:
            print("⚠️  GLOBAL SIZE MISMATCH")
        
        if abs(gflops_pct) > 5:
            print(f"\n⚠️  Performance gap: {abs(gflops_pct):.1f}% difference!")
            print("    Possible causes:")
            print("    - Work size configuration mismatch")
            print("    - Buffer pool overhead")
            print("    - Memory manager overhead")
            print("    - Build options not matching")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
