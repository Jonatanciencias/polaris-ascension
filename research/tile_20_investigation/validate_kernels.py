"""
Simple validation test: Check if tile16 and tile20 kernels work correctly
"""

import sys
sys.path.append('/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/src')

import numpy as np
import pyopencl as cl
import time

# Setup OpenCL
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Load kernels
with open('kernels/baseline_tile16.cl', 'r') as f:
    tile16_code = f.read()

with open('kernels/approach_2_v3_vectorized.cl', 'r') as f:
    tile20_code = f.read()

prg16 = cl.Program(ctx, tile16_code).build()
prg20 = cl.Program(ctx, tile20_code).build()

def test_kernel(name, kernel, tile_size, local_x, local_y, M, N, K, kernel_args_func):
    """Test a single kernel configuration"""
    
    # Create test matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    expected = A @ B
    
    # Upload
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    
    # Work sizes
    global_x = ((M + tile_size - 1) // tile_size) * local_x
    global_y = ((N + tile_size - 1) // tile_size) * local_y
    
    print(f"\nTesting: {name}")
    print(f"  Matrix: {M}×{N}×{K}")
    print(f"  Tile: {tile_size}, Local: {local_x}×{local_y}, Global: {global_x}×{global_y}")
    
    # Get kernel args
    args = kernel_args_func(a_buf, b_buf, c_buf, M, N, K)
    
    try:
        # Warmup
        kernel(queue, (global_x, global_y), (local_x, local_y), *args)
        queue.finish()
        
        # Benchmark
        times = []
        for _ in range(5):
            evt = kernel(queue, (global_x, global_y), (local_x, local_y), *args)
            evt.wait()
            times.append((evt.profile.end - evt.profile.start) * 1e-9)
        
        avg_time = np.median(times)
        gflops = (2.0 * M * N * K) / (avg_time * 1e9)
        
        # Verify
        result = np.empty_like(C)
        cl.enqueue_copy(queue, result, c_buf).wait()
        
        error = np.max(np.abs(result - expected))
        
        if error > 0.1:
            print(f"  ❌ FAILED - error={error:.2e}")
            return False, 0.0
        else:
            print(f"  ✅ PASSED - {gflops:.1f} GFLOPS, error={error:.2e}")
            return True, gflops
            
    except Exception as e:
        print(f"  ❌ EXCEPTION - {e}")
        return False, 0.0


print("=" * 70)
print("PHASE 1 VALIDATION: Kernel Correctness Check")
print("=" * 70)

M = N = K = 1024

# Test 1: tile16 with 16×16 threads (production config)
print("\n" + "=" * 70)
print("TEST: Baseline tile16 (16×16 threads)")
print("=" * 70)

def tile16_args(a_buf, b_buf, c_buf, M, N, K):
    return [a_buf, b_buf, c_buf, np.int32(M), np.int32(N), np.int32(K)]

success, perf = test_kernel(
    "tile16 16×16",
    cl.Kernel(prg16, "gemm_float4_vec"),
    tile_size=16,
    local_x=16,
    local_y=16,
    M=M, N=N, K=K,
    kernel_args_func=tile16_args
)

print(f"\nResult: {'✅ PASS' if success else '❌ FAIL'} - {perf:.1f} GFLOPS")

# Test 2: tile16 with 8×8 threads (SA found this)
print("\n" + "=" * 70)
print("TEST: tile16 with 8×8 threads (SA discovered)")
print("=" * 70)

success2, perf2 = test_kernel(
    "tile16 8×8",
    cl.Kernel(prg16, "gemm_float4_vec"),
    tile_size=16,
    local_x=8,
    local_y=8,
    M=M, N=N, K=K,
    kernel_args_func=tile16_args
)

print(f"\nResult: {'✅ PASS' if success2 else '❌ FAIL'} - {perf2:.1f} GFLOPS")

# Test 3: tile20 with 10×10 threads (known best)
print("\n" + "=" * 70)
print("TEST: tile20 vectorized (10×10 threads) - KNOWN BEST")
print("=" * 70)

def tile20_args(a_buf, b_buf, c_buf, M, N, K):
    return [np.int32(M), np.int32(N), np.int32(K), np.float32(1.0),
            a_buf, b_buf, np.float32(0.0), c_buf]

success3, perf3 = test_kernel(
    "tile20 10×10",
    cl.Kernel(prg20, "gemm_tile20_vectorized"),
    tile_size=20,
    local_x=10,
    local_y=10,
    M=M, N=N, K=K,
    kernel_args_func=tile20_args
)

print(f"\nResult: {'✅ PASS' if success3 else '❌ FAIL'} - {perf3:.1f} GFLOPS")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"tile16 16×16 (production):  {perf:.1f} GFLOPS {'✅' if success else '❌'}")
print(f"tile16 8×8 (SA discovery):  {perf2:.1f} GFLOPS {'✅' if success2 else '❌'}")
print(f"tile20 10×10 (known best):  {perf3:.1f} GFLOPS {'✅' if success3 else '❌'}")
print()

if success2 and perf2 > perf:
    improvement = ((perf2 - perf) / perf) * 100
    print(f"🎉 SA discovered +{improvement:.1f}% improvement with 8×8 threads!")
elif success3 and perf3 > max(perf, perf2):
    print(f"🏆 tile20 10×10 is still the best: {perf3:.1f} GFLOPS")

print("=" * 70)
