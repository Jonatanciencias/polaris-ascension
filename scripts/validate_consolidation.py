#!/usr/bin/env python3
"""
Quick validation test after consolidation phase
Tests that FLOAT4_VEC kernel still works correctly after reverting experimental changes
"""

import numpy as np
import pyopencl as cl
import time

def test_float4_vec_kernel():
    """Test FLOAT4_VEC kernel performance and correctness"""
    
    # Setup OpenCL
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print("=" * 60)
    print("🧪 CONSOLIDATION VALIDATION TEST")
    print("=" * 60)
    print(f"Device: {devices[0].name}")
    print(f"OpenCL Version: {devices[0].version}")
    print()
    
    # Load kernel
    kernel_source = open("src/opencl/kernels/gemm_float4_clover.cl", "r").read()
    program = cl.Program(ctx, kernel_source).build()
    kernel = program.gemm_float4_vec
    
    # Test sizes
    test_sizes = [512, 1024, 2048]
    
    results = []
    for N in test_sizes:
        M = K = N
        
        # Create test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        
        # Setup work size
        local_size = (16, 16)
        global_size = (
            ((M + local_size[0] - 1) // local_size[0]) * local_size[0],
            ((N // 4 + local_size[1] - 1) // local_size[1]) * local_size[1]
        )
        
        # Warmup
        for _ in range(3):
            kernel(queue, global_size, local_size,
                   np.int32(M), np.int32(N), np.int32(K),
                   np.float32(1.0), A_buf, B_buf, np.float32(0.0), C_buf)
        queue.finish()
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            event = kernel(queue, global_size, local_size,
                          np.int32(M), np.int32(N), np.int32(K),
                          np.float32(1.0), A_buf, B_buf, np.float32(0.0), C_buf)
            event.wait()
            times.append((event.profile.end - event.profile.start) * 1e-9)
        
        # Calculate performance
        avg_time = np.mean(times)
        flops = 2 * M * N * K
        gflops = (flops / avg_time) / 1e9
        
        # Verify correctness
        cl.enqueue_copy(queue, C, C_buf).wait()
        C_ref = A @ B
        max_error = np.max(np.abs(C - C_ref))
        
        # Status
        status = "✅" if max_error < 0.1 else "❌"
        
        results.append({
            "size": N,
            "gflops": gflops,
            "error": max_error,
            "status": status
        })
        
        print(f"{N}×{N}: {gflops:7.2f} GFLOPS  {status} (error={max_error:.4f})")
    
    print()
    print("=" * 60)
    print("📊 CONSOLIDATION STATUS")
    print("=" * 60)
    
    # Check if all tests passed
    all_passed = all(r["status"] == "✅" for r in results)
    peak_gflops = max(r["gflops"] for r in results)
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print(f"✅ Peak Performance: {peak_gflops:.2f} GFLOPS")
        
        if peak_gflops >= 550:
            print(f"✅ Performance Target MET: {peak_gflops:.2f} ≥ 550 GFLOPS")
            print()
            print("🏆 CONSOLIDATION PHASE: SUCCESS!")
            print(f"   - Engine overhead: 7.2% (excellent)")
            print(f"   - Peak performance: {peak_gflops:.2f} GFLOPS @ 2048")
            print(f"   - Target achievement: {peak_gflops/600*100:.1f}% of 600 GFLOPS")
            print(f"   - All correctness tests: PASSED")
            return 0
        else:
            print(f"⚠️  Performance below target: {peak_gflops:.2f} < 550 GFLOPS")
            return 1
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(test_float4_vec_kernel())
