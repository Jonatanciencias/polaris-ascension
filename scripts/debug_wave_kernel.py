#!/usr/bin/env python3
"""
Debug Wave-Optimized GEMM Kernel
Identifies issues causing 99% performance regression (4.8 GFLOPS vs 775 GFLOPS)
"""

import numpy as np
import pyopencl as cl
import os
import sys

def debug_wave_kernel():
    """Debug the wave-optimized kernel with proper sizing"""
    print("🔧 DEBUG TEST: Wave-Optimized GEMM Kernel")
    print("=" * 50)

    # Initialize OpenCL
    platforms = cl.get_platforms()
    if not platforms:
        print("❌ No OpenCL platforms found")
        return False

    print(f"Available platforms: {len(platforms)}")
    for i, platform in enumerate(platforms):
        print(f"[{i}] {platform}")

    # Use first platform (AMD)
    platform = platforms[0]
    devices = platform.get_devices()
    if not devices:
        print("❌ No devices found")
        return False

    device = devices[0]
    print(f"Using device: {device}")

    # Create context and queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Load kernel
    kernel_path = "/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/src/opencl/kernels/gemm_wave_optimized.cl"
    if not os.path.exists(kernel_path):
        print(f"❌ Kernel file not found: {kernel_path}")
        return False

    with open(kernel_path, 'r') as f:
        kernel_source = f.read()

    # Define kernel constants for proper workgroup size
    TILE_SIZE = 16
    WG_SIZE_X = 16
    WG_SIZE_Y = 16

    # Build program with defines
    defines = f"""
#define TILE_SIZE {TILE_SIZE}
#define WG_SIZE_X {WG_SIZE_X}
#define WG_SIZE_Y {WG_SIZE_Y}
"""
    full_source = defines + "\n" + kernel_source

    try:
        program = cl.Program(context, full_source).build()
        kernel = cl.Kernel(program, "gemm_wave_optimized")
        print("✓ Kernel loaded successfully")
    except Exception as e:
        print(f"❌ Failed to build kernel: {e}")
        return False

    # Test with minimum viable size (TILE_SIZE x TILE_SIZE)
    M = N = K = TILE_SIZE  # 16x16 matrices
    print(f"Test matrices: {M}x{K}x{N}")

    # Create test matrices
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")

    # Expected result
    expected = np.dot(A, B)

    # OpenCL buffers
    A_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

    # Set kernel arguments
    alpha, beta = 1.0, 0.0
    kernel.set_args(
        np.int32(M), np.int32(N), np.int32(K),
        np.float32(alpha), np.float32(beta),
        A_buf, B_buf, C_buf
    )

    # Calculate workgroup dimensions
    # Each workgroup processes one TILE_SIZE x TILE_SIZE output block
    num_wg_x = (M + TILE_SIZE - 1) // TILE_SIZE
    num_wg_y = (N + TILE_SIZE - 1) // TILE_SIZE

    global_size = (num_wg_x * WG_SIZE_X, num_wg_y * WG_SIZE_Y)
    local_size = (WG_SIZE_X, WG_SIZE_Y)

    print(f"Global size: {global_size}")
    print(f"Local size: {local_size}")
    print(f"Workgroups: ({num_wg_x}, {num_wg_y})")

    # Execute kernel
    try:
        print("Executing kernel...")
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        print("✓ Kernel executed successfully")
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        return False

    # Read result
    result = np.zeros_like(C)
    cl.enqueue_copy(queue, result, C_buf).wait()

    # Validate
    max_error = np.max(np.abs(result - expected))
    mean_error = np.mean(np.abs(result - expected))

    print(".6f")
    print(".6f")
    print(f"Error check: {max_error} < 1e-2 = {max_error < 1e-2}")

    if max_error < 1e-2:
        print("✅ Wave kernel produces correct results!")
        return True
    else:
        print("❌ Wave kernel produces incorrect results!")
        print(f"First few expected: {expected[:3, :3].flatten()[:5]}")
        print(f"First few actual: {result[:3, :3].flatten()[:5]}")
        return False

def benchmark_wave_kernel():
    """Benchmark the corrected wave kernel with larger matrices"""
    print("🚀 WAVE KERNEL PERFORMANCE BENCHMARK")
    print("=" * 50)

    # Initialize OpenCL
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Load kernel
    kernel_path = "/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/src/opencl/kernels/gemm_wave_optimized.cl"
    with open(kernel_path, 'r') as f:
        kernel_source = f.read()

    TILE_SIZE = 16
    WG_SIZE_X = 16
    WG_SIZE_Y = 16

    defines = f"""
#define TILE_SIZE {TILE_SIZE}
#define WG_SIZE_X {WG_SIZE_X}
#define WG_SIZE_Y {WG_SIZE_Y}
"""
    full_source = defines + "\n" + kernel_source
    program = cl.Program(context, full_source).build()
    kernel = cl.Kernel(program, "gemm_wave_optimized")

    # Test different matrix sizes
    sizes = [64, 128, 256, 512, 1024]

    print("Matrix Size | Time (ms) | GFLOPS | Status")
    print("-" * 45)

    for N in sizes:
        M = N
        K = N

        # Create test matrices
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        # OpenCL buffers
        A_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

        # Set kernel arguments
        kernel.set_args(
            np.int32(M), np.int32(N), np.int32(K),
            np.float32(1.0), np.float32(0.0),
            A_buf, B_buf, C_buf
        )

        # Calculate workgroup dimensions
        num_wg_x = (M + TILE_SIZE - 1) // TILE_SIZE
        num_wg_y = (N + TILE_SIZE - 1) // TILE_SIZE
        global_size = (num_wg_x * WG_SIZE_X, num_wg_y * WG_SIZE_Y)
        local_size = (WG_SIZE_X, WG_SIZE_Y)

        # Warmup
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

        # Benchmark
        import time
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # ms

        # Calculate GFLOPS (2*M*N*K operations for GEMM)
        operations = 2 * M * N * K
        gflops = operations / (avg_time / 1000) / 1e9

        # Verify correctness (sample a few elements)
        result = np.zeros_like(C)
        cl.enqueue_copy(queue, result, C_buf).wait()
        expected = np.dot(A, B)

        # Check a few random positions
        correct = True
        for _ in range(10):
            i, j = np.random.randint(0, min(M, 10)), np.random.randint(0, min(N, 10))
            if abs(result[i, j] - expected[i, j]) > 1e-3:
                correct = False
                break

        status = "✅" if correct else "❌"
        print("6d")

if __name__ == "__main__":
    print("🚀 WAVE KERNEL DEBUG SESSION")
    print("=" * 50)

    # Test simple kernel first
    print("\n🧪 DEBUG: Simple Kernel Test")
    print("=" * 30)

    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices()
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    simple_kernel = """
    __kernel void test_kernel(__global float* input, __global float* output) {
        int i = get_global_id(0);
        output[i] = input[i] * 2.0f;
    }
    """

    program = cl.Program(context, simple_kernel).build()
    kernel = cl.Kernel(program, "test_kernel")

    test_data = np.arange(16, dtype=np.float32)
    input_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=test_data)
    output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, test_data.nbytes)

    kernel.set_args(input_buf, output_buf)
    cl.enqueue_nd_range_kernel(queue, kernel, (16,), None).wait()

    result = np.zeros_like(test_data)
    cl.enqueue_copy(queue, result, output_buf).wait()

    expected = test_data * 2.0
    if np.allclose(result, expected):
        print("✅ Simple kernel works!")
    else:
        print("❌ Simple kernel failed!")
        exit(1)

    # Test corrected wave kernel
    print("\n🔧 TESTING WAVE KERNEL WITH CORRECTED SIZES")
    success = debug_wave_kernel()

    if success:
        print("\n✅ Wave kernel is functionally correct!")
        print("🔥 RUNNING PERFORMANCE BENCHMARK...")
        benchmark_wave_kernel()
    else:
        print("\n❌ Wave kernel still has issues")