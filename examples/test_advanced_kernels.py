#!/usr/bin/env python3
"""
Advanced GEMM Kernels Test Suite

Tests innovative kernels inspired by:
- Quantum mechanics (vector spaces)
- Tensor network theory
- Strassen's algorithm
- Monte Carlo methods
- Statistical mechanics

Author: Polaris Ascension Project
Date: 23 de enero de 2026
"""

import sys
import time

import numpy as np
import pyopencl as cl

sys.path.insert(0, "/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580")
from scripts.power_monitor import GPUPowerMonitor


def test_kernel(ctx, queue, kernel_name, kernel, size, A, B, C_expected, use_float4=False):
    """Test a single kernel and return performance metrics"""
    M, N, K = size, size, size

    # For float4 kernels, N must be multiple of 4
    if use_float4 and N % 4 != 0:
        print(f"  Skipping {kernel_name} (requires N % 4 == 0)")
        return None

    C_gpu = np.zeros((M, N), dtype=np.float32)

    # Upload to GPU
    a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, C_gpu.nbytes)

    # Determine work sizes
    if "float4" in kernel_name:
        global_size = (M, N // 4)
    elif "strassen" in kernel_name or "tiled_2x2" in kernel_name:
        global_size = (M // 2, N // 2)
    else:
        global_size = (M, N)

    local_size = (16, 16)

    try:
        # Warmup
        for _ in range(3):
            kernel(
                queue,
                global_size,
                local_size,
                np.int32(M),
                np.int32(N),
                np.int32(K),
                np.float32(1.0),
                np.float32(0.0),
                a_buf,
                b_buf,
                c_buf,
            ).wait()

        # Benchmark
        times = []
        for _ in range(10):
            evt = kernel(
                queue,
                global_size,
                local_size,
                np.int32(M),
                np.int32(N),
                np.int32(K),
                np.float32(1.0),
                np.float32(0.0),
                a_buf,
                b_buf,
                c_buf,
            )
            evt.wait()
            times.append(1e-9 * (evt.profile.end - evt.profile.start))

        # Download and verify
        cl.enqueue_copy(queue, C_gpu, c_buf).wait()
        error = np.abs(C_gpu - C_expected).max()

        avg_time = np.mean(times) * 1000  # ms
        flops = 2 * M * N * K
        gflops = flops / (avg_time / 1000) / 1e9

        return {
            "kernel": kernel_name,
            "gflops": gflops,
            "time_ms": avg_time,
            "error": error,
            "success": error < 1e-3,
        }

    except Exception as e:
        print(f"  Error in {kernel_name}: {e}")
        return None


def main():
    print("=" * 80)
    print("🚀 ADVANCED GEMM KERNELS - INNOVATION TEST SUITE")
    print("=" * 80)

    # Setup OpenCL
    platforms = cl.get_platforms()
    clover = [p for p in platforms if "clover" in p.name.lower()][0]
    ctx = cl.Context(clover.get_devices()[:1])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print(f"Device: {ctx.devices[0].name}")
    print(f"Max Compute Units: {ctx.devices[0].max_compute_units}")

    # Load kernels
    print("\nLoading kernels...")
    with open("src/opencl/kernels/gemm.cl", "r") as f:
        prg = cl.Program(ctx, f.read()).build()

    # Prepare test kernels
    kernels = {
        "tiled (baseline)": cl.Kernel(prg, "gemm_tiled"),
        "tiled_2x2": cl.Kernel(prg, "gemm_tiled_2x2"),
        "vectorized_float4": cl.Kernel(prg, "gemm_vectorized_float4"),
        "tensor_inspired": cl.Kernel(prg, "gemm_tensor_inspired"),
        "strassen_inspired": cl.Kernel(prg, "gemm_strassen_inspired"),
    }

    monitor = GPUPowerMonitor(verbose=False)

    # Test sizes
    test_sizes = [512, 1024]

    print("\n" + "=" * 80)
    print("TEST 1: Performance Comparison")
    print("=" * 80)

    all_results = []

    for size in test_sizes:
        print(f"\n--- Testing {size}×{size} ---")

        # Generate test data
        M = N = K = size
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_expected = A @ B

        for kernel_name, kernel in kernels.items():
            use_float4 = "float4" in kernel_name
            result = test_kernel(
                ctx, queue, kernel_name, kernel, size, A, B, C_expected, use_float4
            )

            if result:
                status = "✅" if result["success"] else "❌"
                print(
                    f"  {result['kernel']:20s}: {result['gflops']:6.1f} GFLOPS | "
                    f"{result['time_ms']:7.2f} ms | Error: {result['error']:.2e} {status}"
                )
                all_results.append({**result, "size": size})

    print("\n" + "=" * 80)
    print("TEST 2: Detailed Vectorization Analysis (1024×1024)")
    print("=" * 80)

    size = 1024
    M = N = K = size
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)

    test_kernels = ["tiled_2x2", "vectorized_float4", "tensor_inspired"]

    for kernel_name in test_kernels:
        if kernel_name not in kernels:
            continue

        kernel = kernels[kernel_name]
        c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, A.nbytes)

        # Determine sizes
        if "float4" in kernel_name:
            global_size = (M, N // 4)
        elif kernel_name == "tiled_2x2":
            global_size = (M // 2, N // 2)
        else:
            global_size = (M, N)

        # Warmup
        for _ in range(3):
            kernel(
                queue,
                global_size,
                (16, 16),
                np.int32(M),
                np.int32(N),
                np.int32(K),
                np.float32(1.0),
                np.float32(0.0),
                a_buf,
                b_buf,
                c_buf,
            ).wait()

        print(f"\nTesting {kernel_name} (15 seconds)...")
        readings = []
        start = time.time()
        iterations = 0
        last_sample = time.time()

        try:
            while time.time() - start < 15:
                if time.time() - last_sample >= 0.1:
                    readings.append(monitor.read_full())
                    last_sample = time.time()

                evt = kernel(
                    queue,
                    global_size,
                    (16, 16),
                    np.int32(M),
                    np.int32(N),
                    np.int32(K),
                    np.float32(1.0),
                    np.float32(0.0),
                    a_buf,
                    b_buf,
                    c_buf,
                )
                evt.wait()
                iterations += 1

            readings.append(monitor.read_full())
            elapsed = time.time() - start

            stats = monitor.calculate_statistics(readings)
            flops = 2 * M * N * K
            gflops = (flops * iterations / elapsed) / 1e9

            print(f"  Performance:  {gflops:6.1f} GFLOPS")
            print(f"  Avg Power:    {stats.mean_power:6.2f} W")
            print(f"  Temperature:  {stats.avg_temperature:5.1f} °C")
            print(f"  Efficiency:   {gflops/stats.mean_power:5.2f} GFLOPS/W")
            print(f"  Iterations:   {iterations}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("TEST 3: Innovation Impact Analysis")
    print("=" * 80)

    # Group results by size
    for size in test_sizes:
        size_results = [r for r in all_results if r["size"] == size and r["success"]]
        if not size_results:
            continue

        print(f"\n{size}×{size} Results:")
        baseline = [r for r in size_results if "baseline" in r["kernel"]]

        if baseline:
            baseline_gflops = baseline[0]["gflops"]

            for result in sorted(size_results, key=lambda x: x["gflops"], reverse=True):
                speedup = result["gflops"] / baseline_gflops
                innovation = ""

                if "float4" in result["kernel"]:
                    innovation = "📐 Vector Math"
                elif "tensor" in result["kernel"]:
                    innovation = "🎯 Tensor Networks"
                elif "strassen" in result["kernel"]:
                    innovation = "🧮 Strassen O(n^2.807)"
                elif "tiled_2x2" in result["kernel"]:
                    innovation = "🔲 2×2 Blocking"

                print(
                    f"  {result['kernel']:20s}: {result['gflops']:6.1f} GFLOPS "
                    f"({speedup:4.2f}x) {innovation}"
                )

    print("\n" + "=" * 80)
    print("SUMMARY: Innovation Report")
    print("=" * 80)

    innovations = [
        ("Vectorization (float4)", "Quantum-inspired parallel processing"),
        ("Tensor Networks", "Optimal contraction ordering"),
        ("Strassen Algorithm", "Reduced multiplications O(n^2.807)"),
        ("Kahan Summation", "Improved numerical stability"),
        ("2×2 Register Blocking", "Enhanced data reuse"),
    ]

    print("\n🧪 Mathematical Innovations Applied:")
    for name, desc in innovations:
        print(f"  • {name:25s} → {desc}")

    print("\n🎯 Expected Performance Gains:")
    print("  • Vectorization:    +100-150 GFLOPS (2-3x bandwidth)")
    print("  • Tensor Inspired:  +20-30% (better locality)")
    print("  • Strassen:         Asymptotic advantage (large N)")
    print("  • Combined:         400-600 GFLOPS target")

    print("\n📊 Scientific Principles:")
    print("  • Vector Spaces (Quantum Mechanics)")
    print("  • Tensor Contraction (Many-Body Physics)")
    print("  • Free Energy Minimization (Statistical Mechanics)")
    print("  • Monte Carlo Sampling (Stochastic Methods)")
    print("  • Algorithmic Complexity Theory (Computer Science)")

    print("\n" + "=" * 80)
    print("✅ ADVANCED KERNEL TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
