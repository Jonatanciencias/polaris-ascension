#!/usr/bin/env python3
"""
Quick GPU Warm-up + Single Benchmark @ 1300

Ensures GPU is at full performance before measuring.
"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyopencl as cl

# Setup
platforms = cl.get_platforms()
ctx = cl.Context([platforms[0].get_devices(device_type=cl.device_type.GPU)[0]])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Load kernel
kernel_path = project_root / "src" / "kernels" / "gemm_tile20_production.cl"
with open(kernel_path, 'r') as f:
    source = f.read()
program = cl.Program(ctx, source).build()
kernel = program.gemm_tile20_optimized

print("🔥 Warming up GPU (forcing high performance state)...")
print("   Running 20 iterations to wake up GPU...\n")

M = N = K = 1300
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

global_size = (65 * 10, 65 * 10)  # 1300/20 = 65 tiles
local_size = (10, 10)

# Intensive warm-up
for i in range(20):
    event = kernel(
        queue, global_size, local_size,
        np.int32(M), np.int32(N), np.int32(K),
        np.float32(1.0), A_buf, B_buf, np.float32(0.0), C_buf
    )
    event.wait()
    elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
    gflops = (2.0 * M * N * K) / (elapsed_ms * 1e-3 * 1e9)
    print(f"   Warmup {i+1:2d}/20: {gflops:7.1f} GFLOPS  ({elapsed_ms:6.2f} ms)")

print(f"\n{'='*70}")
print("BENCHMARK RUNS (10× with hot GPU)")
print(f"{'='*70}\n")

times = []
for i in range(10):
    event = kernel(
        queue, global_size, local_size,
        np.int32(M), np.int32(N), np.int32(K),
        np.float32(1.0), A_buf, B_buf, np.float32(0.0), C_buf
    )
    event.wait()
    elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
    times.append(elapsed_ms)
    gflops = (2.0 * M * N * K) / (elapsed_ms * 1e-3 * 1e9)
    print(f"Run {i+1:2d}/10: {gflops:7.1f} GFLOPS  ({elapsed_ms:6.2f} ms)")

avg_time = np.mean(times)
min_time = np.min(times)
avg_gflops = (2.0 * M * N * K) / (avg_time * 1e-3 * 1e9)
peak_gflops = (2.0 * M * N * K) / (min_time * 1e-3 * 1e9)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Average: {avg_gflops:.1f} GFLOPS ({avg_time:.2f} ms)")
print(f"Peak:    {peak_gflops:.1f} GFLOPS ({min_time:.2f} ms)")
print(f"\nComparison:")
print(f"  Auto-tuner:  824.1 GFLOPS")  
print(f"  This run:    {avg_gflops:.1f} GFLOPS")
print(f"  Difference:  {avg_gflops - 824.1:+.1f} GFLOPS")

# Cleanup
A_buf.release()
B_buf.release()
C_buf.release()

print("\n✅ Done! Check sensors now:")
print("   sensors | grep -A 5 amdgpu")
