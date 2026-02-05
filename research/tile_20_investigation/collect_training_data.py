"""
Phase 2: Neural Performance Predictor - Data Collection

Recolecta datos de benchmarks para entrenar modelo ML
"""

import sys
sys.path.append('/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/src')

import numpy as np
import pyopencl as cl
import json
from datetime import datetime

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Load kernels
kernels = {}

print("Loading kernels...")
with open('kernels/baseline_tile16.cl', 'r') as f:
    prg = cl.Program(ctx, f.read()).build()
    kernels['tile16'] = cl.Kernel(prg, "gemm_float4_vec")

with open('kernels/approach_2_v3_vectorized.cl', 'r') as f:
    prg = cl.Program(ctx, f.read()).build()
    kernels['tile20_vec'] = cl.Kernel(prg, "gemm_tile20_vectorized")

def benchmark(kernel_name, kernel, M, N, K, tile_size, local_x, local_y, kernel_type):
    """Benchmark configuration and return data point"""
    
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    expected = A @ B
    
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    
    global_x = ((M + tile_size - 1) // tile_size) * local_x
    global_y = ((N + tile_size - 1) // tile_size) * local_y
    
    # Kernel args based on type
    if kernel_type == 'tile16':
        args = [a_buf, b_buf, c_buf, np.int32(M), np.int32(N), np.int32(K)]
    else:  # tile20
        args = [np.int32(M), np.int32(N), np.int32(K), np.float32(1.0),
                a_buf, b_buf, np.float32(0.0), c_buf]
    
    try:
        # Warmup
        kernel(queue, (global_x, global_y), (local_x, local_y), *args)
        queue.finish()
        
        # Benchmark (3 runs)
        times = []
        for _ in range(3):
            evt = kernel(queue, (global_x, global_y), (local_x, local_y), *args)
            evt.wait()
            times.append((evt.profile.end - evt.profile.start) * 1e-9)
        
        gflops = (2.0 * M * N * K) / (np.median(times) * 1e9)
        
        # Verify
        result = np.empty_like(C)
        cl.enqueue_copy(queue, result, c_buf).wait()
        error = np.max(np.abs(result - expected))
        
        if error < 0.1:
            return {
                'kernel_name': kernel_name,
                'M': M, 'N': N, 'K': K,
                'tile_size': tile_size,
                'local_x': local_x,
                'local_y': local_y,
                'threads': local_x * local_y,
                'vectorized': 1 if 'vec' in kernel_name else 0,
                'gflops': gflops,
                'error': float(error),
                'valid': True
            }
        else:
            return None
    except:
        return None


print("\n" + "=" * 80)
print("NEURAL PREDICTOR - DATA COLLECTION")
print("=" * 80)
print()

dataset = []

# Matrix sizes to test
sizes = [256, 384, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096]

print(f"Collecting benchmark data for {len(sizes)} matrix sizes...")
print()

# Configuration 1: tile16 16×16 (baseline)
print("Config 1: tile16 16×16 (baseline)")
for size in sizes:
    data = benchmark('tile16_16x16', kernels['tile16'], size, size, size, 
                    16, 16, 16, 'tile16')
    if data:
        dataset.append(data)
        print(f"  {size:4d}×{size:4d}: {data['gflops']:6.1f} GFLOPS ✅")
    else:
        print(f"  {size:4d}×{size:4d}: FAILED ❌")

print()

# Configuration 2: tile20 10×10 (vectorized - best)
print("Config 2: tile20 10×10 (vectorized)")
for size in sizes:
    data = benchmark('tile20_10x10_vec', kernels['tile20_vec'], size, size, size,
                    20, 10, 10, 'tile20')
    if data:
        dataset.append(data)
        print(f"  {size:4d}×{size:4d}: {data['gflops']:6.1f} GFLOPS ✅")
    else:
        print(f"  {size:4d}×{size:4d}: FAILED ❌")

print()

# Additional configurations for variety
# Config 3: tile16 8×8
print("Config 3: tile16 8×8 (less threads)")
test_sizes = [512, 1024, 2048]  # Subset for speed
for size in test_sizes:
    data = benchmark('tile16_8x8', kernels['tile16'], size, size, size,
                    16, 8, 8, 'tile16')
    if data:
        dataset.append(data)
        print(f"  {size:4d}×{size:4d}: {data['gflops']:6.1f} GFLOPS ✅")

print()

# Config 4: tile16 12×12
print("Config 4: tile16 12×12")
for size in test_sizes:
    data = benchmark('tile16_12x12', kernels['tile16'], size, size, size,
                    16, 12, 12, 'tile16')
    if data:
        dataset.append(data)
        print(f"  {size:4d}×{size:4d}: {data['gflops']:6.1f} GFLOPS ✅")

print()

# Non-square matrices (asymmetric workloads)
print("Config 5: Non-square matrices")
non_square = [(512, 1024, 512), (1024, 512, 1024), (2048, 1024, 2048)]
for M, N, K in non_square:
    data_16 = benchmark(f'tile16_16x16_{M}x{N}x{K}', kernels['tile16'], M, N, K,
                       16, 16, 16, 'tile16')
    if data_16:
        dataset.append(data_16)
    
    data_20 = benchmark(f'tile20_10x10_{M}x{N}x{K}', kernels['tile20_vec'], M, N, K,
                       20, 10, 10, 'tile20')
    if data_20:
        dataset.append(data_20)
    
    print(f"  {M}×{N}×{K}: tile16={data_16['gflops'] if data_16 else 0:.1f}, "
          f"tile20={data_20['gflops'] if data_20 else 0:.1f} GFLOPS")

print()
print("=" * 80)
print(f"DATA COLLECTION COMPLETE: {len(dataset)} data points")
print("=" * 80)
print()

# Save dataset
output_file = 'neural_predictor_dataset.json'
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'date': datetime.now().isoformat(),
            'hardware': 'AMD Radeon RX 590 GME',
            'driver': 'Mesa/Clover OpenCL 1.1',
            'num_samples': len(dataset)
        },
        'data': dataset
    }, f, indent=2)

print(f"✅ Dataset saved to: {output_file}")
print()

# Quick statistics
gflops_values = [d['gflops'] for d in dataset]
print(f"Performance statistics:")
print(f"  Min:    {min(gflops_values):.1f} GFLOPS")
print(f"  Max:    {max(gflops_values):.1f} GFLOPS")
print(f"  Mean:   {np.mean(gflops_values):.1f} GFLOPS")
print(f"  Median: {np.median(gflops_values):.1f} GFLOPS")
print()

# Group by kernel
by_kernel = {}
for d in dataset:
    kname = d['kernel_name'].split('_')[0]
    if kname not in by_kernel:
        by_kernel[kname] = []
    by_kernel[kname].append(d['gflops'])

print("By kernel type:")
for kname, perfs in by_kernel.items():
    print(f"  {kname:10s}: {np.mean(perfs):6.1f} GFLOPS average (n={len(perfs)})")

print()
print("🎯 Ready for model training!")
print("=" * 80)
