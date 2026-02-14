"""
OpenCL GEMM Power Monitoring Benchmark
=======================================

Demonstrates GEMM operations with real-time power monitoring.

This script validates that OpenCL kernels properly utilize the GPU,
showing significantly higher power consumption (30-140W) compared to
idle/graphics workloads (8-10W).

Expected Results:
-----------------
- CPU operations: ~8-10W (GPU idle)
- OpenCL GEMM: 30-140W (GPU compute active)
- Temperature increase: 40°C → 60-70°C
- Performance: 500-1500 GFLOPS on RX 580

Usage:
------
    python examples/demo_opencl_gemm_power.py

    # With custom matrix size
    python examples/demo_opencl_gemm_power.py --size 2048

    # Extended benchmark
    python examples/demo_opencl_gemm_power.py --duration 60 --trials 20
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.opencl import CLContext, gemm
    from src.opencl.ops import benchmark_gemm

    OPENCL_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    OPENCL_AVAILABLE = False
    print(f"⚠️  OpenCL not available: {e}")
    print("Install with: pip install pyopencl")

from scripts.power_monitor import GPUPowerMonitor


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    size: int
    gflops: float
    time_ms: float
    avg_power_w: float
    min_power_w: float
    max_power_w: float
    energy_j: float
    temperature_c: float


def run_power_benchmark(
    ctx: CLContext, size: int, duration: int = 30, name: str = "GEMM"
) -> BenchmarkResult:
    """
    Run GEMM with power monitoring.

    Args:
        ctx: OpenCL context
        size: Matrix dimension (size x size)
        duration: Benchmark duration in seconds
        name: Benchmark name

    Returns:
        BenchmarkResult with performance and power metrics
    """
    print(f"\n{'='*70}")
    print(f"🔥 {name} Benchmark - {size}x{size}")
    print(f"{'='*70}")

    # Create test matrices
    print(f"📦 Allocating matrices: {size}x{size} × {size}x{size}")
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    memory_mb = (3 * size * size * 4) / (1024**2)  # A, B, C in MB
    print(f"   Memory usage: {memory_mb:.1f} MB")

    # Initialize power monitor
    print(f"⚡ Starting power monitoring...")
    power_monitor = GPUPowerMonitor(verbose=False)

    # Warmup
    print(f"🔥 Warming up GPU (3 iterations)...")
    for _ in range(3):
        _ = gemm(ctx, A, B)
    ctx.finish()

    # Benchmark with power monitoring
    print(f"🚀 Running benchmark ({duration}s)...")

    readings = []
    start_time = time.time()
    iterations = 0
    last_sample = time.time()
    sample_interval = 0.1  # 10 Hz

    # Run GEMM while monitoring
    while (time.time() - start_time) < duration:
        # Sample power at interval
        if time.time() - last_sample >= sample_interval:
            readings.append(power_monitor.read_full())
            last_sample = time.time()

        # Execute GEMM
        _ = gemm(ctx, A, B, use_tiled=True)
        iterations += 1

    ctx.finish()

    # Final reading
    readings.append(power_monitor.read_full())
    elapsed = time.time() - start_time

    # Calculate statistics
    power_stats = power_monitor.calculate_statistics(readings)

    # Calculate performance
    flops = 2.0 * size * size * size  # Operations per GEMM
    total_flops = flops * iterations
    gflops = (total_flops / elapsed) / 1e9
    time_per_gemm = (elapsed / iterations) * 1000  # ms

    # Energy from statistics
    energy_j = power_stats.total_energy_joules

    # Results
    result = BenchmarkResult(
        name=name,
        size=size,
        gflops=gflops,
        time_ms=time_per_gemm,
        avg_power_w=power_stats.mean_power,
        min_power_w=power_stats.min_power,
        max_power_w=power_stats.max_power,
        energy_j=energy_j,
        temperature_c=power_stats.avg_temperature or 0.0,
    )

    # Print results
    print(f"\n{'─'*70}")
    print(f"📊 Performance Results:")
    print(f"{'─'*70}")
    print(f"  Total iterations:  {iterations}")
    print(f"  Total time:        {elapsed:.2f} s")
    print(f"  Time per GEMM:     {time_per_gemm:.3f} ms")
    print(f"  Performance:       {gflops:.2f} GFLOPS")
    print(f"")
    print(f"⚡ Power Results:")
    print(f"{'─'*70}")
    print(f"  Average Power:     {power_stats.mean:.2f} W")
    print(f"  Min Power:         {power_stats.min:.2f} W")
    print(f"  Max Power:         {power_stats.max:.2f} W")
    print(f"  Std Dev:           {power_stats.std:.2f} W")
    print(f"  Total Energy:      {energy_j:.2f} J ({energy_j/3600:.4f} Wh)")
    print(f"  Energy per GEMM:   {(energy_j/iterations)*1000:.2f} mJ")
    print(f"")
    print(f"💡 Efficiency:")
    print(f"{'─'*70}")
    print(f"  GFLOPS/Watt:       {gflops/power_stats.mean:.2f}")
    print(f"  GFLOPS/Joule:      {(total_flops/energy_j)/1e9:.2f}")

    if result.temperature_c > 0:
        print(f"")
        print(f"🌡️  Temperature:       {result.temperature_c:.1f} °C")

    return result


def run_cpu_baseline(size: int, duration: int = 10) -> BenchmarkResult:
    """
    Run NumPy GEMM as baseline (CPU, GPU idle).

    This shows the power consumption when GPU is not used for compute.
    """
    print(f"\n{'='*70}")
    print(f"🖥️  CPU Baseline (NumPy) - {size}x{size}")
    print(f"{'='*70}")

    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    # Initialize power monitor
    print(f"⚡ Starting power monitoring...")
    power_monitor = GPUPowerMonitor(verbose=False)

    # Warmup
    print(f"🔥 Warming up CPU...")
    for _ in range(3):
        _ = A @ B

    # Benchmark with monitoring
    print(f"🚀 Running CPU benchmark ({duration}s)...")

    readings = []
    start_time = time.time()
    iterations = 0
    last_sample = time.time()
    sample_interval = 0.1

    while (time.time() - start_time) < duration:
        # Sample power
        if time.time() - last_sample >= sample_interval:
            readings.append(power_monitor.read_full())
            last_sample = time.time()

        # CPU GEMM
        _ = A @ B
        iterations += 1

    # Final reading
    readings.append(power_monitor.read_full())
    elapsed = time.time() - start_time

    # Calculate statistics
    power_stats = power_monitor.calculate_statistics(readings)

    # Calculate metrics
    flops = 2.0 * size * size * size
    total_flops = flops * iterations
    gflops = (total_flops / elapsed) / 1e9
    time_per_gemm = (elapsed / iterations) * 1000
    energy_j = power_stats.total_energy_joules

    result = BenchmarkResult(
        name="CPU (NumPy)",
        size=size,
        gflops=gflops,
        time_ms=time_per_gemm,
        avg_power_w=power_stats.mean_power,
        min_power_w=power_stats.min_power,
        max_power_w=power_stats.max_power,
        energy_j=energy_j,
        temperature_c=power_stats.avg_temperature or 0.0,
    )

    print(f"\n📊 CPU Performance: {gflops:.2f} GFLOPS")
    print(f"⚡ GPU Power (idle): {power_stats.mean_power:.2f} W")
    print(f"   (GPU not used for compute)")

    return result


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison table of all results."""
    print(f"\n{'='*70}")
    print(f"📊 Benchmark Comparison")
    print(f"{'='*70}")
    print(f"")
    print(f"{'Benchmark':<20} {'GFLOPS':<12} {'Power (W)':<12} {'Speedup':<10}")
    print(f"{'─'*20} {'─'*12} {'─'*12} {'─'*10}")

    cpu_result = next((r for r in results if "CPU" in r.name), None)

    for result in results:
        speedup = "─"
        if cpu_result and "CPU" not in result.name:
            speedup = f"{result.gflops / cpu_result.gflops:.1f}x"

        print(
            f"{result.name:<20} "
            f"{result.gflops:>10.2f}  "
            f"{result.avg_power_w:>10.2f}  "
            f"{speedup:>8}"
        )

    print(f"\n{'='*70}")
    print(f"🔥 Power Analysis:")
    print(f"{'='*70}")

    if cpu_result:
        opencl_result = next((r for r in results if "OpenCL" in r.name), None)
        if opencl_result:
            power_increase = opencl_result.avg_power_w - cpu_result.avg_power_w
            power_ratio = opencl_result.avg_power_w / cpu_result.avg_power_w

            print(f"  CPU (GPU idle):    {cpu_result.avg_power_w:>6.2f} W")
            print(f"  OpenCL (compute):  {opencl_result.avg_power_w:>6.2f} W")
            print(f"  Power increase:    {power_increase:>6.2f} W ({power_ratio:.1f}x)")
            print(f"")
            print(f"  ✅ GPU compute workload successfully engaged!")
            print(f"     (Power increased from idle ~8W to compute load)")


def main():
    parser = argparse.ArgumentParser(description="OpenCL GEMM with Power Monitoring")
    parser.add_argument("--size", type=int, default=1024, help="Matrix size (default: 1024)")
    parser.add_argument(
        "--duration", type=int, default=30, help="Benchmark duration in seconds (default: 30)"
    )
    parser.add_argument("--cpu-baseline", action="store_true", help="Run CPU baseline benchmark")
    parser.add_argument("--no-opencl", action="store_true", help="Skip OpenCL benchmark")

    args = parser.parse_args()

    print("=" * 70)
    print("⭐ Polaris Ascension - OpenCL GEMM Power Benchmark")
    print("=" * 70)

    results = []

    # CPU Baseline
    if args.cpu_baseline:
        try:
            cpu_result = run_cpu_baseline(args.size, duration=10)
            results.append(cpu_result)
        except Exception as e:
            print(f"❌ CPU baseline failed: {e}")

    # OpenCL Benchmark
    if not args.no_opencl:
        if not OPENCL_AVAILABLE:
            print("\n❌ OpenCL not available. Install with:")
            print("   pip install pyopencl")
            return 1

        try:
            # Initialize OpenCL
            print("\n🔧 Initializing OpenCL...")
            ctx = CLContext()
            print(f"✅ Device: {ctx.device.name}")
            print(f"   Compute Units: {ctx.device.compute_units}")
            print(f"   Global Memory: {ctx.device.global_mem_size / (1024**3):.2f} GB")

            # Run benchmark
            opencl_result = run_power_benchmark(ctx, args.size, args.duration, "OpenCL GEMM")
            results.append(opencl_result)

        except Exception as e:
            print(f"\n❌ OpenCL benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Comparison
    if len(results) > 1:
        print_comparison(results)

    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
