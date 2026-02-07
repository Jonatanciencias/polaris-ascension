#!/usr/bin/env python3
"""
Benchmark Script

Performance benchmarking for GPU operations.
"""

import sys
import os
import time
import argparse
from typing import Dict, List

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verify_hardware import detect_amd_gpu


class Profiler:
    """Minimal local profiler for script-level benchmarks."""

    def __init__(self):
        self._active: Dict[str, float] = {}
        self._samples: Dict[str, List[float]] = {}

    def start(self, name: str) -> None:
        self._active[name] = time.perf_counter()

    def end(self, name: str) -> None:
        start = self._active.pop(name, None)
        if start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._samples.setdefault(name, []).append(elapsed_ms)

    def print_summary(self) -> None:
        if not self._samples:
            print("No profiling samples collected.")
            return

        print(f"{'Metric':<25} {'Calls':<8} {'Avg (ms)':<12} {'Total (ms)':<12}")
        print("-" * 60)
        for name, values in sorted(self._samples.items()):
            total = sum(values)
            avg = total / len(values)
            print(f"{name:<25} {len(values):<8} {avg:<12.2f} {total:<12.2f}")


def benchmark_memory_bandwidth(profiler, size_mb=100):
    """Benchmark memory operations"""
    print(f"\nBenchmarking memory bandwidth ({size_mb} MB)...")
    
    profiler.start("memory_allocation")
    # Simulate allocation
    data = bytearray(size_mb * 1024 * 1024)
    profiler.end("memory_allocation")
    
    profiler.start("memory_write")
    # Write pattern
    for i in range(0, len(data), 4096):
        data[i:i+4096] = b'\xff' * 4096
    profiler.end("memory_write")
    
    profiler.start("memory_read")
    # Read pattern
    total = 0
    for i in range(0, len(data), 4096):
        total += sum(data[i:i+4096])
    profiler.end("memory_read")
    
    print(f"✅ Memory benchmark complete")


def benchmark_compute():
    """Benchmark compute operations"""
    print("\n⚠️  Compute benchmarks coming soon")
    print("   Will include: matrix multiply, convolution, etc.")


def print_memory_stats() -> None:
    """Print host memory statistics."""
    vm = psutil.virtual_memory()
    print(f"System RAM total:     {vm.total / (1024**3):.1f} GB")
    print(f"System RAM available: {vm.available / (1024**3):.1f} GB")
    print(f"System RAM used:      {vm.percent:.1f}%")


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description='RX 580 AI Framework Benchmarks')
    parser.add_argument('--memory', action='store_true', help='Run memory benchmarks')
    parser.add_argument('--compute', action='store_true', help='Run compute benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    args = parser.parse_args()
    
    # If no args, run all
    if not any([args.memory, args.compute, args.all]):
        args.all = True
    
    print("=" * 60)
    print("Radeon RX 580 Benchmark Suite")
    print("=" * 60)
    
    # Initialize
    gpu = detect_amd_gpu()
    if gpu is None:
        print("⚠️  No AMD GPU detected through OpenCL.")
        print("   Running CPU/memory-only benchmark.")
    else:
        print(f"✅ GPU detected: {gpu.name}")
        print(f"   OpenCL runtime: {gpu.opencl_version}")
        print(f"   VRAM: {gpu.vram_gb:.1f} GB")

    profiler = Profiler()
    
    # Run benchmarks
    if args.all or args.memory:
        benchmark_memory_bandwidth(profiler)
    
    if args.all or args.compute:
        benchmark_compute()
    
    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    profiler.print_summary()
    print_memory_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
