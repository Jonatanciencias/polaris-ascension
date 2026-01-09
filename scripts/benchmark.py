#!/usr/bin/env python3
"""
Benchmark Script

Performance benchmarking for GPU operations.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gpu import GPUManager
from src.core.memory import MemoryManager
from src.core.profiler import Profiler


def benchmark_memory_bandwidth(memory_manager, profiler, size_mb=100):
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
    gpu_manager = GPUManager()
    if not gpu_manager.initialize():
        print("Failed to initialize GPU")
        return 1
    
    memory_manager = MemoryManager()
    profiler = Profiler()
    
    # Run benchmarks
    if args.all or args.memory:
        benchmark_memory_bandwidth(memory_manager, profiler)
    
    if args.all or args.compute:
        benchmark_compute()
    
    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    profiler.print_summary()
    memory_manager.print_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
