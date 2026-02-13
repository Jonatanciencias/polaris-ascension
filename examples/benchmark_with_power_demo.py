#!/usr/bin/env python3
"""
Power Profiling Demo
====================

Demonstrates inference benchmarking with real power measurements.

This example shows how to:
1. Monitor GPU power consumption in real-time
2. Profile model inference with power metrics
3. Calculate efficiency metrics (FPS/Watt, energy per inference)
4. Compare different optimization techniques

Usage:
    python examples/benchmark_with_power_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.profiling.power_profiler import PowerProfiler, BenchmarkWithPower
from scripts.power_monitor import GPUPowerMonitor


# Simple test model
class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_test_data(num_samples=500, batch_size=32):
    """Create synthetic test data."""
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def demo1_basic_power_monitor():
    """Demo 1: Basic power monitoring."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Power Monitoring")
    print("=" * 70)

    try:
        monitor = GPUPowerMonitor(verbose=True)

        print("\n📊 Current GPU State:")
        reading = monitor.read_full()
        print(f"  Power:       {reading.power_watts:.2f} W")
        if reading.voltage:
            print(f"  Voltage:     {reading.voltage:.3f} V")
        if reading.temperature:
            print(f"  Temperature: {reading.temperature:.1f} °C")

        print("\n⏱️  Monitoring for 10 seconds...")
        readings = monitor.monitor_continuous(duration=10, interval=0.5)

        stats = monitor.calculate_statistics(readings)
        print(f"\n{stats}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have AMD GPU drivers installed.")


def demo2_inference_with_power():
    """Demo 2: Inference profiling with power measurement."""
    print("\n" + "=" * 70)
    print("DEMO 2: Inference Profiling with Power")
    print("=" * 70)

    try:
        # Create model and data
        print("\n📦 Preparing model and data...")
        model = SimpleCNN()
        data_loader = create_test_data(num_samples=500, batch_size=32)

        # CPU inference
        print("\n🖥️  Profiling on CPU...")
        model_cpu = model.to("cpu")
        profiler = PowerProfiler(verbose=False)

        results_cpu = profiler.profile_model(model_cpu, data_loader, duration=20, warmup_seconds=3)

        print(results_cpu)

        # GPU inference (if available)
        if torch.cuda.is_available():
            print("\n🎮 Profiling on GPU...")
            model_gpu = model.to("cuda")

            # Need to recreate data_loader for GPU
            data_loader_gpu = create_test_data(num_samples=500, batch_size=32)

            results_gpu = profiler.profile_model(
                model_gpu, data_loader_gpu, duration=20, warmup_seconds=3
            )

            print(results_gpu)

            # Compare
            print("\n📊 CPU vs GPU Comparison:")
            print(f"  FPS:          CPU: {results_cpu.fps:.1f}  |  GPU: {results_gpu.fps:.1f}")
            print(
                f"  Power:        CPU: {results_cpu.avg_power_watts:.1f}W  |  GPU: {results_gpu.avg_power_watts:.1f}W"
            )
            print(
                f"  FPS/Watt:     CPU: {results_cpu.fps_per_watt:.2f}  |  GPU: {results_gpu.fps_per_watt:.2f}"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def demo3_simple_api():
    """Demo 3: Simple API usage."""
    print("\n" + "=" * 70)
    print("DEMO 3: Simple API - BenchmarkWithPower")
    print("=" * 70)

    try:
        # Create model and data
        model = SimpleCNN()
        data_loader = create_test_data(num_samples=300, batch_size=32)

        # Use simple API
        benchmark = BenchmarkWithPower(model, data_loader)
        results = benchmark.run(duration=15, warmup_seconds=2)

        print(results)

        # Access specific metrics
        print("\n📊 Key Metrics:")
        print(f"  Throughput:        {results.fps:.1f} FPS")
        print(f"  Power Consumption: {results.avg_power_watts:.1f} W")
        print(f"  Efficiency:        {results.fps_per_watt:.2f} FPS/W")
        print(f"  Energy/Image:      {results.energy_per_inference_joules*1000:.2f} mJ")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo4_compare_batch_sizes():
    """Demo 4: Compare different batch sizes."""
    print("\n" + "=" * 70)
    print("DEMO 4: Batch Size Impact on Power Efficiency")
    print("=" * 70)

    try:
        model = SimpleCNN()

        batch_sizes = [8, 16, 32, 64]
        results = {}

        for batch_size in batch_sizes:
            print(f"\n🔄 Testing batch size: {batch_size}")

            data_loader = create_test_data(num_samples=500, batch_size=batch_size)
            benchmark = BenchmarkWithPower(model, data_loader, verbose=False)

            result = benchmark.run(duration=15, warmup_seconds=2)
            results[batch_size] = result

            print(
                f"  FPS: {result.fps:.1f}  |  Power: {result.avg_power_watts:.1f}W  |  FPS/W: {result.fps_per_watt:.2f}"
            )

        # Summary table
        print("\n📊 Batch Size Comparison:")
        print(f"{'Batch':>6} | {'FPS':>8} | {'Power':>8} | {'FPS/W':>8} | {'Energy/Img':>12}")
        print("-" * 60)

        for bs, res in results.items():
            print(
                f"{bs:>6} | {res.fps:>8.1f} | {res.avg_power_watts:>7.1f}W | "
                f"{res.fps_per_watt:>8.2f} | {res.energy_per_inference_joules*1000:>10.2f} mJ"
            )

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("   Power Profiling Demonstration")
    print("   AMD Radeon RX 580 - Real Power Measurements")
    print("=" * 70)

    demos = [
        ("Basic Power Monitoring", demo1_basic_power_monitor),
        ("Inference with Power", demo2_inference_with_power),
        ("Simple API", demo3_simple_api),
        ("Batch Size Comparison", demo4_compare_batch_sizes),
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos)+1}. Run all demos")

    try:
        choice = input("\nSelect demo (1-{}): ".format(len(demos) + 1))
        choice = int(choice)

        if 1 <= choice <= len(demos):
            demos[choice - 1][1]()
        elif choice == len(demos) + 1:
            for name, demo_fn in demos:
                demo_fn()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
