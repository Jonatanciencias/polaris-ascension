"""
Demo: Hybrid CPU/GPU Scheduler

This demo showcases the hybrid scheduler capabilities:
1. Automatic device selection based on task characteristics
2. Workload partitioning for parallel execution
3. Load balancing across available devices
4. Performance monitoring and statistics

Session 14 - Hybrid Scheduler Demo
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compute.hybrid import HybridScheduler, TaskConfig, OpType, Device


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_basic_scheduling():
    """Demo 1: Basic task scheduling."""
    print_section("Demo 1: Basic Task Scheduling")

    # Create scheduler with small transfer threshold
    scheduler = HybridScheduler(transfer_threshold=50 * 1024, enable_partitioning=True)  # 50 KB

    print(f"Scheduler configuration:")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    print(f"  Transfer Threshold: {scheduler.transfer_threshold / 1024:.1f} KB")
    print(f"  Partitioning: {'Enabled' if scheduler.enable_partitioning else 'Disabled'}")
    print()

    # Task 1: Very small (< threshold)
    task_tiny = TaskConfig(operation=OpType.ELEMENTWISE, input_shapes=[(10, 10)], dtype="float32")
    device1 = scheduler.schedule(task_tiny)
    print(f"1. Tiny task ({task_tiny.total_size / 1024:.2f} KB)")
    print(f"   Scheduled to: {device1}")
    print(f"   Reason: Below transfer threshold")
    print()

    # Task 2: Medium (> threshold, < 10 MB)
    task_medium = TaskConfig(
        operation=OpType.MATMUL, input_shapes=[(500, 500), (500, 500)], dtype="float32"
    )
    device2 = scheduler.schedule(task_medium)
    print(f"2. Medium task ({task_medium.total_size / 1024**2:.2f} MB)")
    print(f"   Scheduled to: {device2}")
    print(f"   FLOPs: {task_medium.flops / 1e9:.2f} GFLOPs")
    print()

    # Task 3: Large (> 10 MB, eligible for partitioning)
    task_large = TaskConfig(
        operation=OpType.CONV, input_shapes=[(128, 3, 224, 224), (64, 3, 7, 7)], dtype="float32"
    )
    device3 = scheduler.schedule(task_large)
    print(f"3. Large task ({task_large.total_size / 1024**2:.2f} MB)")
    print(f"   Scheduled to: {device3}")
    print()

    # Show scheduler state
    print("Scheduler state:")
    print(f"  CPU completion time: {scheduler.load_balancer.completion_times['cpu']:.4f}s")
    if torch.cuda.is_available():
        print(f"  GPU completion time: {scheduler.load_balancer.completion_times['cuda']:.4f}s")


def demo_explicit_devices():
    """Demo 2: Explicit device selection."""
    print_section("Demo 2: Explicit Device Selection")

    scheduler = HybridScheduler()

    task = TaskConfig(
        operation=OpType.MATMUL, input_shapes=[(1000, 1000), (1000, 1000)], dtype="float32"
    )

    # Explicit CPU
    device_cpu = scheduler.schedule(task, Device.CPU)
    print(f"1. Explicit CPU: {device_cpu}")

    # Explicit GPU (falls back to CPU if unavailable)
    device_gpu = scheduler.schedule(task, Device.GPU)
    print(f"2. Explicit GPU: {device_gpu}")
    if device_gpu == "cpu":
        print("   (Fell back to CPU - GPU not available)")

    # Automatic selection
    device_auto = scheduler.schedule(task, Device.AUTO)
    print(f"3. AUTO selection: {device_auto}")


def demo_execution():
    """Demo 3: Actual execution with timing."""
    print_section("Demo 3: Actual Execution with Timing")

    scheduler = HybridScheduler()

    # Create test tensors
    size = 500
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    print(f"Input tensors: {size}x{size} float32")
    print(f"Memory: {(a.numel() + b.numel()) * 4 / 1024**2:.2f} MB")
    print()

    # Define operation
    def matmul_fn(x, y):
        return torch.matmul(x, y)

    # Execute on CPU
    print("Executing on CPU...")
    start = time.time()
    result_cpu = scheduler._execute_single([a, b], matmul_fn, "cpu")
    elapsed_cpu = time.time() - start
    print(f"  Time: {elapsed_cpu*1000:.2f} ms")
    print(f"  Result shape: {result_cpu.shape}")
    print(f"  Result mean: {result_cpu.mean():.4f}")
    print()

    # Execute on GPU if available
    if torch.cuda.is_available():
        print("Executing on GPU...")
        start = time.time()
        result_gpu = scheduler._execute_single([a, b], matmul_fn, "cuda")
        elapsed_gpu = time.time() - start
        print(f"  Time: {elapsed_gpu*1000:.2f} ms")
        print(f"  Result shape: {result_gpu.shape}")
        print(f"  Speedup: {elapsed_cpu/elapsed_gpu:.2f}x")
        print()

        # Verify correctness
        result_gpu_cpu = result_gpu.cpu()
        diff = torch.abs(result_cpu - result_gpu_cpu).max()
        print(f"  Max difference: {diff:.2e}")

    # Show statistics
    stats = scheduler.get_statistics()
    print("\nExecution statistics:")
    print(f"  CPU tasks: {stats['cpu_tasks']}")
    print(f"  Total CPU time: {stats['total_cpu_time']*1000:.2f} ms")
    if torch.cuda.is_available():
        print(f"  GPU tasks: {stats['gpu_tasks']}")
        print(f"  Total GPU time: {stats['total_gpu_time']*1000:.2f} ms")


def demo_partitioning():
    """Demo 4: Workload partitioning."""
    print_section("Demo 4: Workload Partitioning")

    scheduler = HybridScheduler(enable_partitioning=True)

    # Large task eligible for partitioning
    task = TaskConfig(
        operation=OpType.MATMUL, input_shapes=[(2000, 2000), (2000, 2000)], dtype="float32"
    )

    print(f"Task size: {task.total_size / 1024**2:.2f} MB")
    print(f"Task FLOPs: {task.flops / 1e9:.2f} GFLOPs")
    print()

    # Check if should partition
    should_partition = scheduler.partitioner.should_partition(task)
    print(f"Should partition: {should_partition}")

    if should_partition:
        # Calculate partition
        batch_size = 128
        cpu_size, gpu_size = scheduler.partitioner.partition_data(task, batch_size)

        print(f"\nPartition plan:")
        print(f"  Total batch: {batch_size}")
        print(f"  CPU portion: {cpu_size} ({cpu_size/batch_size*100:.1f}%)")
        print(f"  GPU portion: {gpu_size} ({gpu_size/batch_size*100:.1f}%)")

        # Estimate times
        t_cpu = scheduler.profiler.estimate_execution_time(task, "cpu") * (cpu_size / batch_size)
        t_gpu = (
            scheduler.profiler.estimate_execution_time(task, "cuda") * (gpu_size / batch_size)
            if torch.cuda.is_available()
            else 0
        )
        t_transfer = scheduler.profiler.estimate_transfer_time(
            task.total_size * gpu_size / batch_size, "cpu", "cuda"
        )

        print(f"\nEstimated times:")
        print(f"  CPU execution: {t_cpu*1000:.2f} ms")
        if torch.cuda.is_available():
            print(f"  GPU execution: {t_gpu*1000:.2f} ms")
            print(f"  Transfer: {t_transfer*1000:.2f} ms")
            print(f"  Pipeline: {max(t_cpu, t_gpu + t_transfer)*1000:.2f} ms")
    else:
        print("  Not beneficial to partition this task")
        print(f"  Reason: Task too small or GPU unavailable")


def demo_load_balancing():
    """Demo 5: Load balancing across devices."""
    print_section("Demo 5: Load Balancing")

    scheduler = HybridScheduler()

    # Create multiple tasks
    tasks = [
        TaskConfig(OpType.MATMUL, [(1000, 1000), (1000, 1000)], "float32"),
        TaskConfig(OpType.CONV, [(32, 3, 64, 64), (64, 3, 3, 3)], "float32"),
        TaskConfig(OpType.ELEMENTWISE, [(5000, 5000)], "float32"),
        TaskConfig(OpType.MATMUL, [(500, 500), (500, 500)], "float32"),
        TaskConfig(OpType.ACTIVATION, [(10000, 1000)], "float32"),
    ]

    print(f"Scheduling {len(tasks)} tasks...\n")

    # Schedule each task
    for i, task in enumerate(tasks, 1):
        device = scheduler.schedule(task)
        size_mb = task.total_size / 1024**2

        print(f"{i}. {task.operation.name:15s} ({size_mb:6.2f} MB) → {device}")

        # Show estimated times
        t_cpu = scheduler.profiler.estimate_execution_time(task, "cpu")
        t_gpu = (
            scheduler.profiler.estimate_execution_time(task, "cuda")
            if torch.cuda.is_available()
            else 0
        )
        print(f"   Est. CPU: {t_cpu*1000:7.2f} ms", end="")
        if torch.cuda.is_available():
            print(f" | Est. GPU: {t_gpu*1000:7.2f} ms")
        else:
            print()

    # Show load balancer state
    print(f"\nLoad balancer state:")
    print(f"  CPU queue: {len(scheduler.load_balancer.pending_tasks['cpu'])} tasks")
    print(f"  CPU completion: {scheduler.load_balancer.completion_times['cpu']*1000:.2f} ms")
    if torch.cuda.is_available():
        print(f"  GPU queue: {len(scheduler.load_balancer.pending_tasks['cuda'])} tasks")
        print(f"  GPU completion: {scheduler.load_balancer.completion_times['cuda']*1000:.2f} ms")


def demo_statistics():
    """Demo 6: Performance statistics."""
    print_section("Demo 6: Performance Statistics")

    scheduler = HybridScheduler()

    # Simulate some executions
    print("Simulating workload...\n")

    # Create and execute some tasks
    for i in range(5):
        a = torch.randn(200, 200)
        b = torch.randn(200, 200)

        def matmul_fn(x, y):
            return torch.matmul(x, y)

        device = "cpu" if i % 2 == 0 else ("cuda" if torch.cuda.is_available() else "cpu")
        scheduler._execute_single([a, b], matmul_fn, device)

    # Get statistics
    stats = scheduler.get_statistics()

    print("Scheduler Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  CPU tasks: {stats['cpu_tasks']} ({stats['cpu_ratio']*100:.1f}%)")
    print(f"  GPU tasks: {stats['gpu_tasks']} ({stats['gpu_ratio']*100:.1f}%)")
    print(f"  Partitioned: {stats['partitioned_tasks']}")
    print()
    print(f"Execution times:")
    print(f"  Total CPU: {stats['total_cpu_time']*1000:.2f} ms")
    print(f"  Avg CPU: {stats['avg_cpu_time']*1000:.2f} ms")
    if stats["gpu_tasks"] > 0:
        print(f"  Total GPU: {stats['total_gpu_time']*1000:.2f} ms")
        print(f"  Avg GPU: {stats['avg_gpu_time']*1000:.2f} ms")
    print(f"  Transfer: {stats['total_transfer_time']*1000:.2f} ms")


def demo_memory_constraints():
    """Demo 7: Memory constraint handling."""
    print_section("Demo 7: Memory Constraint Handling")

    # Scheduler with limited GPU memory
    scheduler = HybridScheduler(gpu_memory_limit=500 * 1024**2)  # 500 MB limit

    print(f"GPU memory limit: 500 MB")
    print()

    # Task that fits
    task_small = TaskConfig(
        operation=OpType.MATMUL, input_shapes=[(1000, 1000), (1000, 1000)], dtype="float32"
    )
    device1 = scheduler.schedule(task_small)
    print(f"1. Small task ({task_small.total_size / 1024**2:.2f} MB) → {device1}")

    # Task that doesn't fit
    task_huge = TaskConfig(
        operation=OpType.MATMUL, input_shapes=[(10000, 10000), (10000, 10000)], dtype="float32"
    )
    device2 = scheduler.schedule(task_huge)
    print(f"2. Huge task ({task_huge.total_size / 1024**2:.2f} MB) → {device2}")
    print(f"   (Exceeds GPU memory limit, assigned to CPU)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  HYBRID CPU/GPU SCHEDULER DEMO")
    print("  Session 14 - Legacy GPU AI Platform")
    print("=" * 70)

    # Check hardware
    print(f"\nHardware Configuration:")
    print(f"  CPU: Available ({torch.get_num_threads()} threads)")
    print(f"  GPU: {'Available' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Run demos
    try:
        demo_basic_scheduling()
        demo_explicit_devices()
        demo_execution()
        demo_partitioning()
        demo_load_balancing()
        demo_statistics()
        demo_memory_constraints()

        # Final summary
        print_section("Demo Complete")
        print("✓ All demonstrations completed successfully")
        print("\nKey Features Demonstrated:")
        print("  1. Automatic task scheduling based on size and characteristics")
        print("  2. Explicit device control (CPU/GPU/AUTO)")
        print("  3. Actual execution with performance timing")
        print("  4. Intelligent workload partitioning")
        print("  5. Dynamic load balancing across devices")
        print("  6. Comprehensive performance statistics")
        print("  7. Memory-aware constraint handling")
        print("\nThe hybrid scheduler provides:")
        print("  • Transparent CPU/GPU coordination")
        print("  • Minimal transfer overhead")
        print("  • Optimal resource utilization")
        print("  • Production-ready performance monitoring")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
