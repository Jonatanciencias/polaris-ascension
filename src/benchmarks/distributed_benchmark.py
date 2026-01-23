"""
Distributed System Benchmarks - Session 34
==========================================

Comprehensive benchmarking suite for the distributed inference system.
Measures performance across multiple dimensions to validate optimizations.

Benchmark Categories:
--------------------
1. **Task Throughput**: Tasks processed per second
2. **Task Latency**: End-to-end task completion time (p50, p95, p99)
3. **Worker Scalability**: Performance vs number of workers
4. **Message Overhead**: Serialization and network costs
5. **Failover Recovery**: Time to recover from worker failures
6. **Memory Efficiency**: Memory usage under load
7. **Cache Efficiency**: Hit rates for various caches

Benchmark Methodology:
---------------------
- Warmup period to avoid JIT/cache bias
- Statistical analysis (mean, median, percentiles)
- Multiple iterations for reliability
- Controlled load generation
- Isolation from other system activity

Usage:
------
```python
# Run all benchmarks
results = run_all_benchmarks()
print_benchmark_report(results)

# Run specific benchmark
latency_result = benchmark_task_latency(
    coordinator=coordinator,
    num_tasks=1000,
    concurrent_tasks=10
)
print(f"P95 latency: {latency_result['p95_ms']:.2f}ms")

# Compare before/after optimization
baseline = load_baseline_results("baseline.json")
current = run_all_benchmarks()
comparison = compare_results(baseline, current)
print_comparison_report(comparison)
```

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
Session: 34/35
License: MIT
"""

import time
import statistics
import json
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    
    Attributes:
        name: Benchmark name
        duration_seconds: Total benchmark duration
        iterations: Number of iterations
        metrics: Key performance metrics
        statistics: Statistical analysis
        metadata: Additional benchmark information
    """
    name: str
    duration_seconds: float
    iterations: int
    metrics: Dict[str, float]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"\n{'=' * 70}",
            f"Benchmark: {self.name}",
            f"{'=' * 70}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Iterations: {self.iterations}",
            f"\nKey Metrics:",
        ]
        
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            else:
                lines.append(f"  {key}: {value}")
        
        if self.statistics:
            lines.append(f"\nStatistics:")
            for key, value in self.statistics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """
    Comparison between two benchmark results.
    
    Attributes:
        metric_name: Name of compared metric
        baseline_value: Baseline metric value
        current_value: Current metric value
        change_percent: Percentage change
        improved: Whether performance improved
    """
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    improved: bool
    
    def __str__(self) -> str:
        """Human-readable representation."""
        arrow = "↑" if self.improved else "↓"
        sign = "+" if self.change_percent > 0 else ""
        return (
            f"{self.metric_name}: "
            f"{self.baseline_value:.2f} → {self.current_value:.2f} "
            f"({sign}{self.change_percent:.1f}% {arrow})"
        )


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistical metrics
    """
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        'count': n,
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std_dev': statistics.stdev(values) if n > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'p50': sorted_values[int(n * 0.50)],
        'p75': sorted_values[int(n * 0.75)],
        'p90': sorted_values[int(n * 0.90)],
        'p95': sorted_values[int(n * 0.95)],
        'p99': sorted_values[int(n * 0.99)],
    }


def warmup_system(coordinator, num_warmup: int = 10):
    """
    Warm up system before benchmarking.
    
    Args:
        coordinator: Coordinator instance
        num_warmup: Number of warmup tasks
    """
    logger.info(f"Warming up system with {num_warmup} tasks...")
    
    for i in range(num_warmup):
        task_id = coordinator.submit_task({
            'operation': 'warmup',
            'index': i
        })
        # Don't wait for results - just submit
    
    # Brief pause to let warmup complete
    time.sleep(2.0)
    logger.info("Warmup complete")


# ============================================================================
# THROUGHPUT BENCHMARKS
# ============================================================================

def benchmark_task_throughput(
    coordinator,
    num_tasks: int = 1000,
    concurrent_limit: Optional[int] = None,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Benchmark task submission and completion throughput.
    
    Measures how many tasks the system can process per second.
    
    Args:
        coordinator: Coordinator instance
        num_tasks: Total number of tasks to submit
        concurrent_limit: Maximum concurrent tasks (None = unlimited)
        warmup: Number of warmup tasks
    
    Returns:
        BenchmarkResult with throughput metrics
    """
    logger.info(f"Starting throughput benchmark: {num_tasks} tasks")
    
    # Warmup
    if warmup > 0:
        warmup_system(coordinator, warmup)
    
    # Prepare tasks
    task_payloads = [
        {'operation': 'benchmark', 'task_id': i, 'data': f'data_{i}'}
        for i in range(num_tasks)
    ]
    
    # Measure submission time
    start_time = time.time()
    
    if hasattr(coordinator, 'submit_batch'):
        # Use batch submission if available (faster)
        task_ids = coordinator.submit_batch(task_payloads)
    else:
        # Fall back to individual submission
        task_ids = []
        for payload in task_payloads:
            task_id = coordinator.submit_task(payload)
            task_ids.append(task_id)
    
    submission_time = time.time() - start_time
    
    # Wait for all tasks to complete
    completion_start = time.time()
    completed = 0
    errors = 0
    
    for task_id in task_ids:
        try:
            result = coordinator.get_result(task_id, timeout=60.0)
            if result is not None:
                completed += 1
            else:
                errors += 1
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            errors += 1
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    submission_throughput = num_tasks / submission_time
    completion_throughput = completed / total_time
    
    return BenchmarkResult(
        name="Task Throughput",
        duration_seconds=total_time,
        iterations=num_tasks,
        metrics={
            'submission_throughput_tasks_per_sec': submission_throughput,
            'completion_throughput_tasks_per_sec': completion_throughput,
            'completed_tasks': completed,
            'failed_tasks': errors,
            'success_rate': completed / num_tasks,
        },
        statistics={
            'submission_time_seconds': submission_time,
            'completion_time_seconds': total_time,
            'avg_task_time_ms': (total_time / completed * 1000) if completed > 0 else 0,
        }
    )


def benchmark_burst_throughput(
    coordinator,
    burst_size: int = 100,
    num_bursts: int = 10,
    burst_interval: float = 1.0
) -> BenchmarkResult:
    """
    Benchmark throughput under burst load.
    
    Tests system's ability to handle sudden load spikes.
    
    Args:
        coordinator: Coordinator instance
        burst_size: Number of tasks per burst
        num_bursts: Number of bursts to generate
        burst_interval: Seconds between bursts
    
    Returns:
        BenchmarkResult with burst handling metrics
    """
    logger.info(f"Starting burst throughput benchmark: {num_bursts} bursts of {burst_size}")
    
    burst_times = []
    total_completed = 0
    total_failed = 0
    
    start_time = time.time()
    
    for burst_idx in range(num_bursts):
        # Generate burst
        burst_start = time.time()
        
        payloads = [
            {'burst': burst_idx, 'task': i}
            for i in range(burst_size)
        ]
        
        if hasattr(coordinator, 'submit_batch'):
            task_ids = coordinator.submit_batch(payloads)
        else:
            task_ids = [coordinator.submit_task(p) for p in payloads]
        
        # Wait for burst to complete
        burst_completed = 0
        for task_id in task_ids:
            try:
                result = coordinator.get_result(task_id, timeout=30.0)
                if result is not None:
                    burst_completed += 1
            except:
                pass
        
        burst_time = time.time() - burst_start
        burst_times.append(burst_time)
        
        total_completed += burst_completed
        total_failed += (burst_size - burst_completed)
        
        # Wait before next burst
        if burst_idx < num_bursts - 1:
            time.sleep(burst_interval)
    
    total_time = time.time() - start_time
    burst_stats = calculate_statistics(burst_times)
    
    return BenchmarkResult(
        name="Burst Throughput",
        duration_seconds=total_time,
        iterations=num_bursts * burst_size,
        metrics={
            'avg_burst_time_seconds': burst_stats['mean'],
            'p95_burst_time_seconds': burst_stats['p95'],
            'total_throughput_tasks_per_sec': total_completed / total_time,
            'success_rate': total_completed / (num_bursts * burst_size),
        },
        statistics=burst_stats
    )


# ============================================================================
# LATENCY BENCHMARKS
# ============================================================================

def benchmark_task_latency(
    coordinator,
    num_tasks: int = 1000,
    concurrent_tasks: int = 10,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Benchmark end-to-end task latency.
    
    Measures time from submission to completion with percentile analysis.
    
    Args:
        coordinator: Coordinator instance
        num_tasks: Number of tasks to measure
        concurrent_tasks: Number of concurrent tasks
        warmup: Number of warmup tasks
    
    Returns:
        BenchmarkResult with latency percentiles
    """
    logger.info(f"Starting latency benchmark: {num_tasks} tasks")
    
    if warmup > 0:
        warmup_system(coordinator, warmup)
    
    latencies = []
    start_time = time.time()
    
    # Submit and measure tasks
    for i in range(num_tasks):
        task_start = time.time()
        
        task_id = coordinator.submit_task({
            'operation': 'latency_test',
            'index': i
        })
        
        try:
            result = coordinator.get_result(task_id, timeout=30.0)
            if result is not None:
                latency = (time.time() - task_start) * 1000  # Convert to ms
                latencies.append(latency)
        except Exception as e:
            logger.warning(f"Task {task_id} timeout/error: {e}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if latencies:
        stats = calculate_statistics(latencies)
    else:
        stats = {}
    
    return BenchmarkResult(
        name="Task Latency",
        duration_seconds=total_time,
        iterations=len(latencies),
        metrics={
            'mean_latency_ms': stats.get('mean', 0),
            'median_latency_ms': stats.get('median', 0),
            'p95_latency_ms': stats.get('p95', 0),
            'p99_latency_ms': stats.get('p99', 0),
            'min_latency_ms': stats.get('min', 0),
            'max_latency_ms': stats.get('max', 0),
        },
        statistics=stats
    )


def benchmark_message_overhead(
    num_iterations: int = 1000
) -> BenchmarkResult:
    """
    Benchmark message serialization overhead.
    
    Measures time to serialize/deserialize messages.
    
    Args:
        num_iterations: Number of serialize/deserialize cycles
    
    Returns:
        BenchmarkResult with serialization metrics
    """
    logger.info(f"Starting message overhead benchmark: {num_iterations} iterations")
    
    try:
        import msgpack
    except ImportError:
        logger.error("msgpack not available - skipping benchmark")
        return BenchmarkResult(
            name="Message Overhead",
            duration_seconds=0,
            iterations=0,
            metrics={},
            statistics={}
        )
    
    # Test payloads of different sizes
    payloads = {
        'small': {'id': 123, 'data': 'test'},
        'medium': {'id': 123, 'data': 'x' * 1024},  # 1KB
        'large': {'id': 123, 'data': 'x' * (64 * 1024)},  # 64KB
    }
    
    results = {}
    
    for size_name, payload in payloads.items():
        serialize_times = []
        deserialize_times = []
        
        for _ in range(num_iterations):
            # Serialize
            start = time.perf_counter()
            packed = msgpack.packb(payload)
            serialize_time = (time.perf_counter() - start) * 1000
            serialize_times.append(serialize_time)
            
            # Deserialize
            start = time.perf_counter()
            unpacked = msgpack.unpackb(packed)
            deserialize_time = (time.perf_counter() - start) * 1000
            deserialize_times.append(deserialize_time)
        
        results[f'{size_name}_serialize_ms'] = statistics.mean(serialize_times)
        results[f'{size_name}_deserialize_ms'] = statistics.mean(deserialize_times)
        results[f'{size_name}_size_bytes'] = len(packed)
    
    return BenchmarkResult(
        name="Message Overhead",
        duration_seconds=0,  # Not time-based
        iterations=num_iterations,
        metrics=results,
        statistics={}
    )


# ============================================================================
# SCALABILITY BENCHMARKS
# ============================================================================

def benchmark_worker_scalability(
    coordinator,
    max_workers: int = 10,
    tasks_per_worker: int = 100
) -> BenchmarkResult:
    """
    Benchmark scaling with increasing worker count.
    
    Tests how well the system scales from 1 to max_workers.
    
    Args:
        coordinator: Coordinator instance
        max_workers: Maximum number of workers to test
        tasks_per_worker: Tasks to submit per worker
    
    Returns:
        BenchmarkResult with scaling metrics
    """
    logger.info(f"Starting scalability benchmark: up to {max_workers} workers")
    
    # Note: This benchmark requires ability to dynamically add/remove workers
    # For now, we measure throughput at current worker count
    
    worker_stats = coordinator.get_worker_stats()
    current_workers = worker_stats['healthy_workers']
    
    if current_workers == 0:
        logger.warning("No workers available for scalability benchmark")
        return BenchmarkResult(
            name="Worker Scalability",
            duration_seconds=0,
            iterations=0,
            metrics={'error': 'no_workers'},
            statistics={}
        )
    
    # Run throughput test
    num_tasks = current_workers * tasks_per_worker
    throughput_result = benchmark_task_throughput(
        coordinator,
        num_tasks=num_tasks,
        warmup=10
    )
    
    return BenchmarkResult(
        name="Worker Scalability",
        duration_seconds=throughput_result.duration_seconds,
        iterations=num_tasks,
        metrics={
            'num_workers': current_workers,
            'throughput_per_worker': (
                throughput_result.metrics['completion_throughput_tasks_per_sec'] / 
                current_workers
            ),
            'total_throughput': throughput_result.metrics['completion_throughput_tasks_per_sec'],
            'tasks_per_worker': tasks_per_worker,
        },
        statistics=throughput_result.statistics
    )


# ============================================================================
# MEMORY & EFFICIENCY BENCHMARKS
# ============================================================================

def benchmark_memory_efficiency(
    coordinator,
    num_tasks: int = 1000,
    task_size_kb: int = 10
) -> BenchmarkResult:
    """
    Benchmark memory efficiency under load.
    
    Measures memory usage and pool efficiency.
    
    Args:
        coordinator: Coordinator instance
        num_tasks: Number of tasks to process
        task_size_kb: Approximate size of each task payload
    
    Returns:
        BenchmarkResult with memory metrics
    """
    logger.info(f"Starting memory efficiency benchmark: {num_tasks} tasks")
    
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    except ImportError:
        memory_before = 0
    
    # Run tasks
    payload_data = 'x' * (task_size_kb * 1024)
    task_ids = []
    
    for i in range(num_tasks):
        task_id = coordinator.submit_task({
            'data': payload_data,
            'index': i
        })
        task_ids.append(task_id)
    
    # Wait for completion
    for task_id in task_ids:
        try:
            coordinator.get_result(task_id, timeout=60.0)
        except:
            pass
    
    # Measure memory after
    try:
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_delta = memory_after - memory_before
    except:
        memory_after = 0
        memory_delta = 0
    
    # Get pool stats if available
    pool_stats = {}
    if hasattr(coordinator, 'get_performance_stats'):
        perf_stats = coordinator.get_performance_stats()
        pool_stats = {
            'message_pool_hit_rate': perf_stats.get('message_pool', {}).get('hit_rate', 0),
            'connection_pool_hit_rate': perf_stats.get('connection_pool', {}).get('hit_rate', 0),
            'cache_hit_rate': perf_stats.get('cache_hit_rate', 0),
        }
    
    return BenchmarkResult(
        name="Memory Efficiency",
        duration_seconds=0,
        iterations=num_tasks,
        metrics={
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta,
            'memory_per_task_kb': (memory_delta * 1024) / num_tasks if num_tasks > 0 else 0,
            **pool_stats
        },
        statistics={}
    )


# ============================================================================
# COMPREHENSIVE BENCHMARK SUITE
# ============================================================================

def run_all_benchmarks(
    coordinator,
    quick: bool = False
) -> Dict[str, BenchmarkResult]:
    """
    Run complete benchmark suite.
    
    Args:
        coordinator: Coordinator instance
        quick: If True, run smaller/faster benchmarks
    
    Returns:
        Dictionary mapping benchmark names to results
    """
    logger.info("Starting comprehensive benchmark suite")
    
    if quick:
        params = {
            'num_tasks': 100,
            'num_iterations': 100,
            'warmup': 5,
        }
    else:
        params = {
            'num_tasks': 1000,
            'num_iterations': 1000,
            'warmup': 10,
        }
    
    results = {}
    
    # Throughput benchmarks
    logger.info("Running throughput benchmarks...")
    results['throughput'] = benchmark_task_throughput(
        coordinator,
        num_tasks=params['num_tasks'],
        warmup=params['warmup']
    )
    
    results['burst_throughput'] = benchmark_burst_throughput(
        coordinator,
        burst_size=50 if quick else 100,
        num_bursts=5 if quick else 10
    )
    
    # Latency benchmarks
    logger.info("Running latency benchmarks...")
    results['latency'] = benchmark_task_latency(
        coordinator,
        num_tasks=params['num_tasks'],
        warmup=params['warmup']
    )
    
    # Message overhead
    logger.info("Running message overhead benchmark...")
    results['message_overhead'] = benchmark_message_overhead(
        num_iterations=params['num_iterations']
    )
    
    # Scalability
    logger.info("Running scalability benchmark...")
    results['scalability'] = benchmark_worker_scalability(
        coordinator,
        tasks_per_worker=50 if quick else 100
    )
    
    # Memory efficiency
    logger.info("Running memory efficiency benchmark...")
    results['memory'] = benchmark_memory_efficiency(
        coordinator,
        num_tasks=params['num_tasks']
    )
    
    logger.info("Benchmark suite complete")
    return results


def print_benchmark_report(results: Dict[str, BenchmarkResult]):
    """
    Print comprehensive benchmark report.
    
    Args:
        results: Dictionary of benchmark results
    """
    print("\n" + "=" * 70)
    print("DISTRIBUTED SYSTEM PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for name, result in results.items():
        print(result)
    
    print("=" * 70)


def save_benchmark_results(results: Dict[str, BenchmarkResult], filename: str):
    """
    Save benchmark results to JSON file.
    
    Args:
        results: Benchmark results
        filename: Output filename
    """
    data = {
        name: result.to_dict()
        for name, result in results.items()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Benchmark results saved to {filename}")


def load_benchmark_results(filename: str) -> Dict[str, Dict]:
    """
    Load benchmark results from JSON file.
    
    Args:
        filename: Input filename
    
    Returns:
        Dictionary of benchmark results
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Benchmark results loaded from {filename}")
    return data


def compare_results(
    baseline: Dict[str, Dict],
    current: Dict[str, BenchmarkResult],
    key_metrics: Optional[List[str]] = None
) -> List[ComparisonResult]:
    """
    Compare current results against baseline.
    
    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        key_metrics: Specific metrics to compare (None = all)
    
    Returns:
        List of ComparisonResult objects
    """
    if key_metrics is None:
        key_metrics = [
            'completion_throughput_tasks_per_sec',
            'p95_latency_ms',
            'memory_delta_mb',
            'cache_hit_rate',
        ]
    
    comparisons = []
    
    for bench_name, current_result in current.items():
        if bench_name not in baseline:
            continue
        
        baseline_metrics = baseline[bench_name].get('metrics', {})
        current_metrics = current_result.metrics
        
        for metric_name in key_metrics:
            if metric_name not in baseline_metrics or metric_name not in current_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]
            
            # Calculate change
            if baseline_value != 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_percent = 0
            
            # Determine if improvement (depends on metric)
            if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
                improved = current_value < baseline_value  # Lower is better
            else:
                improved = current_value > baseline_value  # Higher is better
            
            comparison = ComparisonResult(
                metric_name=f"{bench_name}.{metric_name}",
                baseline_value=baseline_value,
                current_value=current_value,
                change_percent=change_percent,
                improved=improved
            )
            comparisons.append(comparison)
    
    return comparisons


def print_comparison_report(comparisons: List[ComparisonResult]):
    """
    Print comparison report.
    
    Args:
        comparisons: List of comparison results
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 70)
    
    improvements = [c for c in comparisons if c.improved]
    regressions = [c for c in comparisons if not c.improved]
    
    print(f"\nImprovements: {len(improvements)}")
    print(f"Regressions: {len(regressions)}")
    print()
    
    if improvements:
        print("✓ IMPROVEMENTS:")
        for comp in improvements:
            print(f"  {comp}")
    
    if regressions:
        print("\n✗ REGRESSIONS:")
        for comp in regressions:
            print(f"  {comp}")
    
    print("\n" + "=" * 70)


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Distributed System Benchmarks")
    print("=" * 70)
    print("\nThis module provides comprehensive benchmarking for distributed systems.")
    print("\nAvailable benchmarks:")
    print("  - Task throughput")
    print("  - Burst throughput")
    print("  - Task latency (with percentiles)")
    print("  - Message serialization overhead")
    print("  - Worker scalability")
    print("  - Memory efficiency")
    print("\nUsage:")
    print("  results = run_all_benchmarks(coordinator)")
    print("  print_benchmark_report(results)")
    print("  save_benchmark_results(results, 'results.json')")
    print("\n" + "=" * 70)
