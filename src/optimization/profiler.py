"""
Performance Profiling Module - Session 34
==========================================

Professional performance analysis tools for the Legacy GPU AI platform.

This module provides comprehensive profiling capabilities including:
- CPU profiling (function-level timing)
- Memory profiling (allocation tracking)
- Latency measurements (microsecond precision)
- Throughput analysis (operations per second)
- Bottleneck identification

Features:
---------
1. **Decorator-based profiling**: Add @profile_cpu to any function
2. **Context managers**: Use with statements for fine-grained tracking
3. **Low overhead**: Minimal impact on production performance
4. **Detailed reports**: Human-readable and machine-parseable output
5. **Integration-ready**: Works with pytest, benchmarks, and CI/CD

Usage Examples:
--------------
```python
# CPU profiling
@profile_cpu(name="critical_function")
def process_data(data):
    return expensive_operation(data)

# Memory tracking
with track_memory("model_loading"):
    model = load_large_model()

# Latency measurement
with measure_latency() as timer:
    result = api_call()
print(f"Latency: {timer.elapsed_ms:.2f}ms")

# Throughput testing
throughput = measure_throughput(
    func=process_tasks,
    iterations=1000,
    warmup=10
)
print(f"Throughput: {throughput:.1f} ops/sec")
```

Author: Radeon RX 580 AI Framework Team
Date: Enero 22, 2026
Session: 34/35
License: MIT
"""

import time
import functools
import threading
import tracemalloc
import statistics
from typing import Callable, Any, Optional, Dict, List, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json
import logging

# Try to import optional profiling libraries
try:
    import cProfile
    import pstats
    from pstats import SortKey
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProfilingResult:
    """
    Container for profiling results.
    
    Attributes:
        name: Identifier for the profiled operation
        duration_ms: Execution time in milliseconds
        memory_delta_mb: Memory usage change in megabytes
        cpu_percent: CPU utilization percentage (if available)
        timestamp: When the profiling started
        metadata: Additional profiling information
    """
    name: str
    duration_ms: float
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'duration_ms': round(self.duration_ms, 3),
            'memory_delta_mb': round(self.memory_delta_mb, 3),
            'cpu_percent': round(self.cpu_percent, 1),
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ProfilingResult(name='{self.name}', "
            f"duration={self.duration_ms:.2f}ms, "
            f"memory={self.memory_delta_mb:+.1f}MB, "
            f"cpu={self.cpu_percent:.0f}%)"
        )


@dataclass
class LatencyStats:
    """
    Statistical analysis of latency measurements.
    
    Attributes:
        name: Identifier for the measured operation
        count: Number of measurements
        mean_ms: Average latency in milliseconds
        median_ms: Median latency (p50)
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        min_ms: Minimum observed latency
        max_ms: Maximum observed latency
        std_dev_ms: Standard deviation
    """
    name: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'count': self.count,
            'mean_ms': round(self.mean_ms, 3),
            'median_ms': round(self.median_ms, 3),
            'p95_ms': round(self.p95_ms, 3),
            'p99_ms': round(self.p99_ms, 3),
            'min_ms': round(self.min_ms, 3),
            'max_ms': round(self.max_ms, 3),
            'std_dev_ms': round(self.std_dev_ms, 3)
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"\nLatency Statistics for '{self.name}':\n"
            f"  Count:      {self.count}\n"
            f"  Mean:       {self.mean_ms:.2f}ms\n"
            f"  Median:     {self.median_ms:.2f}ms\n"
            f"  P95:        {self.p95_ms:.2f}ms\n"
            f"  P99:        {self.p99_ms:.2f}ms\n"
            f"  Min/Max:    {self.min_ms:.2f}ms / {self.max_ms:.2f}ms\n"
            f"  Std Dev:    {self.std_dev_ms:.2f}ms"
        )


# ============================================================================
# GLOBAL STATE (Thread-safe)
# ============================================================================

class _ProfilerState:
    """
    Global profiler state with thread-safe access.
    
    Stores profiling results and configuration across the application.
    """
    
    def __init__(self):
        self._results: List[ProfilingResult] = []
        self._latencies: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._enabled = True
    
    def add_result(self, result: ProfilingResult):
        """Thread-safe addition of profiling result."""
        if not self._enabled:
            return
        
        with self._lock:
            self._results.append(result)
    
    def add_latency(self, name: str, latency_ms: float):
        """Thread-safe addition of latency measurement."""
        if not self._enabled:
            return
        
        with self._lock:
            if name not in self._latencies:
                self._latencies[name] = []
            self._latencies[name].append(latency_ms)
    
    def get_results(self) -> List[ProfilingResult]:
        """Get all profiling results."""
        with self._lock:
            return self._results.copy()
    
    def get_latencies(self, name: str) -> List[float]:
        """Get latency measurements for a specific operation."""
        with self._lock:
            return self._latencies.get(name, []).copy()
    
    def clear(self):
        """Clear all profiling data."""
        with self._lock:
            self._results.clear()
            self._latencies.clear()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling (for production)."""
        self._enabled = False


# Global profiler state
_profiler_state = _ProfilerState()


# ============================================================================
# DECORATORS - Function-level profiling
# ============================================================================

def profile_cpu(name: Optional[str] = None, save_stats: bool = False):
    """
    Decorator for CPU profiling of functions.
    
    Measures execution time and optionally CPU usage. Results are stored
    in the global profiler state for later analysis.
    
    Args:
        name: Custom name for the profiling result (defaults to function name)
        save_stats: If True, save detailed cProfile statistics to file
    
    Returns:
        Decorated function that profiles execution
    
    Example:
        ```python
        @profile_cpu(name="data_processing")
        def process_large_dataset(data):
            # Complex processing
            return transformed_data
        
        # After execution, view results:
        results = get_profiling_results()
        for r in results:
            print(r)
        ```
    
    Performance Impact:
        - Minimal overhead (~1-5% for simple timing)
        - Higher overhead with save_stats=True (~10-20%)
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get initial CPU usage if available
            cpu_before = 0.0
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    cpu_before = process.cpu_percent()
                except:
                    pass
            
            # Optional detailed profiling
            profiler = None
            if save_stats and CPROFILE_AVAILABLE:
                profiler = cProfile.Profile()
                profiler.enable()
            
            # Time execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Stop detailed profiling
            if profiler:
                profiler.disable()
                stats_file = f"profile_{op_name}_{int(time.time())}.prof"
                profiler.dump_stats(stats_file)
                logger.info(f"Saved profile stats to {stats_file}")
            
            # Get final CPU usage
            cpu_after = 0.0
            if PSUTIL_AVAILABLE:
                try:
                    cpu_after = process.cpu_percent()
                except:
                    pass
            
            # Store result
            profiling_result = ProfilingResult(
                name=op_name,
                duration_ms=duration,
                cpu_percent=(cpu_before + cpu_after) / 2,
                metadata={
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            _profiler_state.add_result(profiling_result)
            
            logger.debug(f"Profiled {op_name}: {duration:.2f}ms")
            
            return result
        
        return wrapper
    return decorator


def profile_memory(name: Optional[str] = None):
    """
    Decorator for memory profiling of functions.
    
    Tracks memory allocation and deallocation during function execution
    using Python's tracemalloc.
    
    Args:
        name: Custom name for the profiling result
    
    Returns:
        Decorated function that profiles memory usage
    
    Example:
        ```python
        @profile_memory(name="model_loading")
        def load_model(path):
            model = torch.load(path)  # Large allocation
            return model
        
        # Check memory impact:
        results = get_profiling_results()
        for r in results:
            if 'model_loading' in r.name:
                print(f"Memory delta: {r.memory_delta_mb:.1f}MB")
        ```
    
    Note:
        - Requires Python 3.4+
        - tracemalloc has ~10% overhead
        - Best used for profiling, not production
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start memory tracking
            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()
            
            # Time execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            
            # Measure memory delta
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            # Calculate total memory change
            total_diff = sum(stat.size_diff for stat in stats)
            memory_mb = total_diff / (1024 * 1024)  # Convert to MB
            
            tracemalloc.stop()
            
            # Store result
            profiling_result = ProfilingResult(
                name=op_name,
                duration_ms=duration,
                memory_delta_mb=memory_mb,
                metadata={
                    'allocations': len([s for s in stats if s.size_diff > 0]),
                    'deallocations': len([s for s in stats if s.size_diff < 0])
                }
            )
            _profiler_state.add_result(profiling_result)
            
            logger.debug(f"Profiled {op_name}: {memory_mb:+.1f}MB, {duration:.2f}ms")
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# CONTEXT MANAGERS - Fine-grained profiling
# ============================================================================

@contextmanager
def measure_latency(name: str = "operation"):
    """
    Context manager for precise latency measurement.
    
    Provides microsecond-precision timing for critical code sections.
    Results are stored for statistical analysis.
    
    Args:
        name: Identifier for the measured operation
    
    Yields:
        Timer object with .elapsed_ms property
    
    Example:
        ```python
        with measure_latency("api_request") as timer:
            response = requests.get(url)
        
        print(f"Request took {timer.elapsed_ms:.2f}ms")
        
        # Later, get statistics:
        stats = get_latency_stats("api_request")
        print(f"P95 latency: {stats.p95_ms:.2f}ms")
        ```
    
    Performance Impact:
        - Near-zero overhead (~0.01ms)
        - Uses time.perf_counter() for precision
    """
    class Timer:
        def __init__(self):
            self.start_time = time.perf_counter()
            self.elapsed_ms = 0.0
    
    timer = Timer()
    
    try:
        yield timer
    finally:
        timer.elapsed_ms = (time.perf_counter() - timer.start_time) * 1000
        _profiler_state.add_latency(name, timer.elapsed_ms)


@contextmanager
def track_memory(name: str = "operation"):
    """
    Context manager for memory usage tracking.
    
    Monitors memory allocation within a code block using tracemalloc.
    
    Args:
        name: Identifier for the tracked operation
    
    Yields:
        MemoryTracker object with .delta_mb property
    
    Example:
        ```python
        with track_memory("data_loading") as tracker:
            data = load_large_dataset()
        
        print(f"Loaded data used {tracker.delta_mb:.1f}MB")
        ```
    
    Note:
        - tracemalloc must be available
        - Has ~10% performance overhead
    """
    class MemoryTracker:
        def __init__(self):
            self.delta_mb = 0.0
    
    tracker = MemoryTracker()
    
    # Start tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    start_time = time.perf_counter()
    
    try:
        yield tracker
    finally:
        # Calculate delta
        duration = (time.perf_counter() - start_time) * 1000
        snapshot_after = tracemalloc.take_snapshot()
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_diff = sum(stat.size_diff for stat in stats)
        tracker.delta_mb = total_diff / (1024 * 1024)
        
        tracemalloc.stop()
        
        # Store result
        result = ProfilingResult(
            name=name,
            duration_ms=duration,
            memory_delta_mb=tracker.delta_mb
        )
        _profiler_state.add_result(result)


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def measure_throughput(
    func: Callable,
    iterations: int = 1000,
    warmup: int = 10,
    *args,
    **kwargs
) -> float:
    """
    Measure throughput (operations per second) for a function.
    
    Runs the function multiple times and calculates average ops/sec.
    Includes warmup period to account for JIT compilation and caching.
    
    Args:
        func: Function to measure
        iterations: Number of times to run (excluding warmup)
        warmup: Number of warmup iterations
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Throughput in operations per second
    
    Example:
        ```python
        def process_item(item):
            return expensive_computation(item)
        
        throughput = measure_throughput(
            process_item,
            iterations=1000,
            warmup=10,
            item=test_data
        )
        
        print(f"Throughput: {throughput:.1f} ops/sec")
        ```
    
    Note:
        - Use warmup to avoid measurement bias
        - Higher iterations = more accurate but slower
    """
    # Warmup phase
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Measurement phase
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        func(*args, **kwargs)
    
    elapsed = time.perf_counter() - start_time
    throughput = iterations / elapsed
    
    logger.info(
        f"Throughput measurement: {throughput:.1f} ops/sec "
        f"({iterations} iterations in {elapsed:.2f}s)"
    )
    
    return throughput


def measure_batch_latency(
    func: Callable,
    iterations: int = 100,
    *args,
    **kwargs
) -> LatencyStats:
    """
    Measure latency distribution for a function over multiple runs.
    
    Collects detailed latency statistics including percentiles.
    
    Args:
        func: Function to measure
        iterations: Number of measurements
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        LatencyStats object with statistical analysis
    
    Example:
        ```python
        stats = measure_batch_latency(
            coordinator.submit_task,
            iterations=1000,
            payload={'test': 'data'}
        )
        
        print(stats)
        # Output:
        # Latency Statistics for 'submit_task':
        #   Mean:    2.45ms
        #   Median:  2.31ms
        #   P95:     4.12ms
        #   P99:     6.78ms
        ```
    """
    latencies = []
    func_name = func.__name__ if hasattr(func, '__name__') else 'function'
    
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    # Calculate statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies)
    
    stats = LatencyStats(
        name=func_name,
        count=n,
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=latencies_sorted[int(n * 0.95)],
        p99_ms=latencies_sorted[int(n * 0.99)],
        min_ms=min(latencies),
        max_ms=max(latencies),
        std_dev_ms=statistics.stdev(latencies) if n > 1 else 0.0
    )
    
    return stats


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def get_profiling_results(name_filter: Optional[str] = None) -> List[ProfilingResult]:
    """
    Retrieve all profiling results, optionally filtered by name.
    
    Args:
        name_filter: If provided, only return results matching this name
    
    Returns:
        List of ProfilingResult objects
    
    Example:
        ```python
        # Get all results
        all_results = get_profiling_results()
        
        # Get specific results
        coordinator_results = get_profiling_results(name_filter="coordinator")
        ```
    """
    results = _profiler_state.get_results()
    
    if name_filter:
        results = [r for r in results if name_filter in r.name]
    
    return results


def get_latency_stats(name: str) -> Optional[LatencyStats]:
    """
    Get statistical analysis of latency measurements for an operation.
    
    Args:
        name: Name of the operation to analyze
    
    Returns:
        LatencyStats object or None if no measurements found
    
    Example:
        ```python
        stats = get_latency_stats("api_request")
        if stats:
            print(f"P95: {stats.p95_ms:.2f}ms")
        ```
    """
    latencies = _profiler_state.get_latencies(name)
    
    if not latencies:
        return None
    
    latencies_sorted = sorted(latencies)
    n = len(latencies)
    
    return LatencyStats(
        name=name,
        count=n,
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=latencies_sorted[int(n * 0.95)],
        p99_ms=latencies_sorted[int(n * 0.99)],
        min_ms=min(latencies),
        max_ms=max(latencies),
        std_dev_ms=statistics.stdev(latencies) if n > 1 else 0.0
    )


def generate_report(output_file: Optional[str] = None) -> str:
    """
    Generate a comprehensive profiling report.
    
    Creates a human-readable report of all profiling data collected
    during the session. Optionally saves to a file.
    
    Args:
        output_file: Path to save report (None = return string only)
    
    Returns:
        Report as a formatted string
    
    Example:
        ```python
        # Generate and print report
        report = generate_report()
        print(report)
        
        # Save to file
        generate_report("performance_report.txt")
        ```
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PERFORMANCE PROFILING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary statistics
    results = get_profiling_results()
    if results:
        lines.append(f"Total Operations Profiled: {len(results)}")
        total_time = sum(r.duration_ms for r in results)
        lines.append(f"Total Time: {total_time:.2f}ms")
        
        if any(r.memory_delta_mb != 0 for r in results):
            total_memory = sum(r.memory_delta_mb for r in results)
            lines.append(f"Total Memory Delta: {total_memory:+.1f}MB")
        
        lines.append("")
        
        # Top operations by time
        lines.append("TOP OPERATIONS BY TIME:")
        lines.append("-" * 80)
        sorted_by_time = sorted(results, key=lambda r: r.duration_ms, reverse=True)
        for r in sorted_by_time[:10]:
            lines.append(f"  {r.name:<40} {r.duration_ms:>10.2f}ms")
        
        lines.append("")
        
        # Memory-intensive operations
        memory_ops = [r for r in results if abs(r.memory_delta_mb) > 0.1]
        if memory_ops:
            lines.append("MEMORY-INTENSIVE OPERATIONS:")
            lines.append("-" * 80)
            sorted_by_memory = sorted(memory_ops, key=lambda r: abs(r.memory_delta_mb), reverse=True)
            for r in sorted_by_memory[:10]:
                lines.append(f"  {r.name:<40} {r.memory_delta_mb:>10.1f}MB")
            lines.append("")
    
    # Latency statistics
    lines.append("LATENCY STATISTICS:")
    lines.append("-" * 80)
    
    # Get unique operation names
    all_latencies = _profiler_state._latencies
    for op_name in sorted(all_latencies.keys()):
        stats = get_latency_stats(op_name)
        if stats and stats.count > 0:
            lines.append(f"\n{op_name}:")
            lines.append(f"  Count:   {stats.count}")
            lines.append(f"  Mean:    {stats.mean_ms:.2f}ms")
            lines.append(f"  Median:  {stats.median_ms:.2f}ms")
            lines.append(f"  P95:     {stats.p95_ms:.2f}ms")
            lines.append(f"  P99:     {stats.p99_ms:.2f}ms")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    # Save to file if requested
    if output_file:
        Path(output_file).write_text(report)
        logger.info(f"Report saved to {output_file}")
    
    return report


def export_to_json(output_file: str):
    """
    Export all profiling data to JSON format.
    
    Useful for machine processing, visualization, or archiving.
    
    Args:
        output_file: Path to JSON file
    
    Example:
        ```python
        export_to_json("profiling_data.json")
        
        # Later, load and analyze:
        import json
        with open("profiling_data.json") as f:
            data = json.load(f)
        ```
    """
    results = get_profiling_results()
    
    data = {
        'timestamp': time.time(),
        'results': [r.to_dict() for r in results],
        'latency_stats': {}
    }
    
    # Add latency statistics
    all_latencies = _profiler_state._latencies
    for op_name in all_latencies.keys():
        stats = get_latency_stats(op_name)
        if stats:
            data['latency_stats'][op_name] = stats.to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Profiling data exported to {output_file}")


def clear_profiling_data():
    """
    Clear all collected profiling data.
    
    Useful for starting fresh measurements or freeing memory.
    
    Example:
        ```python
        # Profile operation A
        with measure_latency("operation_a"):
            do_operation_a()
        
        # Clear and profile operation B separately
        clear_profiling_data()
        
        with measure_latency("operation_b"):
            do_operation_b()
        ```
    """
    _profiler_state.clear()
    logger.debug("Profiling data cleared")


def enable_profiling():
    """Enable profiling globally."""
    _profiler_state.enable()
    logger.info("Profiling enabled")


def disable_profiling():
    """
    Disable profiling globally.
    
    Use in production to eliminate profiling overhead.
    """
    _profiler_state.disable()
    logger.info("Profiling disabled")


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def compare_implementations(
    implementations: Dict[str, Callable],
    iterations: int = 100,
    *args,
    **kwargs
) -> Dict[str, LatencyStats]:
    """
    Compare performance of multiple implementations.
    
    Useful for A/B testing optimizations or choosing between algorithms.
    
    Args:
        implementations: Dict mapping names to functions
        iterations: Number of runs per implementation
        *args: Arguments to pass to all functions
        **kwargs: Keyword arguments to pass to all functions
    
    Returns:
        Dict mapping names to LatencyStats
    
    Example:
        ```python
        results = compare_implementations({
            'original': original_function,
            'optimized': optimized_function,
            'alternative': alternative_function
        }, iterations=1000, input_data=data)
        
        for name, stats in results.items():
            print(f"{name}: {stats.mean_ms:.2f}ms (±{stats.std_dev_ms:.2f})")
        ```
    """
    results = {}
    
    for name, func in implementations.items():
        stats = measure_batch_latency(func, iterations, *args, **kwargs)
        stats.name = name  # Override with implementation name
        results[name] = stats
        
        logger.info(
            f"{name}: {stats.mean_ms:.2f}ms "
            f"(p95={stats.p95_ms:.2f}ms, p99={stats.p99_ms:.2f}ms)"
        )
    
    return results


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Check for optional dependencies
if not CPROFILE_AVAILABLE:
    logger.warning("cProfile not available - detailed profiling disabled")

if not PSUTIL_AVAILABLE:
    logger.warning("psutil not available - CPU usage tracking disabled")


__all__ = [
    # Decorators
    'profile_cpu',
    'profile_memory',
    
    # Context managers
    'measure_latency',
    'track_memory',
    
    # Measurement functions
    'measure_throughput',
    'measure_batch_latency',
    
    # Analysis & reporting
    'get_profiling_results',
    'get_latency_stats',
    'generate_report',
    'export_to_json',
    'clear_profiling_data',
    'enable_profiling',
    'disable_profiling',
    
    # Benchmarking
    'compare_implementations',
    
    # Data structures
    'ProfilingResult',
    'LatencyStats'
]
