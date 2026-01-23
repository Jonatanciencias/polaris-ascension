"""
Performance Regression Tests - Session 34
=========================================

Critical performance tests to prevent regressions.

Run with: pytest tests/test_performance_regression.py -v -m performance
"""

import pytest
import time
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.profiler import profile_cpu, measure_latency, measure_throughput, clear_profiling_data
from optimization.memory_pool import MessagePool, BufferPool


@pytest.fixture(autouse=True)
def reset():
    clear_profiling_data()
    yield


@pytest.mark.performance
class TestProfilingPerformance:
    """Profiling performance tests."""
    
    def test_overhead_acceptable(self):
        """Profiling adds < 5% overhead."""
        def func():
            time.sleep(0.01)
        
        # Baseline
        times = [time.perf_counter() for _ in range(100)]
        for _ in range(100):
            func()
        baseline = statistics.mean([time.perf_counter() - t for t in times[:100]])
        
        # With profiling
        @profile_cpu(name="test")
        def profiled():
            time.sleep(0.01)
        
        times2 = [time.perf_counter() for _ in range(100)]
        for _ in range(100):
            profiled()
        profiled_time = statistics.mean([time.perf_counter() - t for t in times2[:100]])
        
        overhead = (profiled_time / baseline - 1) * 100
        assert overhead < 5.0, f"Overhead {overhead:.1f}% > 5%"
    
    def test_latency_precision(self):
        """Latency measurement precise to 0.1ms."""
        target = 10.0
        
        latencies = []
        for _ in range(50):
            with measure_latency() as t:
                time.sleep(target/1000)
            latencies.append(t.elapsed_ms)
        
        error = abs(statistics.mean(latencies) - target)
        assert error < 0.1, f"Error {error:.3f}ms > 0.1ms"
    
    def test_throughput_fast(self):
        """Throughput > 1000 ops/sec."""
        def fast():
            return sum(range(100))
        
        tp = measure_throughput(fast, iterations=1000, warmup=10)
        assert tp > 1000, f"Throughput {tp:.0f} < 1000 ops/sec"


@pytest.mark.performance
class TestMemoryPoolPerformance:
    """Memory pool performance tests."""
    
    def test_message_pool_hit_rate(self):
        """Message pool hit rate > 80%."""
        pool = MessagePool(max_size=100, initial_size=10)
        
        for _ in range(20):
            pool.release(pool.acquire())
        
        pool._stats.pool_hits = 0
        pool._stats.pool_misses = 0
        
        for _ in range(100):
            pool.release(pool.acquire())
        
        assert pool.stats.hit_rate > 0.80
    
    def test_buffer_pool_reuse(self):
        """Buffer pool reuses > 50% of buffers."""
        pool = BufferPool([1024, 4096])
        
        for _ in range(20):
            pool.return_buffer(pool.get_buffer(1024))
        
        ids = set()
        for _ in range(50):
            buf = pool.get_buffer(1024)
            ids.add(id(buf))
            pool.return_buffer(buf)
        
        assert len(ids) < 25, f"{len(ids)} unique buffers > 25"


@pytest.mark.performance
class TestPerformanceTargets:
    """Test key performance targets."""
    
    def test_p95_latency_target(self):
        """P95 latency < 10ms (simulated)."""
        latencies = []
        for _ in range(100):
            with measure_latency() as t:
                time.sleep(0.005)  # 5ms work
            latencies.append(t.elapsed_ms)
        
        p95 = sorted(latencies)[95]
        assert p95 < 10.0, f"P95 {p95:.2f}ms >= 10ms"
    
    def test_throughput_target(self):
        """Throughput > 500 tasks/sec (simple ops)."""
        def task():
            return sum(range(100))
        
        tp = measure_throughput(task, iterations=1000)
        assert tp > 500, f"Throughput {tp:.0f} < 500/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
