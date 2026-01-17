"""
Tests for Statistical Profiler
"""
import pytest
import time
from core.statistical_profiler import (
    StatisticalProfiler,
    SignificanceLevel,
    StatisticalMetrics
)


def test_profiler_initialization():
    """Test profiler can be initialized"""
    profiler = StatisticalProfiler()
    assert profiler is not None


def test_basic_profiling():
    """Test basic start/end profiling"""
    profiler = StatisticalProfiler()
    
    # Need at least 2 samples for statistics
    profiler.start("test_op")
    time.sleep(0.01)  # 10ms
    profiler.end("test_op")
    
    profiler.start("test_op")
    time.sleep(0.01)  # 10ms
    profiler.end("test_op")
    
    metrics = profiler.get_metrics("test_op")
    assert metrics is not None
    assert metrics.count == 2


def test_percentile_calculation():
    """Test percentile calculation accuracy"""
    profiler = StatisticalProfiler()
    
    # Simulate 100 operations with known distribution
    timings = list(range(1, 101))  # 1-100 ms
    
    for t in timings:
        profiler.start("test")
        time.sleep(t / 1000.0)
        profiler.end("test")
    
    metrics = profiler.get_metrics("test")
    
    # P50 should be around 50
    assert 45 <= metrics.p50 <= 55
    
    # P90 should be around 90
    assert 85 <= metrics.p90 <= 95
    
    # P99 should be around 99
    assert 94 <= metrics.p99 <= 100


def test_outlier_detection():
    """Test outlier detection using IQR"""
    profiler = StatisticalProfiler(outlier_detection=True)
    
    # Normal values: 10-15ms
    for _ in range(90):
        profiler.start("test")
        time.sleep(0.012)
        profiler.end("test")
    
    # Outliers: 50ms
    for _ in range(10):
        profiler.start("test")
        time.sleep(0.05)
        profiler.end("test")
    
    metrics = profiler.get_metrics("test")
    
    # Should detect some outliers (allow some variation due to timing)
    assert metrics.outlier_count > 0
    assert metrics.outlier_count <= 15  # Allow some variability


def test_confidence_interval():
    """Test confidence interval calculation"""
    profiler = StatisticalProfiler()
    
    # Generate consistent timings
    for _ in range(50):
        profiler.start("test")
        time.sleep(0.01)
        profiler.end("test")
    
    metrics = profiler.get_metrics("test", confidence=SignificanceLevel.P95)
    
    # Mean should be within CI
    assert metrics.ci_lower <= metrics.mean_ms <= metrics.ci_upper
    
    # CI should be reasonable size
    ci_width = metrics.ci_upper - metrics.ci_lower
    assert ci_width < metrics.mean_ms * 0.2  # Less than 20% of mean


def test_regression_detection():
    """Test performance regression detection"""
    profiler = StatisticalProfiler()
    
    # Set baseline at 10ms
    profiler.set_baseline("test_op", 10.0)
    
    # Simulate good performance (9-11ms)
    for _ in range(30):
        profiler.start("test_op")
        time.sleep(0.010)
        profiler.end("test_op")
    
    # Should not detect regression
    assert not profiler.detect_regression("test_op", 10.0)
    
    # Simulate degraded performance (15-17ms)
    for _ in range(30):
        profiler.start("test_op")
        time.sleep(0.016)
        profiler.end("test_op")
    
    # Now should detect regression
    metrics = profiler.get_metrics("test_op")
    assert profiler.detect_regression("test_op", 10.0)


def test_statistics_accuracy():
    """Test statistical calculations are accurate"""
    profiler = StatisticalProfiler()
    
    # Known values: 5, 10, 15, 20, 25
    known_values = [5, 10, 15, 20, 25]
    
    for val in known_values:
        profiler.start("test")
        time.sleep(val / 1000.0)
        profiler.end("test")
    
    metrics = profiler.get_metrics("test")
    
    # Mean should be 15
    assert 14 <= metrics.mean_ms <= 16
    
    # Median should be 15
    assert 14 <= metrics.median_ms <= 16
    
    # Range should be 20 (25-5)
    assert 19 <= metrics.range_ms <= 21


def test_coefficient_of_variation():
    """Test CV calculation"""
    profiler = StatisticalProfiler()
    
    # Low variation data
    for _ in range(50):
        profiler.start("stable")
        time.sleep(0.01)  # Consistent 10ms
        profiler.end("stable")
    
    metrics = profiler.get_metrics("stable")
    
    # CV should be low (< 10%)
    assert metrics.cv < 0.1
    
    # High variation data
    import random
    for _ in range(50):
        profiler.start("unstable")
        time.sleep(random.uniform(0.005, 0.020))
        profiler.end("unstable")
    
    metrics = profiler.get_metrics("unstable")
    
    # CV should be higher
    assert metrics.cv > 0.1


def test_exclude_outliers():
    """Test outlier exclusion feature"""
    profiler = StatisticalProfiler()
    
    # Normal values
    for _ in range(90):
        profiler.start("test")
        time.sleep(0.01)
        profiler.end("test")
    
    # Extreme outliers
    for _ in range(10):
        profiler.start("test")
        time.sleep(0.10)  # 10x longer
        profiler.end("test")
    
    # With outliers
    metrics_with = profiler.get_metrics("test", exclude_outliers=False)
    
    # Without outliers
    metrics_without = profiler.get_metrics("test", exclude_outliers=True)
    
    # Mean without outliers should be lower
    assert metrics_without.mean_ms < metrics_with.mean_ms


def test_multiple_operations():
    """Test tracking multiple operations"""
    profiler = StatisticalProfiler()
    
    # Profile different operations
    for op in ["kernel1", "kernel2", "kernel3"]:
        for _ in range(10):
            profiler.start(op)
            time.sleep(0.005)
            profiler.end(op)
    
    # All should have metrics
    for op in ["kernel1", "kernel2", "kernel3"]:
        metrics = profiler.get_metrics(op)
        assert metrics is not None
        assert metrics.count == 10


def test_baseline_check():
    """Test baseline checking functionality"""
    profiler = StatisticalProfiler()
    
    # Set baseline
    profiler.set_baseline("test_op", 10.0)
    
    # Generate some data
    for _ in range(50):
        profiler.start("test_op")
        time.sleep(0.012)  # 12ms (20% slower)
        profiler.end("test_op")
    
    # Check baseline
    result = profiler.check_baseline("test_op")
    assert result is not None
    
    is_regression, percent_change = result
    assert percent_change > 15  # Should show ~20% increase


def test_empty_metrics():
    """Test handling of operations with no data"""
    profiler = StatisticalProfiler()
    
    # Query non-existent operation
    metrics = profiler.get_metrics("nonexistent")
    assert metrics is None
    
    # Operation with single sample
    profiler.start("single")
    time.sleep(0.01)
    profiler.end("single")
    
    # Should return None (need >= 2 for statistics)
    metrics = profiler.get_metrics("single")
    assert metrics is None


def test_reset():
    """Test profiler reset functionality"""
    profiler = StatisticalProfiler()
    
    # Add some data
    for _ in range(10):
        profiler.start("test")
        time.sleep(0.01)
        profiler.end("test")
    
    # Reset
    profiler.reset()
    
    # Should have no metrics
    metrics = profiler.get_metrics("test")
    assert metrics is None
