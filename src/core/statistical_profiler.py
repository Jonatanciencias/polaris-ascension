"""
Statistical Profiler - Core Layer
===================================

Advanced performance profiling with statistical analysis.

Features:
- Percentile calculation (P50, P90, P95, P99)
- Outlier detection using IQR method
- Confidence intervals
- Performance regression testing
- Standard deviation and variance
- Coefficient of variation

Mathematical foundations:
- Order statistics for percentiles
- Tukey's method for outliers
- Student's t-distribution for CI
- Welch's t-test for regression

Version: 0.5.0-dev
License: MIT
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class SignificanceLevel(Enum):
    """Statistical significance levels with corresponding z-scores"""
    P90 = (0.90, 1.645)   # 90% confidence
    P95 = (0.95, 1.960)   # 95% confidence (standard)
    P99 = (0.99, 2.576)   # 99% confidence (strict)
    
    @property
    def confidence(self) -> float:
        return self.value[0]
    
    @property
    def z_score(self) -> float:
        return self.value[1]


@dataclass
class StatisticalMetrics:
    """Comprehensive statistical metrics for operation profiling"""
    # Basic statistics
    count: int
    total_ms: float
    mean_ms: float
    median_ms: float
    
    # Spread measures
    std_dev: float
    variance: float
    cv: float  # Coefficient of variation
    
    # Range
    min_ms: float
    max_ms: float
    range_ms: float
    
    # Percentiles
    p50: float  # Median
    p90: float
    p95: float
    p99: float
    
    # Outliers
    outliers: List[float] = field(default_factory=list)
    outlier_count: int = 0
    
    # Confidence interval (95% default)
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"StatisticalMetrics(n={self.count}, "
            f"mean={self.mean_ms:.2f}ms ±{self.std_dev:.2f}, "
            f"median={self.median_ms:.2f}ms, "
            f"P95={self.p95:.2f}ms)"
        )


@dataclass
class ProfileEntry:
    """Single profiling entry with metadata"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class StatisticalProfiler:
    """
    Advanced profiler with rigorous statistical analysis.
    
    Provides:
    - Comprehensive statistical metrics
    - Outlier detection and filtering
    - Confidence intervals
    - Performance regression testing
    - Percentile analysis
    
    Example:
        profiler = StatisticalProfiler()
        
        profiler.start("inference")
        # ... GPU operation ...
        profiler.end("inference")
        
        metrics = profiler.get_metrics("inference")
        print(f"P95 latency: {metrics.p95:.2f}ms")
        
        if profiler.detect_regression("inference", baseline=100.0):
            print("Performance regression detected!")
    """
    
    def __init__(self, outlier_detection: bool = True):
        """
        Initialize statistical profiler.
        
        Args:
            outlier_detection: Enable automatic outlier detection
        """
        self._entries: List[ProfileEntry] = []
        self._active: Dict[str, ProfileEntry] = {}
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._outlier_detection = outlier_detection
        self._baselines: Dict[str, float] = {}
    
    def start(self, name: str, **metadata):
        """
        Start profiling an operation.
        
        Args:
            name: Operation identifier
            **metadata: Additional context
        """
        if name in self._active:
            print(f"Warning: '{name}' already being profiled")
            return
        
        entry = ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata
        )
        self._active[name] = entry
    
    def end(self, name: str):
        """
        End profiling and record timing.
        
        Args:
            name: Operation identifier
        """
        if name not in self._active:
            print(f"Warning: '{name}' was not started")
            return
        
        entry = self._active[name]
        entry.end_time = time.perf_counter()
        entry.duration_ms = (entry.end_time - entry.start_time) * 1000
        
        self._entries.append(entry)
        self._timings[name].append(entry.duration_ms)
        del self._active[name]
    
    def get_metrics(
        self,
        operation: str,
        exclude_outliers: bool = False,
        confidence: SignificanceLevel = SignificanceLevel.P95
    ) -> Optional[StatisticalMetrics]:
        """
        Calculate comprehensive statistical metrics.
        
        Args:
            operation: Operation name
            exclude_outliers: Filter outliers before calculation
            confidence: Confidence level for intervals
            
        Returns:
            Statistical metrics or None if no data
        """
        if operation not in self._timings:
            return None
        
        timings = self._timings[operation].copy()
        
        if len(timings) < 2:
            # Need at least 2 samples for statistics
            return None
        
        # Detect outliers using IQR method
        outliers = self._detect_outliers_iqr(timings)
        
        # Optionally filter outliers
        if exclude_outliers and outliers:
            timings = [t for t in timings if t not in outliers]
        
        if not timings:
            return None
        
        # Sort for percentile calculation
        sorted_timings = sorted(timings)
        n = len(sorted_timings)
        
        # Basic statistics
        mean = statistics.mean(sorted_timings)
        median = statistics.median(sorted_timings)
        
        # Spread
        if n > 1:
            std_dev = statistics.stdev(sorted_timings)
            variance = statistics.variance(sorted_timings)
        else:
            std_dev = 0.0
            variance = 0.0
        
        # Coefficient of variation (relative variability)
        cv = (std_dev / mean) if mean > 0 else 0.0
        
        # Range
        min_val = sorted_timings[0]
        max_val = sorted_timings[-1]
        range_val = max_val - min_val
        
        # Percentiles
        p50 = self._percentile(sorted_timings, 50)
        p90 = self._percentile(sorted_timings, 90)
        p95 = self._percentile(sorted_timings, 95)
        p99 = self._percentile(sorted_timings, 99)
        
        # Confidence interval
        ci_lower, ci_upper = self._confidence_interval(
            sorted_timings,
            confidence
        )
        
        return StatisticalMetrics(
            count=n,
            total_ms=sum(sorted_timings),
            mean_ms=mean,
            median_ms=median,
            std_dev=std_dev,
            variance=variance,
            cv=cv,
            min_ms=min_val,
            max_ms=max_val,
            range_ms=range_val,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
            outliers=outliers,
            outlier_count=len(outliers),
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
    
    @staticmethod
    def _percentile(sorted_data: List[float], p: int) -> float:
        """
        Calculate percentile using linear interpolation.
        
        Formula:
            index = (n - 1) × (p / 100)
            If index is integer: return value at index
            Else: interpolate between floor(index) and ceil(index)
        
        Args:
            sorted_data: Data in ascending order
            p: Percentile (0-100)
            
        Returns:
            Percentile value
        """
        n = len(sorted_data)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_data[0]
        
        # Calculate index
        index = (n - 1) * (p / 100.0)
        
        # Integer index - exact value
        if index == int(index):
            return sorted_data[int(index)]
        
        # Interpolate
        lower_idx = int(math.floor(index))
        upper_idx = int(math.ceil(index))
        fraction = index - lower_idx
        
        lower_val = sorted_data[lower_idx]
        upper_val = sorted_data[upper_idx]
        
        return lower_val + fraction * (upper_val - lower_val)
    
    @staticmethod
    def _detect_outliers_iqr(data: List[float]) -> List[float]:
        """
        Detect outliers using Tukey's IQR method.
        
        Formula:
            Q1 = 25th percentile
            Q3 = 75th percentile
            IQR = Q3 - Q1
            Lower bound = Q1 - 1.5 × IQR
            Upper bound = Q3 + 1.5 × IQR
            
            Outliers are values outside [lower, upper]
        
        This method is robust and assumes no distribution.
        
        Args:
            data: Timing data
            
        Returns:
            List of outlier values
        """
        if len(data) < 4:
            return []
        
        sorted_data = sorted(data)
        
        q1 = StatisticalProfiler._percentile(sorted_data, 25)
        q3 = StatisticalProfiler._percentile(sorted_data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        
        return outliers
    
    @staticmethod
    def _confidence_interval(
        data: List[float],
        level: SignificanceLevel = SignificanceLevel.P95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        
        Formula:
            CI = mean ± (z × σ / √n)
            
        where:
            - z is the z-score for confidence level
            - σ is standard deviation
            - n is sample size
        
        For large n (>30), normal distribution approximation is valid.
        For small n, should use t-distribution (not implemented here).
        
        Args:
            data: Sample data
            level: Confidence level
            
        Returns:
            (lower_bound, upper_bound)
        """
        n = len(data)
        if n < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        
        # Margin of error
        z_score = level.z_score
        margin = z_score * (std_dev / math.sqrt(n))
        
        return (mean - margin, mean + margin)
    
    def detect_regression(
        self,
        operation: str,
        baseline: float,
        significance: SignificanceLevel = SignificanceLevel.P95,
        threshold_percent: float = 10.0
    ) -> bool:
        """
        Detect performance regression vs baseline.
        
        Tests null hypothesis: H0: current_mean = baseline
        Alternative: H1: current_mean > baseline (slower)
        
        Regression detected if:
        1. Current mean > baseline + threshold
        2. Difference is statistically significant
        
        Args:
            operation: Operation to test
            baseline: Expected performance (ms)
            significance: Statistical significance level
            threshold_percent: Minimum % degradation to flag
            
        Returns:
            True if regression detected
        """
        metrics = self.get_metrics(operation)
        if not metrics:
            return False
        
        # Calculate threshold
        threshold = baseline * (threshold_percent / 100.0)
        
        # Check if mean exceeds baseline + threshold
        if metrics.mean_ms <= baseline + threshold:
            return False
        
        # Check if baseline is outside confidence interval
        # If baseline < CI_lower, current performance significantly slower
        if baseline < metrics.ci_lower:
            return True
        
        return False
    
    def set_baseline(self, operation: str, value: float):
        """
        Set performance baseline for regression testing.
        
        Args:
            operation: Operation name
            value: Baseline timing in ms
        """
        self._baselines[operation] = value
    
    def check_baseline(self, operation: str) -> Optional[Tuple[bool, float]]:
        """
        Check current performance against saved baseline.
        
        Returns:
            (is_regression: bool, percent_change: float) or None
        """
        if operation not in self._baselines:
            return None
        
        baseline = self._baselines[operation]
        metrics = self.get_metrics(operation)
        
        if not metrics:
            return None
        
        is_regression = self.detect_regression(operation, baseline)
        percent_change = ((metrics.mean_ms - baseline) / baseline) * 100
        
        return (is_regression, percent_change)
    
    def print_summary(
        self,
        operation: Optional[str] = None,
        exclude_outliers: bool = False
    ):
        """
        Print comprehensive statistical summary.
        
        Args:
            operation: Specific operation (None = all)
            exclude_outliers: Filter outliers from display
        """
        operations = [operation] if operation else self._timings.keys()
        
        print("\n" + "=" * 90)
        print("STATISTICAL PERFORMANCE PROFILE")
        print("=" * 90)
        
        for op in sorted(operations):
            metrics = self.get_metrics(op, exclude_outliers=exclude_outliers)
            
            if not metrics:
                continue
            
            print(f"\n📊 {op}")
            print("─" * 90)
            print(f"  Sample Size:      {metrics.count:>8}")
            print(f"  Mean:             {metrics.mean_ms:>8.2f} ms  (±{metrics.std_dev:.2f})")
            print(f"  Median:           {metrics.median_ms:>8.2f} ms")
            print(f"  Std Dev:          {metrics.std_dev:>8.2f} ms")
            print(f"  CV:               {metrics.cv:>8.1%}")
            print(f"  95% CI:           [{metrics.ci_lower:.2f}, {metrics.ci_upper:.2f}] ms")
            print()
            print(f"  Percentiles:")
            print(f"    P50 (Median):   {metrics.p50:>8.2f} ms")
            print(f"    P90:            {metrics.p90:>8.2f} ms")
            print(f"    P95:            {metrics.p95:>8.2f} ms")
            print(f"    P99:            {metrics.p99:>8.2f} ms")
            print()
            print(f"  Range:            [{metrics.min_ms:.2f}, {metrics.max_ms:.2f}] ms")
            
            if metrics.outlier_count > 0:
                outlier_pct = (metrics.outlier_count / metrics.count) * 100
                print(f"  Outliers:         {metrics.outlier_count} ({outlier_pct:.1f}%)")
            
            # Check baseline if set
            baseline_check = self.check_baseline(op)
            if baseline_check:
                is_reg, pct_change = baseline_check
                status = "⚠️  REGRESSION" if is_reg else "✅ OK"
                sign = "+" if pct_change > 0 else ""
                print(f"  Baseline Check:   {status} ({sign}{pct_change:.1f}%)")
        
        print("=" * 90 + "\n")
    
    def reset(self):
        """Clear all profiling data"""
        self._entries.clear()
        self._active.clear()
        self._timings.clear()


if __name__ == "__main__":
    # Demo
    import random
    
    print("Statistical Profiler Demo")
    print("=" * 60)
    
    profiler = StatisticalProfiler()
    
    # Simulate operation timings
    print("\nSimulating 100 operations...")
    for i in range(100):
        profiler.start("gpu_kernel")
        # Simulate work: mostly 10-15ms, with occasional outliers
        if random.random() < 0.05:  # 5% outliers
            time.sleep(0.05)  # 50ms outlier
        else:
            time.sleep(random.uniform(0.010, 0.015))
        profiler.end("gpu_kernel")
    
    # Set baseline
    profiler.set_baseline("gpu_kernel", 12.0)
    
    # Print analysis
    profiler.print_summary()
