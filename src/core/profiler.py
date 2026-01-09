"""
Performance Profiler Module

Tracks and reports performance metrics for GPU operations.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ProfileEntry:
    """Single profiling entry"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class Profiler:
    """Performance profiler for GPU operations"""
    
    def __init__(self):
        self._entries: List[ProfileEntry] = []
        self._active: Dict[str, ProfileEntry] = {}
        self._summary: Dict[str, List[float]] = defaultdict(list)
    
    def start(self, name: str, **metadata):
        """
        Start profiling an operation.
        
        Args:
            name: Operation name
            **metadata: Additional metadata to store
        """
        if name in self._active:
            print(f"Warning: '{name}' is already being profiled")
            return
        
        entry = ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata
        )
        self._active[name] = entry
    
    def end(self, name: str):
        """
        End profiling an operation.
        
        Args:
            name: Operation name
        """
        if name not in self._active:
            print(f"Warning: '{name}' was not started")
            return
        
        entry = self._active[name]
        entry.end_time = time.perf_counter()
        entry.duration_ms = (entry.end_time - entry.start_time) * 1000
        
        self._entries.append(entry)
        self._summary[name].append(entry.duration_ms)
        del self._active[name]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all profiled operations.
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        summary = {}
        for name, durations in self._summary.items():
            if durations:
                summary[name] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                }
        return summary
    
    def print_summary(self):
        """Print profiling summary"""
        summary = self.get_summary()
        
        if not summary:
            print("No profiling data available")
            return
        
        print("\n=== Performance Profile ===")
        print(f"{'Operation':<30} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
        print("-" * 92)
        
        for name, stats in sorted(summary.items()):
            print(
                f"{name:<30} {stats['count']:>8} "
                f"{stats['total_ms']:>12.2f} {stats['avg_ms']:>12.2f} "
                f"{stats['min_ms']:>12.2f} {stats['max_ms']:>12.2f}"
            )
        
        print("=" * 92)
    
    def reset(self):
        """Clear all profiling data"""
        self._entries.clear()
        self._active.clear()
        self._summary.clear()
