"""
Simple profiler utilities compatible with legacy examples.
"""

from __future__ import annotations

import time
from typing import Dict, List


class Profiler:
    """Small named-section profiler."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._active: Dict[str, float] = {}
        self._samples: Dict[str, List[float]] = {}

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        self._active[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        start = self._active.pop(name, None)
        if start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._samples.setdefault(name, []).append(elapsed_ms)

    def reset(self) -> None:
        self._active.clear()
        self._samples.clear()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for key, values in self._samples.items():
            total = float(sum(values))
            count = len(values)
            avg = total / count if count > 0 else 0.0
            summary[key] = {
                "count": float(count),
                "total_ms": total,
                "avg_ms": avg,
                "min_ms": float(min(values)) if values else 0.0,
                "max_ms": float(max(values)) if values else 0.0,
            }
        return summary

    def get_statistics(self) -> Dict[str, float]:
        """Aggregate statistics across all sections."""
        all_values: List[float] = []
        for values in self._samples.values():
            all_values.extend(values)

        if not all_values:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

        total = float(sum(all_values))
        count = float(len(all_values))
        return {
            "count": count,
            "mean": total / count,
            "min": float(min(all_values)),
            "max": float(max(all_values)),
            "total": total,
        }

    def print_summary(self) -> None:
        summary = self.get_summary()
        if not summary:
            print("No profile samples collected.")
            return

        print(f"{'Section':<24} {'Calls':<8} {'Avg (ms)':<12} {'Total (ms)':<12}")
        print("-" * 62)
        for name in sorted(summary):
            data = summary[name]
            print(
                f"{name:<24} {int(data['count']):<8} "
                f"{data['avg_ms']:<12.2f} {data['total_ms']:<12.2f}"
            )
