"""Benchmarking package with lazy optional imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MAP = {
    "FastIntegratedBenchmark": (".fast_integrated_benchmark", "FastIntegratedBenchmark"),
    "IntegratedBreakthroughBenchmark": (
        ".integrated_breakthrough_benchmark",
        "IntegratedBreakthroughBenchmark",
    ),
    "GEMMProgressReport": (".gemm_progress_report", "GEMMProgressReport"),
    "PerformanceSummary": (".performance_summary", "PerformanceSummary"),
    "PolarisBreakthroughBenchmark": (
        ".polaris_breakthrough_benchmark",
        "PolarisBreakthroughBenchmark",
    ),
    "ComprehensiveBreakthroughBenchmark": (
        ".comprehensive_breakthrough_benchmark",
        "ComprehensiveBreakthroughBenchmark",
    ),
    "run_production_benchmark": (".production_kernel_benchmark", "run_production_benchmark"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load optional benchmarking symbols and cache the result."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    try:
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
    except ImportError:
        value = None

    globals()[name] = value
    return value
