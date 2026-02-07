"""
OpenCL operations and benchmarks compatibility.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import CLContext, gemm


def benchmark_gemm(
    cl_context: CLContext,
    *,
    M: int,
    N: int,
    K: int,
    num_trials: int = 10,
) -> Dict[str, float]:
    """
    Benchmark GEMM and report throughput metrics.
    """
    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)

    # Warm-up
    _ = cl_context.engine.gemm(A, B)

    kernel_ms: list[float] = []
    kernel_gflops: list[float] = []
    for _ in range(num_trials):
        result = cl_context.engine.gemm(A, B)
        kernel_ms.append(float(result.kernel_metrics.exec_time_ms))
        kernel_gflops.append(float(result.kernel_metrics.gflops))

    mean_ms = float(np.mean(kernel_ms))
    gflops = float(np.mean(kernel_gflops))

    bytes_moved = float((M * K + K * N + M * N) * 4.0)
    bandwidth_gb_s = float((bytes_moved / 1e9) / (mean_ms / 1000.0))

    return {
        "gflops": gflops,
        "time_ms": mean_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
    }


__all__ = ["benchmark_gemm"]
