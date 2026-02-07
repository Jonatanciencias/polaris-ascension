"""
Production kernel benchmarking helpers for tile20/tile24 GEMM kernels.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np

import pyopencl as cl


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _kernel_spec(size: int, kernel: str) -> tuple[str, str, tuple[int, int], int, str]:
    key = kernel.lower()
    if key == "auto":
        # Heuristic aligned with observed behavior in this project:
        # tile20 dominates medium matrices; tile24 dominates large matrices.
        key = "tile20" if size < 1800 else "tile24"

    if key == "tile20":
        return ("src/kernels/gemm_tile20_production.cl", "gemm_tile20_optimized", (10, 10), 20, key)
    if key == "tile24":
        return ("src/kernels/gemm_tile24_production.cl", "gemm_tile24_vectorized", (12, 12), 24, key)
    raise ValueError(f"Unsupported kernel '{kernel}'. Use auto|tile20|tile24.")


def _benchmark_once(
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    size: int,
    tile_size: int,
    local_size: tuple[int, int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size), dtype=np.float32)
    B = rng.standard_normal((size, size), dtype=np.float32)
    C = np.zeros((size, size), dtype=np.float32)

    mf = cl.mem_flags
    ctx = queue.context
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

    global_size = (
        ((size + tile_size - 1) // tile_size) * local_size[0],
        ((size + tile_size - 1) // tile_size) * local_size[1],
    )

    kernel.set_args(
        np.int32(size), np.int32(size), np.int32(size),
        np.float32(1.0), a_buf, b_buf, np.float32(0.0), c_buf
    )

    for _ in range(2):
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
        times.append(time.perf_counter() - start)

    c_gpu = np.empty_like(C)
    cl.enqueue_copy(queue, c_gpu, c_buf).wait()

    max_error = float(np.max(np.abs(c_gpu - (A @ B))))
    flops = float(2 * size * size * size)
    min_time = float(np.min(times))
    avg_time = float(np.mean(times))
    return {
        "peak_gflops": flops / min_time / 1e9,
        "avg_gflops": flops / avg_time / 1e9,
        "time_ms": min_time * 1000.0,
        "max_error": max_error,
    }


def _build_program(ctx: cl.Context, source: str) -> cl.Program:
    # Mitigate known pyopencl cache warning bug without disabling cache globally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*PyOpenCL compiler caching failed.*",
        )
        return cl.Program(ctx, source).build(options=["-cl-fast-relaxed-math"])


def run_production_benchmark(
    *,
    size: int,
    iterations: int = 20,
    sessions: int = 5,
    kernel: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run multi-session benchmark on production kernels and return aggregate stats.
    """
    kernel_file, kernel_name, local_size, tile_size, resolved_kernel = _kernel_spec(
        size=size,
        kernel=kernel,
    )

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    source = Path(kernel_file).read_text()
    program = _build_program(ctx, source)
    kernel_obj = getattr(program, kernel_name)

    peak: list[float] = []
    avg: list[float] = []
    time_ms: list[float] = []
    errors: list[float] = []

    for _ in range(sessions):
        result = _benchmark_once(
            queue=queue,
            kernel=kernel_obj,
            size=size,
            tile_size=tile_size,
            local_size=local_size,
            iterations=iterations,
            seed=seed,
        )
        peak.append(result["peak_gflops"])
        avg.append(result["avg_gflops"])
        time_ms.append(result["time_ms"])
        errors.append(result["max_error"])

    return {
        "metadata": {
            "size": size,
            "iterations_per_session": iterations,
            "sessions": sessions,
            "seed": seed,
            "kernel_mode_requested": kernel,
            "kernel_mode_resolved": resolved_kernel,
            "kernel_name": kernel_name,
            "kernel_file": kernel_file,
            "tile_size": tile_size,
            "local_size": list(local_size),
            "platform": platform.name,
            "device": device.name,
        },
        "summary": {
            "peak_gflops": _stats(peak),
            "avg_gflops": _stats(avg),
            "time_ms": _stats(time_ms),
            "max_error": _stats(errors),
        },
    }
