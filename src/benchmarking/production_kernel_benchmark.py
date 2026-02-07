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

from src.optimization_engines.adaptive_kernel_selector import ProductionKernelSelector


KERNEL_IMPLS: dict[str, dict[str, Any]] = {
    "tile20": {
        "kernel_file": "src/kernels/gemm_tile20_production.cl",
        "kernel_name": "gemm_tile20_optimized",
        "local_size": (10, 10),
        "tile_size": 20,
    },
    "tile20_v3_1400": {
        "kernel_file": "src/kernels/gemm_tile20_v3_vectorized.cl",
        "kernel_name": "gemm_tile20_vectorized",
        "local_size": (10, 10),
        "tile_size": 20,
    },
    "tile24": {
        "kernel_file": "src/kernels/gemm_tile24_production.cl",
        "kernel_name": "gemm_tile24_vectorized",
        "local_size": (12, 12),
        "tile_size": 24,
    },
}


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


def _resolve_static_auto_key(size: int) -> str:
    # Scoped promoted candidate for exact 1400 benchmark profile,
    # with conservative fallback to existing production policy.
    if size == 1400:
        return "tile20_v3_1400"
    return "tile20" if size < 1800 else "tile24"


def _kernel_spec_from_key(key: str) -> tuple[str, str, tuple[int, int], int, str]:
    if key not in KERNEL_IMPLS:
        raise ValueError(
            f"Unsupported kernel '{key}'. Use auto|auto_t3_controlled|tile20|tile20_v3_1400|tile24."
        )
    spec = KERNEL_IMPLS[key]
    return (
        str(spec["kernel_file"]),
        str(spec["kernel_name"]),
        tuple(spec["local_size"]),
        int(spec["tile_size"]),
        key,
    )


def _kernel_spec(size: int, kernel: str) -> tuple[str, str, tuple[int, int], int, str]:
    key = kernel.lower()
    if key == "auto":
        key = _resolve_static_auto_key(size)
    if key == "auto_t3_controlled":
        key = _resolve_static_auto_key(size)
    return _kernel_spec_from_key(key)


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


def _build_kernel_cache(
    ctx: cl.Context,
) -> dict[str, tuple[cl.Kernel, tuple[int, int], int, str, str]]:
    cache: dict[str, tuple[cl.Kernel, tuple[int, int], int, str, str]] = {}
    for key in sorted(KERNEL_IMPLS.keys()):
        kernel_file, kernel_name, local_size, tile_size, _ = _kernel_spec_from_key(key)
        source = Path(kernel_file).read_text()
        program = _build_program(ctx, source)
        cache[key] = (
            getattr(program, kernel_name),
            local_size,
            tile_size,
            kernel_file,
            kernel_name,
        )
    return cache


def _run_t3_controlled_benchmark(
    *,
    size: int,
    iterations: int,
    sessions: int,
    seed: int,
    t3_policy_path: str | None,
) -> dict[str, Any]:
    static_selector = ProductionKernelSelector()
    controlled_selector = ProductionKernelSelector(
        enable_t3_controlled=True,
        t3_policy_path=t3_policy_path,
        t3_policy_seed=seed,
    )

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    kernel_cache = _build_kernel_cache(ctx)

    static_peak: list[float] = []
    static_avg: list[float] = []
    static_time_ms: list[float] = []
    static_errors: list[float] = []

    controlled_peak: list[float] = []
    controlled_avg: list[float] = []
    controlled_time_ms: list[float] = []
    controlled_errors: list[float] = []

    decisions: list[dict[str, Any]] = []
    fallback_count = 0

    for session_idx in range(sessions):
        session_seed = seed + session_idx * 1000
        static_rec = static_selector.select_kernel(size, size, size)
        static_key = str(static_rec["kernel_key"])

        online_rec = controlled_selector.select_kernel(size, size, size)
        online_key = str(online_rec["kernel_key"])

        static_kernel, static_local, static_tile, static_file, static_name = kernel_cache[
            static_key
        ]
        static_result = _benchmark_once(
            queue=queue,
            kernel=static_kernel,
            size=size,
            tile_size=static_tile,
            local_size=static_local,
            iterations=iterations,
            seed=session_seed + 1,
        )

        if online_key == static_key:
            online_result = dict(static_result)
            online_kernel_file = static_file
            online_kernel_name = static_name
        else:
            online_kernel, online_local, online_tile, online_file, online_name = kernel_cache[
                online_key
            ]
            online_result = _benchmark_once(
                queue=queue,
                kernel=online_kernel,
                size=size,
                tile_size=online_tile,
                local_size=online_local,
                iterations=iterations,
                seed=session_seed + 2,
            )
            online_kernel_file = online_file
            online_kernel_name = online_name

        feedback = controlled_selector.record_runtime_feedback(
            M=size,
            N=size,
            K=size,
            static_arm=static_key,
            online_arm=online_key,
            online_gflops=float(online_result["avg_gflops"]),
            static_gflops=float(static_result["avg_gflops"]),
            online_max_error=float(online_result["max_error"]),
        )
        fallback_triggered = bool(feedback["fallback_triggered"])
        if fallback_triggered:
            fallback_count += 1

        executed_key = str(feedback["executed_arm"])
        if executed_key == static_key:
            executed_result = static_result
            executed_kernel_file = static_file
            executed_kernel_name = static_name
        elif executed_key == online_key:
            executed_result = online_result
            executed_kernel_file = online_kernel_file
            executed_kernel_name = online_kernel_name
        else:
            # Defensive fallback if policy reports unknown arm.
            executed_result = static_result
            executed_kernel_file = static_file
            executed_kernel_name = static_name

        static_peak.append(static_result["peak_gflops"])
        static_avg.append(static_result["avg_gflops"])
        static_time_ms.append(static_result["time_ms"])
        static_errors.append(static_result["max_error"])

        controlled_peak.append(executed_result["peak_gflops"])
        controlled_avg.append(executed_result["avg_gflops"])
        controlled_time_ms.append(executed_result["time_ms"])
        controlled_errors.append(executed_result["max_error"])

        decisions.append(
            {
                "session": int(session_idx),
                "seed": int(session_seed),
                "static_arm": static_key,
                "online_arm": online_key,
                "executed_arm": executed_key,
                "fallback_triggered": fallback_triggered,
                "fallback_reason": feedback["fallback_reason"],
                "disable_signal": bool(feedback["disable_signal"]),
                "disable_reason": feedback["disable_reason"],
                "policy_fallback_rate": float(feedback["fallback_rate"]),
                "static_avg_gflops": float(static_result["avg_gflops"]),
                "online_avg_gflops": float(online_result["avg_gflops"]),
                "executed_avg_gflops": float(executed_result["avg_gflops"]),
                "static_peak_gflops": float(static_result["peak_gflops"]),
                "online_peak_gflops": float(online_result["peak_gflops"]),
                "executed_peak_gflops": float(executed_result["peak_gflops"]),
                "online_max_error": float(online_result["max_error"]),
                "executed_max_error": float(executed_result["max_error"]),
                "static_kernel_file": static_file,
                "online_kernel_file": online_kernel_file,
                "executed_kernel_file": executed_kernel_file,
                "static_kernel_name": static_name,
                "online_kernel_name": online_kernel_name,
                "executed_kernel_name": executed_kernel_name,
                "selection_method": str(online_rec["selection_method"]),
                "policy_selection_reason": (
                    None
                    if online_rec.get("policy") is None
                    else str(online_rec["policy"]["selection_reason"])
                ),
                "allowed_arms": (
                    []
                    if online_rec.get("policy") is None
                    else list(online_rec["policy"]["allowed_arms"])
                ),
            }
        )

    static_summary = {
        "peak_gflops": _stats(static_peak),
        "avg_gflops": _stats(static_avg),
        "time_ms": _stats(static_time_ms),
        "max_error": _stats(static_errors),
    }
    controlled_summary = {
        "peak_gflops": _stats(controlled_peak),
        "avg_gflops": _stats(controlled_avg),
        "time_ms": _stats(controlled_time_ms),
        "max_error": _stats(controlled_errors),
    }

    static_mean = float(static_summary["avg_gflops"]["mean"])
    controlled_mean = float(controlled_summary["avg_gflops"]["mean"])
    delta_vs_static_percent = (
        ((controlled_mean - static_mean) / static_mean * 100.0)
        if static_mean > 0
        else 0.0
    )
    policy_snapshot = controlled_selector.get_t3_policy_snapshot()
    policy_disabled = bool(policy_snapshot.get("disabled", False)) if policy_snapshot is not None else False

    return {
        "metadata": {
            "size": size,
            "iterations_per_session": iterations,
            "sessions": sessions,
            "seed": seed,
            "kernel_mode_requested": "auto_t3_controlled",
            "kernel_mode_resolved": "auto_t3_controlled",
            "kernel_name": "dynamic_t3_controlled",
            "kernel_file": None,
            "tile_size": None,
            "local_size": None,
            "candidate_kernels": sorted(KERNEL_IMPLS.keys()),
            "static_reference_mode": "auto",
            "t3_policy_path": (
                t3_policy_path
                if t3_policy_path is not None
                else "research/breakthrough_lab/t3_online_control/policy_controlled_block1.json"
            ),
            "platform": platform.name,
            "device": device.name,
        },
        "summary": {
            "peak_gflops": controlled_summary["peak_gflops"],
            "avg_gflops": controlled_summary["avg_gflops"],
            "time_ms": controlled_summary["time_ms"],
            "max_error": controlled_summary["max_error"],
            "static_reference": static_summary,
            "delta_vs_static_percent": float(delta_vs_static_percent),
            "fallback_count": int(fallback_count),
            "fallback_rate": float(fallback_count / max(1, sessions)),
            "policy_disabled": policy_disabled,
        },
        "decisions": decisions,
        "policy_snapshot": policy_snapshot,
    }


def run_production_benchmark(
    *,
    size: int,
    iterations: int = 20,
    sessions: int = 5,
    kernel: str = "auto",
    seed: int = 42,
    t3_policy_path: str | None = None,
) -> dict[str, Any]:
    """
    Run multi-session benchmark on production kernels and return aggregate stats.
    """
    if kernel.lower() == "auto_t3_controlled":
        return _run_t3_controlled_benchmark(
            size=size,
            iterations=iterations,
            sessions=sessions,
            seed=seed,
            t3_policy_path=t3_policy_path,
        )

    kernel_file, kernel_name, local_size, tile_size, resolved_kernel = _kernel_spec(size=size, kernel=kernel)

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
