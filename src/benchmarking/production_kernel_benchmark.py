"""
Production kernel benchmarking helpers for tile20/tile24 GEMM kernels.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyopencl as cl

from src.optimization_engines.adaptive_kernel_selector import ProductionKernelSelector
from src.optimization_engines.t5_abft_guardrails import (
    DEFAULT_POLICY_PATH as T5_DEFAULT_POLICY_PATH,
)
from src.optimization_engines.t5_abft_guardrails import DEFAULT_STATE_PATH as T5_DEFAULT_STATE_PATH
from src.optimization_engines.t5_abft_guardrails import (
    T5ABFTAutoDisableGuard,
)

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

ENV_OPENCL_PLATFORM = "RX580_OPENCL_PLATFORM"
ENV_OPENCL_DEVICE = "RX580_OPENCL_DEVICE"


def _normalize_selector(value: str | None) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip()
    return normalized if normalized else "auto"


def _matches_selector(candidates: list[str], selector: str) -> bool:
    normalized = _normalize_selector(selector)
    if normalized == "auto":
        return True
    needle = normalized.lower()
    return any(
        needle in str(candidate).lower() for candidate in candidates if candidate is not None
    )


def _is_amd_device(device: cl.Device) -> bool:
    text = f"{getattr(device, 'name', '')} {getattr(device, 'vendor', '')}".lower()
    return "amd" in text or "radeon" in text or "advanced micro devices" in text


def _select_opencl_runtime(
    *,
    opencl_platform: str | None,
    opencl_device: str | None,
) -> tuple[cl.Platform, cl.Device, dict[str, Any]]:
    platform_from_env = opencl_platform is None and os.getenv(ENV_OPENCL_PLATFORM) is not None
    device_from_env = opencl_device is None and os.getenv(ENV_OPENCL_DEVICE) is not None

    platform_selector = _normalize_selector(
        opencl_platform if opencl_platform is not None else os.getenv(ENV_OPENCL_PLATFORM)
    )
    device_selector = _normalize_selector(
        opencl_device if opencl_device is not None else os.getenv(ENV_OPENCL_DEVICE)
    )

    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms detected.")

    matching_platforms = [
        p
        for p in platforms
        if _matches_selector([str(p.name), str(p.vendor), str(p.version)], platform_selector)
    ]
    if not matching_platforms:
        available = [str(p.name) for p in platforms]
        raise ValueError(
            f"OpenCL platform selector '{platform_selector}' matched none. "
            f"Available platforms: {available}"
        )

    candidates: list[tuple[cl.Platform, cl.Device]] = []
    for platform in matching_platforms:
        try:
            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.RuntimeError:
            gpu_devices = []
        for device in gpu_devices:
            if _matches_selector(
                [
                    str(device.name),
                    str(device.vendor),
                    str(device.version),
                    str(device.driver_version),
                ],
                device_selector,
            ):
                candidates.append((platform, device))

    if not candidates:
        details: list[dict[str, Any]] = []
        for platform in matching_platforms:
            try:
                gpu_names = [
                    str(dev.name) for dev in platform.get_devices(device_type=cl.device_type.GPU)
                ]
            except cl.RuntimeError:
                gpu_names = []
            details.append({"platform": str(platform.name), "gpu_devices": gpu_names})
        hint = ""
        if platform_selector.lower() == "rusticl":
            hint = (
                " Hint: if using Rusticl canary, export RUSTICL_ENABLE=radeonsi before process startup "
                "or use CLI --rusticl-enable with fresh process."
            )
        raise ValueError(
            f"OpenCL device selector '{device_selector}' matched none in selected platforms. "
            f"Platform GPU inventory: {details}.{hint}"
        )

    def _score(item: tuple[cl.Platform, cl.Device]) -> tuple[int, int, int]:
        platform, device = item
        platform_exact = int(
            platform_selector != "auto" and str(platform.name).lower() == platform_selector.lower()
        )
        device_exact = int(
            device_selector != "auto" and str(device.name).lower() == device_selector.lower()
        )
        amd_priority = int(_is_amd_device(device))
        return (platform_exact, device_exact, amd_priority)

    best_platform, best_device = candidates[0]
    best_score = _score(candidates[0])
    for platform, device in candidates[1:]:
        score = _score((platform, device))
        if score > best_score:
            best_platform, best_device = platform, device
            best_score = score

    selection = {
        "platform_selector": platform_selector,
        "device_selector": device_selector,
        "platform_selector_from_env": platform_from_env,
        "device_selector_from_env": device_from_env,
        "selected_platform": str(best_platform.name),
        "selected_device": str(best_device.name),
        "candidate_count": int(len(candidates)),
        "available_platforms": [str(p.name) for p in platforms],
    }
    return best_platform, best_device, selection


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
            "Unsupported kernel "
            f"'{key}'. Use auto|auto_t3_controlled|auto_t5_guarded|tile20|tile20_v3_1400|tile24."
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
        np.int32(size),
        np.int32(size),
        np.int32(size),
        np.float32(1.0),
        a_buf,
        b_buf,
        np.float32(0.0),
        c_buf,
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
        program = cl.Program(ctx, source).build(options=["-cl-fast-relaxed-math"])
        return cast(cl.Program, program)


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


def _select_indices(size: int, sample_count: int) -> np.ndarray:
    if sample_count >= size:
        return np.arange(size, dtype=np.int32)
    idx = np.linspace(0, size - 1, num=sample_count, dtype=np.int32)
    return np.unique(idx)


def _prepare_abft_expectations(
    *,
    a: np.ndarray,
    b: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
) -> dict[str, Any]:
    ones_n = np.ones((b.shape[1],), dtype=np.float32)
    b_colsum = b @ ones_n
    expected_rows = a[row_indices, :] @ b_colsum

    a_rowsum = np.sum(a, axis=0)
    expected_cols = a_rowsum @ b[:, col_indices]
    return {
        "row_indices": row_indices,
        "col_indices": col_indices,
        "expected_rows": expected_rows.astype(np.float32),
        "expected_cols": expected_cols.astype(np.float32),
    }


def _abft_residuals(c: np.ndarray, abft: dict[str, Any]) -> tuple[float, float]:
    row_indices = abft["row_indices"]
    col_indices = abft["col_indices"]
    expected_rows = abft["expected_rows"]
    expected_cols = abft["expected_cols"]

    observed_rows = np.sum(c[row_indices, :], axis=1)
    observed_cols = np.sum(c[:, col_indices], axis=0)
    row_residual = float(np.max(np.abs(observed_rows - expected_rows)))
    col_residual = float(np.max(np.abs(observed_cols - expected_cols)))
    return row_residual, col_residual


def _prepare_projection_bank(
    *,
    a: np.ndarray,
    b: np.ndarray,
    seed: int,
    projection_count: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    size = a.shape[0]
    u = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(projection_count, size))
    v = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(size, projection_count))

    left = u @ a
    right = b @ v
    expected = np.sum(left * right.T, axis=1).astype(np.float32)
    return {"u": u, "v": v, "expected": expected}


def _projection_residuals(c: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    observed = np.sum(projection["u"] * (c @ projection["v"]).T, axis=1).astype(np.float32)
    return cast(np.ndarray, np.abs(observed - projection["expected"]).astype(np.float32))


def _run_kernel_with_copy(
    *,
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    global_size: tuple[int, int],
    local_size: tuple[int, int],
    c_buf: cl.Buffer,
    c_host: np.ndarray,
) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    queue.finish()
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)
    cl.enqueue_copy(queue, c_host, c_buf).wait()
    return elapsed_ms, c_host.copy()


def _benchmark_once_with_t5_guard(
    *,
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    size: int,
    tile_size: int,
    local_size: tuple[int, int],
    iterations: int,
    seed: int,
    sampling_period: int,
    row_samples: int,
    col_samples: int,
    projection_count: int,
    residual_scale: float,
    residual_margin: float,
    residual_floor: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    c_host = np.zeros((size, size), dtype=np.float32)

    row_indices = _select_indices(size, row_samples)
    col_indices = _select_indices(size, col_samples)
    abft = _prepare_abft_expectations(
        a=a,
        b=b,
        row_indices=row_indices,
        col_indices=col_indices,
    )
    projection = _prepare_projection_bank(
        a=a,
        b=b,
        seed=seed + 9001,
        projection_count=projection_count,
    )

    mf = cl.mem_flags
    ctx = queue.context
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c_host)

    global_size = (
        ((size + tile_size - 1) // tile_size) * local_size[0],
        ((size + tile_size - 1) // tile_size) * local_size[1],
    )

    kernel.set_args(
        np.int32(size),
        np.int32(size),
        np.int32(size),
        np.float32(1.0),
        a_buf,
        b_buf,
        np.float32(0.0),
        c_buf,
    )

    for _ in range(2):
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()

    cal_ms, c_cal = _run_kernel_with_copy(
        queue=queue,
        kernel=kernel,
        global_size=global_size,
        local_size=local_size,
        c_buf=c_buf,
        c_host=c_host,
    )
    _ = cal_ms
    cal_row_res, cal_col_res = _abft_residuals(c_cal, abft)
    cal_proj_residuals = _projection_residuals(c_cal, projection)
    row_threshold = max(residual_floor, cal_row_res * residual_scale + residual_margin)
    col_threshold = max(residual_floor, cal_col_res * residual_scale + residual_margin)
    proj_thresholds = np.maximum(
        residual_floor,
        cal_proj_residuals * residual_scale + residual_margin,
    ).astype(np.float32)

    times_kernel_ms: list[float] = []
    times_verify_ms: list[float] = []
    anomalies = 0
    checked_runs = 0
    last_c = c_cal

    for step in range(iterations):
        kernel_ms, c = _run_kernel_with_copy(
            queue=queue,
            kernel=kernel,
            global_size=global_size,
            local_size=local_size,
            c_buf=c_buf,
            c_host=c_host,
        )
        times_kernel_ms.append(kernel_ms)
        last_c = c

        if step % max(1, sampling_period) != 0:
            continue

        checked_runs += 1
        verify_start = time.perf_counter()
        row_res, col_res = _abft_residuals(c, abft)
        proj_residuals = _projection_residuals(c, projection)
        verify_ms = float((time.perf_counter() - verify_start) * 1000.0)
        times_verify_ms.append(verify_ms)

        anomaly = (
            row_res > row_threshold
            or col_res > col_threshold
            or bool(np.any(proj_residuals > proj_thresholds))
        )
        if anomaly:
            anomalies += 1

    c_ref = a @ b
    max_error = float(np.max(np.abs(last_c - c_ref)))
    flops = float(2 * size * size * size)
    kernel_total_ms = float(np.sum(times_kernel_ms)) if times_kernel_ms else 0.0
    verify_total_ms = float(np.sum(times_verify_ms)) if times_verify_ms else 0.0
    effective_total_ms = kernel_total_ms + verify_total_ms

    a_buf.release()
    b_buf.release()
    c_buf.release()

    peak_ms = float(np.min(times_kernel_ms)) if times_kernel_ms else 1.0
    avg_ms = float(np.mean(times_kernel_ms)) if times_kernel_ms else 1.0

    return {
        "peak_gflops": flops / (peak_ms * 1e6),
        "avg_gflops": flops / (avg_ms * 1e6),
        "time_ms": peak_ms,
        "max_error": max_error,
        "t5_abft": {
            "sampling_period": int(sampling_period),
            "checked_runs": int(checked_runs),
            "total_runs": int(iterations),
            "sampling_coverage": float(checked_runs / max(1, iterations)),
            "false_positive_count": int(anomalies),
            "false_positive_rate": float(anomalies / max(1, checked_runs)),
            "effective_overhead_percent": (
                float(verify_total_ms / kernel_total_ms * 100.0) if kernel_total_ms > 0 else 0.0
            ),
            "kernel_time_total_ms": kernel_total_ms,
            "verify_time_total_ms": verify_total_ms,
            "effective_gflops": (
                float(flops * iterations / (effective_total_ms * 1e6))
                if effective_total_ms > 0
                else 0.0
            ),
            "max_error": max_error,
        },
    }


def _run_t5_guarded_benchmark(
    *,
    size: int,
    iterations: int,
    sessions: int,
    seed: int,
    t5_policy_path: str,
    t5_state_path: str,
    platform: cl.Platform,
    device: cl.Device,
    platform_selection: dict[str, Any],
) -> dict[str, Any]:
    selector = ProductionKernelSelector()
    guard = T5ABFTAutoDisableGuard(policy_path=t5_policy_path, state_path=t5_state_path)

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    kernel_cache = _build_kernel_cache(ctx)

    peak_vals: list[float] = []
    avg_vals: list[float] = []
    time_vals: list[float] = []
    err_vals: list[float] = []

    checked_runs_total = 0
    total_runs_total = 0
    false_positive_total = 0
    kernel_time_total_ms = 0.0
    verify_time_total_ms = 0.0
    guardrails_all_passed = True

    session_details: list[dict[str, Any]] = []

    static_rec = selector.select_kernel(size, size, size)
    kernel_key = str(static_rec["kernel_key"])
    kernel_obj, local_size, tile_size, kernel_file, kernel_name = kernel_cache[kernel_key]

    for session_idx in range(sessions):
        session_seed = seed + session_idx * 1000
        if guard.enabled:
            result = _benchmark_once_with_t5_guard(
                queue=queue,
                kernel=kernel_obj,
                size=size,
                tile_size=tile_size,
                local_size=local_size,
                iterations=iterations,
                seed=session_seed,
                sampling_period=guard.sampling.sampling_period,
                row_samples=guard.sampling.row_samples,
                col_samples=guard.sampling.col_samples,
                projection_count=guard.sampling.projection_count,
                residual_scale=guard.sampling.residual_scale,
                residual_margin=guard.sampling.residual_margin,
                residual_floor=guard.sampling.residual_floor,
            )
            t5_metrics = result["t5_abft"]
            eval_result = guard.evaluate_session(
                session_id=session_idx,
                metrics={
                    "false_positive_rate": float(t5_metrics["false_positive_rate"]),
                    "effective_overhead_percent": float(t5_metrics["effective_overhead_percent"]),
                    "max_error": float(t5_metrics["max_error"]),
                },
            )
            guardrails_all_passed = guardrails_all_passed and bool(eval_result["all_passed"])

            checked_runs_total += int(t5_metrics["checked_runs"])
            total_runs_total += int(t5_metrics["total_runs"])
            false_positive_total += int(t5_metrics["false_positive_count"])
            kernel_time_total_ms += float(t5_metrics["kernel_time_total_ms"])
            verify_time_total_ms += float(t5_metrics["verify_time_total_ms"])
            abft_enabled = True
        else:
            result = _benchmark_once(
                queue=queue,
                kernel=kernel_obj,
                size=size,
                tile_size=tile_size,
                local_size=local_size,
                iterations=iterations,
                seed=session_seed,
            )
            t5_metrics = {
                "checked_runs": 0,
                "total_runs": int(iterations),
                "sampling_coverage": 0.0,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "effective_overhead_percent": 0.0,
                "kernel_time_total_ms": 0.0,
                "verify_time_total_ms": 0.0,
                "effective_gflops": float(result["avg_gflops"]),
                "max_error": float(result["max_error"]),
            }
            eval_result = {
                "all_passed": True,
                "disable_signal": False,
                "failed_guardrails": [],
                "enabled_after_eval": False,
                "disable_reason": guard.disable_reason,
                "fallback_action": "abft_already_disabled_runtime_plain_path",
            }
            total_runs_total += int(iterations)
            abft_enabled = False

        peak_vals.append(float(result["peak_gflops"]))
        avg_vals.append(float(result["avg_gflops"]))
        time_vals.append(float(result["time_ms"]))
        err_vals.append(float(result["max_error"]))

        session_details.append(
            {
                "session": int(session_idx),
                "seed": int(session_seed),
                "kernel_key": kernel_key,
                "kernel_name": kernel_name,
                "abft_enabled": abft_enabled,
                "runtime_metrics": t5_metrics,
                "guardrail_eval": eval_result,
            }
        )

    false_positive_rate = float(false_positive_total / max(1, checked_runs_total))
    effective_overhead_percent = (
        float(verify_time_total_ms / kernel_time_total_ms * 100.0)
        if kernel_time_total_ms > 0.0
        else 0.0
    )

    policy_evidence = guard.policy.get("stress_evidence", {})
    critical_recall = float(policy_evidence.get("critical_recall", 0.0))
    uniform_recall = float(policy_evidence.get("uniform_recall", 0.0))

    return {
        "metadata": {
            "size": size,
            "iterations_per_session": iterations,
            "sessions": sessions,
            "seed": seed,
            "kernel_mode_requested": "auto_t5_guarded",
            "kernel_mode_resolved": "auto_t5_guarded",
            "kernel_name": kernel_name,
            "kernel_file": kernel_file,
            "tile_size": tile_size,
            "local_size": list(local_size),
            "platform": platform.name,
            "device": device.name,
            "platform_selection": platform_selection,
            "t5_policy_path": t5_policy_path,
            "t5_state_path": t5_state_path,
            "t5_policy_id": guard.policy["policy_id"],
        },
        "summary": {
            "peak_gflops": _stats(peak_vals),
            "avg_gflops": _stats(avg_vals),
            "time_ms": _stats(time_vals),
            "max_error": _stats(err_vals),
            "t5_abft": {
                "guardrails_all_passed": bool(guardrails_all_passed),
                "enabled_final": bool(guard.enabled),
                "disable_events": int(guard.disable_events),
                "disable_reason": guard.disable_reason,
                "checked_runs": int(checked_runs_total),
                "total_runs": int(total_runs_total),
                "sampling_coverage": float(checked_runs_total / max(1, total_runs_total)),
                "false_positive_rate": false_positive_rate,
                "effective_overhead_percent": effective_overhead_percent,
                "critical_recall_reference": critical_recall,
                "uniform_recall_reference": uniform_recall,
            },
        },
        "sessions": session_details,
        "guard_state": guard.state_snapshot(),
    }


def _run_t3_controlled_benchmark(
    *,
    size: int,
    iterations: int,
    sessions: int,
    seed: int,
    t3_policy_path: str | None,
    platform: cl.Platform,
    device: cl.Device,
    platform_selection: dict[str, Any],
) -> dict[str, Any]:
    static_selector = ProductionKernelSelector()
    controlled_selector = ProductionKernelSelector(
        enable_t3_controlled=True,
        t3_policy_path=t3_policy_path,
        t3_policy_seed=seed,
    )

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
        ((controlled_mean - static_mean) / static_mean * 100.0) if static_mean > 0 else 0.0
    )
    policy_snapshot = controlled_selector.get_t3_policy_snapshot()
    policy_disabled = (
        bool(policy_snapshot.get("disabled", False)) if policy_snapshot is not None else False
    )

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
            "platform_selection": platform_selection,
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
    t5_policy_path: str = T5_DEFAULT_POLICY_PATH,
    t5_state_path: str = T5_DEFAULT_STATE_PATH,
    opencl_platform: str | None = None,
    opencl_device: str | None = None,
    rusticl_enable: str | None = None,
) -> dict[str, Any]:
    """
    Run multi-session benchmark on production kernels and return aggregate stats.
    """
    if rusticl_enable is not None:
        os.environ["RUSTICL_ENABLE"] = str(rusticl_enable)

    platform, device, platform_selection = _select_opencl_runtime(
        opencl_platform=opencl_platform,
        opencl_device=opencl_device,
    )

    if kernel.lower() == "auto_t3_controlled":
        return _run_t3_controlled_benchmark(
            size=size,
            iterations=iterations,
            sessions=sessions,
            seed=seed,
            t3_policy_path=t3_policy_path,
            platform=platform,
            device=device,
            platform_selection=platform_selection,
        )
    if kernel.lower() == "auto_t5_guarded":
        return _run_t5_guarded_benchmark(
            size=size,
            iterations=iterations,
            sessions=sessions,
            seed=seed,
            t5_policy_path=t5_policy_path,
            t5_state_path=t5_state_path,
            platform=platform,
            device=device,
            platform_selection=platform_selection,
        )

    kernel_file, kernel_name, local_size, tile_size, resolved_kernel = _kernel_spec(
        size=size, kernel=kernel
    )
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
            "platform_selection": platform_selection,
        },
        "summary": {
            "peak_gflops": _stats(peak),
            "avg_gflops": _stats(avg),
            "time_ms": _stats(time_ms),
            "max_error": _stats(errors),
        },
    }
