#!/usr/bin/env python3
"""
Week 4 T5 ABFT-lite detect-only runner (coverage refinement).

Implements a lab-side ABFT-lite checksum verifier with:
- deterministic workload generation
- explicit fault injection campaign
- overhead and detection-quality measurement
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.optimization_engines.adaptive_kernel_selector import ProductionKernelSelector


@dataclass(frozen=True)
class KernelSpec:
    kernel_file: str
    kernel_name: str
    tile_size: int
    local_size: tuple[int, int]


KERNEL_SPECS: dict[str, KernelSpec] = {
    "tile20": KernelSpec(
        kernel_file="src/kernels/gemm_tile20_production.cl",
        kernel_name="gemm_tile20_optimized",
        tile_size=20,
        local_size=(10, 10),
    ),
    "tile20_v3_1400": KernelSpec(
        kernel_file="src/kernels/gemm_tile20_v3_vectorized.cl",
        kernel_name="gemm_tile20_vectorized",
        tile_size=20,
        local_size=(10, 10),
    ),
    "tile24": KernelSpec(
        kernel_file="src/kernels/gemm_tile24_production.cl",
        kernel_name="gemm_tile24_vectorized",
        tile_size=24,
        local_size=(12, 12),
    ),
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


def _build_program(ctx: cl.Context, source: str) -> cl.Program:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*PyOpenCL compiler caching failed.*",
        )
        return cl.Program(ctx, source).build(options=["-cl-fast-relaxed-math"])


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
    # Rademacher vectors provide stable magnitude and broad fault observability.
    rng = np.random.default_rng(seed)
    size = a.shape[0]
    u = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(projection_count, size))
    v = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(size, projection_count))

    left = u @ a
    right = b @ v
    expected = np.sum(left * right.T, axis=1).astype(np.float32)
    return {
        "u": u,
        "v": v,
        "expected": expected,
    }


def _projection_residuals(c: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    observed = np.sum(projection["u"] * (c @ projection["v"]).T, axis=1).astype(np.float32)
    return np.abs(observed - projection["expected"]).astype(np.float32)


def _inject_faults(
    *,
    c: np.ndarray,
    rng: np.random.Generator,
    fault_model: str,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    faults_per_matrix: int,
    magnitude: float,
) -> np.ndarray:
    out = c.copy()
    size = out.shape[0]
    sign = lambda: 1.0 if float(rng.random()) < 0.5 else -1.0

    for k in range(faults_per_matrix):
        if fault_model == "critical_monitored":
            if k % 2 == 0:
                i = int(rng.choice(row_indices))
                j = int(rng.integers(0, size))
            else:
                i = int(rng.integers(0, size))
                j = int(rng.choice(col_indices))
        elif fault_model == "uniform_random":
            i = int(rng.integers(0, size))
            j = int(rng.integers(0, size))
        else:
            raise ValueError(f"Unsupported fault model: {fault_model}")
        out[i, j] += np.float32(sign() * magnitude)
    return out


def _run_kernel_once(
    *,
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    global_size: tuple[int, int],
    local_size: tuple[int, int],
    size: int,
    a_buf: cl.Buffer,
    b_buf: cl.Buffer,
    c_buf: cl.Buffer,
    c_host: np.ndarray,
) -> tuple[float, np.ndarray]:
    event = kernel(
        queue,
        global_size,
        local_size,
        np.int32(size),
        np.int32(size),
        np.int32(size),
        np.float32(1.0),
        a_buf,
        b_buf,
        np.float32(0.0),
        c_buf,
    )
    event.wait()
    elapsed_ms = float((event.profile.end - event.profile.start) * 1e-6)
    cl.enqueue_copy(queue, c_host, c_buf).wait()
    return elapsed_ms, c_host.copy()


def _mode_label(period: int) -> str:
    if period <= 1:
        return "always"
    return f"periodic_{period}"


def _run_mode(
    *,
    mode_period: int,
    sizes: list[int],
    sessions: int,
    iterations: int,
    warmup: int,
    row_samples: int,
    col_samples: int,
    faults_per_matrix: int,
    fault_scale: float,
    fault_abs_min: float,
    residual_scale: float,
    residual_margin: float,
    residual_floor: float,
    projection_count: int,
    correctness_threshold: float,
    seed: int,
    selector: ProductionKernelSelector,
    ctx: cl.Context,
    queue: cl.CommandQueue,
) -> dict[str, Any]:
    label = _mode_label(mode_period)
    fault_models = ["critical_monitored", "uniform_random"]

    program_cache: dict[tuple[str, str], cl.Kernel] = {}
    kernel_times_ms: list[float] = []
    verify_times_ms: list[float] = []
    gflops_kernel: list[float] = []
    gflops_effective: list[float] = []
    clean_anomaly_count = 0
    checked_runs = 0
    total_runs = 0
    correctness_errors: list[float] = []
    fault_trials: list[dict[str, Any]] = []
    per_size: dict[str, dict[str, Any]] = {}

    for size in sizes:
        rec = selector.select_kernel(size, size, size)
        kernel_key = rec["kernel_key"]
        spec = KERNEL_SPECS[kernel_key]
        key = (spec.kernel_file, spec.kernel_name)

        if key not in program_cache:
            source = (Path(__file__).resolve().parents[3] / spec.kernel_file).read_text()
            program = _build_program(ctx, source)
            program_cache[key] = getattr(program, spec.kernel_name)
        kernel_obj = program_cache[key]

        global_size = (
            ((size + spec.tile_size - 1) // spec.tile_size) * spec.local_size[0],
            ((size + spec.tile_size - 1) // spec.tile_size) * spec.local_size[1],
        )

        size_kernel_times: list[float] = []
        size_verify_times: list[float] = []
        size_gflops_kernel: list[float] = []
        size_gflops_effective: list[float] = []
        size_checked = 0
        size_clean_anomalies = 0
        size_correctness_errors: list[float] = []
        size_fault_trials: list[dict[str, Any]] = []

        for session in range(sessions):
            case_seed = seed + size * 1000 + session * 17 + mode_period
            rng = np.random.default_rng(case_seed)
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
                seed=case_seed + 9001,
                projection_count=projection_count,
            )

            mf = cl.mem_flags
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            c_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c_host)

            for _ in range(warmup):
                _run_kernel_once(
                    queue=queue,
                    kernel=kernel_obj,
                    global_size=global_size,
                    local_size=spec.local_size,
                    size=size,
                    a_buf=a_buf,
                    b_buf=b_buf,
                    c_buf=c_buf,
                    c_host=c_host,
                )

            cal_ms, c_cal = _run_kernel_once(
                queue=queue,
                kernel=kernel_obj,
                global_size=global_size,
                local_size=spec.local_size,
                size=size,
                a_buf=a_buf,
                b_buf=b_buf,
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

            c_ref = a @ b
            max_error = float(np.max(np.abs(c_cal - c_ref)))
            correctness_errors.append(max_error)
            size_correctness_errors.append(max_error)

            for step in range(iterations):
                total_runs += 1
                kernel_ms, c = _run_kernel_once(
                    queue=queue,
                    kernel=kernel_obj,
                    global_size=global_size,
                    local_size=spec.local_size,
                    size=size,
                    a_buf=a_buf,
                    b_buf=b_buf,
                    c_buf=c_buf,
                    c_host=c_host,
                )

                ops = float(2 * size * size * size)
                k_gflops = ops / (kernel_ms * 1e6)

                kernel_times_ms.append(kernel_ms)
                size_kernel_times.append(kernel_ms)
                gflops_kernel.append(float(k_gflops))
                size_gflops_kernel.append(float(k_gflops))

                verify_ms = 0.0
                checked = (step % mode_period == 0)
                if checked:
                    checked_runs += 1
                    size_checked += 1
                    t0 = time.perf_counter()
                    clean_row_res, clean_col_res = _abft_residuals(c, abft)
                    clean_proj_residuals = _projection_residuals(c, projection)
                    verify_ms = float((time.perf_counter() - t0) * 1000.0)
                    verify_times_ms.append(verify_ms)
                    size_verify_times.append(verify_ms)

                    clean_anomaly = (
                        clean_row_res > row_threshold
                        or clean_col_res > col_threshold
                        or bool(np.any(clean_proj_residuals > proj_thresholds))
                    )
                    if clean_anomaly:
                        clean_anomaly_count += 1
                        size_clean_anomalies += 1

                    max_abs_c = float(np.max(np.abs(c)))
                    fault_mag = max(fault_abs_min, max_abs_c * fault_scale)

                    for model_idx, model in enumerate(fault_models):
                        fault_seed = case_seed + step * 100 + model_idx
                        fault_rng = np.random.default_rng(fault_seed)
                        c_fault = _inject_faults(
                            c=c,
                            rng=fault_rng,
                            fault_model=model,
                            row_indices=row_indices,
                            col_indices=col_indices,
                            faults_per_matrix=faults_per_matrix,
                            magnitude=fault_mag,
                        )
                        f_row_res, f_col_res = _abft_residuals(c_fault, abft)
                        f_proj_residuals = _projection_residuals(c_fault, projection)
                        detected = (
                            f_row_res > row_threshold
                            or f_col_res > col_threshold
                            or bool(np.any(f_proj_residuals > proj_thresholds))
                        )
                        entry = {
                            "mode": label,
                            "size": size,
                            "session": session,
                            "iteration": step,
                            "fault_model": model,
                            "detected": bool(detected),
                            "row_residual": float(f_row_res),
                            "col_residual": float(f_col_res),
                            "projection_residual_max": float(np.max(f_proj_residuals)),
                            "row_threshold": float(row_threshold),
                            "col_threshold": float(col_threshold),
                            "projection_threshold_max": float(np.max(proj_thresholds)),
                            "fault_magnitude": float(fault_mag),
                        }
                        fault_trials.append(entry)
                        size_fault_trials.append(entry)

                effective_ms = kernel_ms + verify_ms
                e_gflops = ops / (effective_ms * 1e6)
                gflops_effective.append(float(e_gflops))
                size_gflops_effective.append(float(e_gflops))

            a_buf.release()
            b_buf.release()
            c_buf.release()

        size_trials = size_fault_trials
        by_fault_model_size: dict[str, Any] = {}
        for model in fault_models:
            model_trials = [x for x in size_trials if x["fault_model"] == model]
            detected_count = sum(1 for x in model_trials if x["detected"])
            trials = len(model_trials)
            by_fault_model_size[model] = {
                "trials": trials,
                "detected": detected_count,
                "misses": int(trials - detected_count),
                "recall": float(detected_count / trials) if trials > 0 else 0.0,
            }

        size_kernel_total = float(np.sum(size_kernel_times)) if size_kernel_times else 0.0
        size_verify_total = float(np.sum(size_verify_times)) if size_verify_times else 0.0
        size_overhead = (
            float(size_verify_total / size_kernel_total * 100.0)
            if size_kernel_total > 0
            else 0.0
        )

        per_size[str(size)] = {
            "kernel_key": kernel_key,
            "kernel_name": spec.kernel_name,
            "runs": len(size_kernel_times),
            "checked_runs": size_checked,
            "sampling_coverage": float(size_checked / len(size_kernel_times))
            if size_kernel_times
            else 0.0,
            "kernel_time_ms": _stats(size_kernel_times),
            "verify_time_ms": _stats(size_verify_times) if size_verify_times else None,
            "kernel_gflops": _stats(size_gflops_kernel),
            "effective_gflops": _stats(size_gflops_effective),
            "effective_overhead_percent": float(size_overhead),
            "false_positives_clean": int(size_clean_anomalies),
            "max_error": float(np.max(size_correctness_errors))
            if size_correctness_errors
            else 0.0,
            "by_fault_model": by_fault_model_size,
        }

    by_fault_model: dict[str, Any] = {}
    for model in fault_models:
        model_trials = [x for x in fault_trials if x["fault_model"] == model]
        detected_count = sum(1 for x in model_trials if x["detected"])
        trials = len(model_trials)
        by_fault_model[model] = {
            "trials": trials,
            "detected": detected_count,
            "misses": int(trials - detected_count),
            "recall": float(detected_count / trials) if trials > 0 else 0.0,
        }

    kernel_total_ms = float(np.sum(kernel_times_ms)) if kernel_times_ms else 0.0
    verify_total_ms = float(np.sum(verify_times_ms)) if verify_times_ms else 0.0
    effective_overhead_percent = (
        float(verify_total_ms / kernel_total_ms * 100.0) if kernel_total_ms > 0 else 0.0
    )

    false_positive_rate = (
        float(clean_anomaly_count / checked_runs) if checked_runs > 0 else 0.0
    )
    critical_recall = by_fault_model["critical_monitored"]["recall"]
    critical_misses = by_fault_model["critical_monitored"]["misses"]
    uniform_recall = by_fault_model["uniform_random"]["recall"]

    mode_pass = (
        critical_recall >= 0.95
        and uniform_recall >= 0.95
        and critical_misses == 0
        and effective_overhead_percent <= 5.0
        and false_positive_rate <= 0.05
    )
    stop_rule_triggered = (
        effective_overhead_percent > 8.0
        and (critical_recall < 0.95 or uniform_recall < 0.95)
    )

    return {
        "mode": label,
        "sampling_period": mode_period,
        "total_runs": total_runs,
        "checked_runs": checked_runs,
        "sampling_coverage": float(checked_runs / total_runs) if total_runs > 0 else 0.0,
        "kernel_time_ms": _stats(kernel_times_ms),
        "verify_time_ms": _stats(verify_times_ms) if verify_times_ms else None,
        "kernel_gflops": _stats(gflops_kernel),
        "effective_gflops": _stats(gflops_effective),
        "effective_overhead_percent": float(effective_overhead_percent),
        "false_positive_count": int(clean_anomaly_count),
        "false_positive_rate": float(false_positive_rate),
        "max_error": float(np.max(correctness_errors)) if correctness_errors else 0.0,
        "correctness_passed": bool(
            (np.max(correctness_errors) if correctness_errors else 0.0)
            <= correctness_threshold
        ),
        "by_fault_model": by_fault_model,
        "critical_recall": float(critical_recall),
        "uniform_recall": float(uniform_recall),
        "critical_misses": int(critical_misses),
        "mode_pass": bool(mode_pass),
        "stop_rule_triggered": bool(stop_rule_triggered),
        "per_size": per_size,
    }


def _pick_recommended_mode(modes: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [m for m in modes if m["mode_pass"]]
    if passing:
        return sorted(
            passing,
            key=lambda x: (
                -x["critical_recall"],
                -x["uniform_recall"],
                x["effective_overhead_percent"],
                x["false_positive_rate"],
            ),
        )[0]
    return sorted(
        modes,
        key=lambda x: (
            x["critical_misses"],
            -x["critical_recall"],
            -x["uniform_recall"],
            x["effective_overhead_percent"],
        ),
    )[0]


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T5 Week 4 Block 2 - ABFT-lite Coverage Refinement Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Sizes: {report['metadata']['sizes']} | Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']}"
    )
    lines.append(
        f"- Sampling periods: {report['metadata']['sampling_periods']} | Row samples={report['metadata']['row_samples']} | Col samples={report['metadata']['col_samples']}"
    )
    lines.append(
        f"- Projection checks: count={report['metadata']['projection_count']}"
    )
    lines.append(
        f"- Fault injection: faults_per_matrix={report['metadata']['faults_per_matrix']}, models={report['metadata']['fault_models']}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Recommended mode: `{report['summary']['recommended_mode']}`")
    lines.append(
        f"- Decision hint: `{report['summary']['decision_hint']}` ({report['summary']['decision_rationale']})"
    )
    lines.append(
        f"- Stop rule triggered: {report['summary']['stop_rule_triggered']} ({report['summary']['stop_rule_reason']})"
    )
    lines.append("")
    lines.append("## Mode Comparison")
    lines.append("")
    lines.append(
        "| Mode | Coverage | Overhead % | Critical Recall | Uniform Recall | Critical Misses | False Pos Rate | Correctness | Pass |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for mode in report["modes"]:
        lines.append(
            f"| {mode['mode']} | {mode['sampling_coverage']:.3f} | "
            f"{mode['effective_overhead_percent']:.3f} | {mode['critical_recall']:.3f} | "
            f"{mode['uniform_recall']:.3f} | "
            f"{mode['critical_misses']} | {mode['false_positive_rate']:.3f} | "
            f"{mode['correctness_passed']} | {mode['mode_pass']} |"
        )

    lines.append("")
    lines.append("## Recommended Mode Details")
    lines.append("")
    rec = report["summary"]["recommended_mode_details"]
    lines.append(f"- Kernel GFLOPS mean: {rec['kernel_gflops']['mean']:.3f}")
    lines.append(f"- Effective GFLOPS mean (with ABFT): {rec['effective_gflops']['mean']:.3f}")
    lines.append(f"- Effective overhead: {rec['effective_overhead_percent']:.3f}%")
    lines.append(f"- Critical recall: {rec['critical_recall']:.3f}")
    lines.append(f"- Uniform-random recall: {rec['by_fault_model']['uniform_random']['recall']:.3f}")
    lines.append("")
    lines.append("## Per-Size (Recommended Mode)")
    lines.append("")
    lines.append("| Size | Kernel | Coverage | Overhead % | Critical Recall | Uniform Recall |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for size_key, data in sorted(rec["per_size"].items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {size_key} | {data['kernel_key']} | {data['sampling_coverage']:.3f} | "
            f"{data['effective_overhead_percent']:.3f} | "
            f"{data['by_fault_model']['critical_monitored']['recall']:.3f} | "
            f"{data['by_fault_model']['uniform_random']['recall']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    sizes: list[int],
    sessions: int,
    iterations: int,
    warmup: int,
    sampling_periods: list[int],
    row_samples: int,
    col_samples: int,
    faults_per_matrix: int,
    fault_scale: float,
    fault_abs_min: float,
    residual_scale: float,
    residual_margin: float,
    residual_floor: float,
    projection_count: int,
    correctness_threshold: float,
    seed: int,
) -> dict[str, Any]:
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )

    selector = ProductionKernelSelector()
    modes: list[dict[str, Any]] = []
    for period in sampling_periods:
        modes.append(
            _run_mode(
                mode_period=period,
                sizes=sizes,
                sessions=sessions,
                iterations=iterations,
                warmup=warmup,
                row_samples=row_samples,
                col_samples=col_samples,
                faults_per_matrix=faults_per_matrix,
                fault_scale=fault_scale,
                fault_abs_min=fault_abs_min,
                residual_scale=residual_scale,
                residual_margin=residual_margin,
                residual_floor=residual_floor,
                projection_count=projection_count,
                correctness_threshold=correctness_threshold,
                seed=seed,
                selector=selector,
                ctx=ctx,
                queue=queue,
            )
        )

    recommended = _pick_recommended_mode(modes)
    passes = [m for m in modes if m["mode_pass"]]
    stop_modes = [m for m in modes if m["stop_rule_triggered"]]

    if passes:
        decision_hint = "iterate"
        rationale = (
            "ABFT-lite detect-only achieves critical and uniform recall targets with low overhead "
            "in validated periodic mode; continue toward integration hardening."
        )
    elif len(stop_modes) == len(modes):
        decision_hint = "drop"
        rationale = (
            "All tested modes trigger stop rule (overhead > 8% with insufficient fault recall)."
        )
    else:
        decision_hint = "refine"
        rationale = (
            "Prototype is functional but current mode set does not fully satisfy ABFT target gates."
        )

    stop_rule_triggered = len(stop_modes) == len(modes)
    stop_reason = (
        "all modes exceeded overhead-without-reliability threshold"
        if stop_rule_triggered
        else "not_triggered"
    )

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": sizes,
            "sessions": sessions,
            "iterations": iterations,
            "warmup": warmup,
            "sampling_periods": sampling_periods,
            "row_samples": row_samples,
            "col_samples": col_samples,
            "faults_per_matrix": faults_per_matrix,
            "fault_scale": fault_scale,
            "fault_abs_min": fault_abs_min,
            "fault_models": ["critical_monitored", "uniform_random"],
            "residual_scale": residual_scale,
            "residual_margin": residual_margin,
            "residual_floor": residual_floor,
            "projection_count": projection_count,
            "correctness_threshold": correctness_threshold,
            "seed": seed,
            "opencl_platform": platform.name,
            "opencl_device": device.name,
        },
        "modes": modes,
        "summary": {
            "recommended_mode": recommended["mode"],
            "recommended_mode_details": recommended,
            "any_mode_passed": len(passes) > 0,
            "passed_modes": [m["mode"] for m in passes],
            "stop_rule_triggered": stop_rule_triggered,
            "stop_rule_reason": stop_reason,
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T5 ABFT-lite detect-only campaign")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sampling-periods", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--row-samples", type=int, default=16)
    parser.add_argument("--col-samples", type=int, default=16)
    parser.add_argument("--faults-per-matrix", type=int, default=2)
    parser.add_argument("--fault-scale", type=float, default=0.05)
    parser.add_argument("--fault-abs-min", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=5.0)
    parser.add_argument("--residual-margin", type=float, default=1e-2)
    parser.add_argument("--residual-floor", type=float, default=5e-2)
    parser.add_argument("--projection-count", type=int, default=4)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t5_reliability_abft",
    )
    args = parser.parse_args()

    report = run_experiment(
        sizes=args.sizes,
        sessions=args.sessions,
        iterations=args.iterations,
        warmup=args.warmup,
        sampling_periods=args.sampling_periods,
        row_samples=args.row_samples,
        col_samples=args.col_samples,
        faults_per_matrix=args.faults_per_matrix,
        fault_scale=args.fault_scale,
        fault_abs_min=args.fault_abs_min,
        residual_scale=args.residual_scale,
        residual_margin=args.residual_margin,
        residual_floor=args.residual_floor,
        projection_count=args.projection_count,
        correctness_threshold=args.correctness_threshold,
        seed=args.seed,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week4_t5_abft_detect_only_{timestamp}.json"
    md_path = output_dir / f"week4_t5_abft_detect_only_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T5 ABFT JSON: {json_path}")
    print(f"T5 ABFT MD:   {md_path}")
    print(f"Decision hint: {report['summary']['decision_hint']}")
    print(f"Recommended mode: {report['summary']['recommended_mode']}")
    print(f"Stop rule: {report['summary']['stop_rule_triggered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
