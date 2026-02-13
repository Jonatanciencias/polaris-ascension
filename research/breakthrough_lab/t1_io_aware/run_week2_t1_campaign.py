#!/usr/bin/env python3
"""
Week 2 T1 campaign runner.

Runs baseline + three IO-aware variants with 10x20 protocol on sizes
1400/2048/3072 and writes JSON + Markdown evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kernel_file: str
    kernel_name: str
    local_size: tuple[int, int]
    tile_size: int
    category: str


DEFAULT_SIZE_WEIGHTS: dict[int, float] = {
    1400: 0.2,
    2048: 0.4,
    3072: 0.4,
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


def _benchmark_once(
    *,
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

    elapsed_s: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        queue.finish()
        elapsed_s.append(time.perf_counter() - start)

    c_gpu = np.empty_like(C)
    cl.enqueue_copy(queue, c_gpu, c_buf).wait()
    max_error = float(np.max(np.abs(c_gpu - (A @ B))))

    flops = float(2 * size * size * size)
    min_time_s = float(np.min(elapsed_s))
    avg_time_s = float(np.mean(elapsed_s))
    return {
        "peak_gflops": flops / min_time_s / 1e9,
        "avg_gflops": flops / avg_time_s / 1e9,
        "time_ms": min_time_s * 1000.0,
        "max_error": max_error,
    }


def _baseline_for_size(size: int) -> KernelSpec:
    if size < 1800:
        return KernelSpec(
            name="baseline_production_tile20",
            kernel_file="src/kernels/gemm_tile20_production.cl",
            kernel_name="gemm_tile20_optimized",
            local_size=(10, 10),
            tile_size=20,
            category="baseline",
        )
    return KernelSpec(
        name="baseline_production_tile24",
        kernel_file="src/kernels/gemm_tile24_production.cl",
        kernel_name="gemm_tile24_vectorized",
        local_size=(12, 12),
        tile_size=24,
        category="baseline",
    )


def _hybrid_variant_for_size(size: int) -> KernelSpec:
    # Large-size strategy: keep IO prefetch at 1400 and switch to tile24 path
    # for 2048/3072 to avoid known large-size regressions.
    if size < 1800:
        return KernelSpec(
            name="io_hybrid_sizeaware_v1",
            kernel_file="research/breakthrough_lab/t1_io_aware/kernels/gemm_tile20_io_prefetch.cl",
            kernel_name="gemm_tile20_io_prefetch",
            local_size=(10, 10),
            tile_size=20,
            category="variant",
        )
    return KernelSpec(
        name="io_hybrid_sizeaware_v1",
        kernel_file="src/kernels/gemm_tile24_production.cl",
        kernel_name="gemm_tile24_vectorized",
        local_size=(12, 12),
        tile_size=24,
        category="variant",
    )


def _variants_for_size(size: int) -> list[KernelSpec]:
    return [
        KernelSpec(
            name="io_prefetch_v1",
            kernel_file="research/breakthrough_lab/t1_io_aware/kernels/gemm_tile20_io_prefetch.cl",
            kernel_name="gemm_tile20_io_prefetch",
            local_size=(10, 10),
            tile_size=20,
            category="variant",
        ),
        KernelSpec(
            name="io_regblock_v1",
            kernel_file="research/breakthrough_lab/t1_io_aware/kernels/gemm_tile20_io_regblock.cl",
            kernel_name="gemm_tile20_io_regblock",
            local_size=(5, 5),
            tile_size=20,
            category="variant",
        ),
        _hybrid_variant_for_size(size),
    ]


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T1 Week 2 IO-Aware Campaign")
    lines.append("")
    lines.append(
        f"- Date: {report['metadata']['timestamp_utc']}"
    )
    lines.append(
        f"- Protocol: sessions={report['metadata']['sessions']}, iterations={report['metadata']['iterations']}, seed={report['metadata']['seed']}"
    )
    lines.append(
        f"- Weighted objective: {report['summary']['size_weights']}"
    )
    lines.append(
        f"- Device: {report['environment']['platform']} / {report['environment']['device']}"
    )
    lines.append("")
    lines.append("## Per-size Summary")
    lines.append("")
    lines.append(
        "| Size | Candidate | Peak Mean GFLOPS | Avg Mean GFLOPS | CV Peak | Max Error Mean | Delta vs Baseline |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    for size in report["sizes"]:
        baseline_peak = report["results"][str(size)]["baseline"]["peak_gflops"]["mean"]
        for item in report["results"][str(size)]["candidates"]:
            peak = item["peak_gflops"]["mean"]
            delta = ((peak - baseline_peak) / baseline_peak * 100.0) if baseline_peak else 0.0
            lines.append(
                f"| {size} | {item['name']} | {peak:.3f} | {item['avg_gflops']['mean']:.3f} | {item['cv_peak']:.5f} | {item['max_error']['mean']:.6f} | {delta:+.3f}% |"
            )

    lines.append("")
    lines.append("## Decision Hint")
    lines.append("")
    best = report["summary"]["best_variant"]
    if best is None:
        lines.append("- No valid variant completed all required runs.")
    else:
        lines.append(
            f"- Best variant by weighted delta: `{best['name']}` ({best['weighted_delta_percent']:+.3f}%)."
        )
        lines.append(
            f"- Mean delta (unweighted): `{best['mean_delta_percent']:+.3f}%`."
        )
        lines.append(
            f"- Correctness pass (<=1e-3) on all sizes: `{best['correctness_all_sizes']}`."
        )
        lines.append(f"- Stability pass (cv<=0.03) on all sizes: `{best['stability_all_sizes']}`.")
    stop_rule = report["summary"]["stop_rule"]
    lines.append(
        f"- Stop rule triggered: `{stop_rule['triggered']}` (min +5% on 1400 or 2048 after 3 variants)."
    )
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    repo_root: Path,
    sizes: list[int],
    sessions: int,
    iterations: int,
    seed: int,
    size_weights: dict[int, float] | None = None,
) -> dict[str, Any]:
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    results: dict[str, Any] = {}
    weights = {
        size: float(DEFAULT_SIZE_WEIGHTS.get(size, 1.0))
        for size in sizes
    }
    if size_weights:
        for size in sizes:
            if size in size_weights:
                weights[size] = float(size_weights[size])

    variant_summary: dict[str, dict[str, Any]] = {}

    for size in sizes:
        baseline_spec = _baseline_for_size(size)
        variants = _variants_for_size(size)
        specs = [baseline_spec] + variants
        per_size: dict[str, Any] = {"baseline": {}, "candidates": []}

        for spec in specs:
            kernel_path = repo_root / spec.kernel_file
            if not kernel_path.exists():
                entry = {
                    "name": spec.name,
                    "category": spec.category,
                    "status": "failed",
                    "error": f"missing kernel file: {kernel_path}",
                    "kernel_file": spec.kernel_file,
                    "kernel_name": spec.kernel_name,
                    "local_size": list(spec.local_size),
                    "tile_size": spec.tile_size,
                }
                if spec.category == "baseline":
                    per_size["baseline"] = entry
                else:
                    per_size["candidates"].append(entry)
                continue

            try:
                source = kernel_path.read_text()
                program = _build_program(ctx, source)
                kernel_obj = getattr(program, spec.kernel_name)
            except Exception as exc:
                entry = {
                    "name": spec.name,
                    "category": spec.category,
                    "status": "failed",
                    "error": str(exc),
                    "kernel_file": spec.kernel_file,
                    "kernel_name": spec.kernel_name,
                    "local_size": list(spec.local_size),
                    "tile_size": spec.tile_size,
                }
                if spec.category == "baseline":
                    per_size["baseline"] = entry
                else:
                    per_size["candidates"].append(entry)
                continue

            peak: list[float] = []
            avg: list[float] = []
            time_ms: list[float] = []
            errors: list[float] = []
            for session_idx in range(sessions):
                metrics = _benchmark_once(
                    queue=queue,
                    kernel=kernel_obj,
                    size=size,
                    tile_size=spec.tile_size,
                    local_size=spec.local_size,
                    iterations=iterations,
                    seed=seed + session_idx,
                )
                peak.append(metrics["peak_gflops"])
                avg.append(metrics["avg_gflops"])
                time_ms.append(metrics["time_ms"])
                errors.append(metrics["max_error"])

            entry = {
                "name": spec.name,
                "category": spec.category,
                "status": "completed",
                "kernel_file": spec.kernel_file,
                "kernel_name": spec.kernel_name,
                "local_size": list(spec.local_size),
                "tile_size": spec.tile_size,
                "peak_gflops": _stats(peak),
                "avg_gflops": _stats(avg),
                "time_ms": _stats(time_ms),
                "max_error": _stats(errors),
                "cv_peak": float(np.std(peak) / np.mean(peak)) if np.mean(peak) > 0 else 1.0,
            }

            if spec.category == "baseline":
                per_size["baseline"] = entry
            else:
                per_size["candidates"].append(entry)
                variant_summary.setdefault(
                    spec.name,
                    {
                        "name": spec.name,
                        "size_deltas": {},
                        "correctness_all_sizes": True,
                        "stability_all_sizes": True,
                    },
                )

        baseline_peak = (
            per_size["baseline"].get("peak_gflops", {}).get("mean")
            if per_size["baseline"].get("status") == "completed"
            else None
        )
        for candidate in per_size["candidates"]:
            if candidate.get("status") != "completed" or baseline_peak is None:
                candidate["delta_vs_baseline_percent"] = None
                continue
            delta = (candidate["peak_gflops"]["mean"] - baseline_peak) / baseline_peak * 100.0
            candidate["delta_vs_baseline_percent"] = float(delta)

            summary = variant_summary[candidate["name"]]
            summary["size_deltas"][str(size)] = float(delta)
            if candidate["max_error"]["mean"] > 1e-3:
                summary["correctness_all_sizes"] = False
            if candidate["cv_peak"] > 0.03:
                summary["stability_all_sizes"] = False

        results[str(size)] = per_size

    ranked_variants = []
    for variant in variant_summary.values():
        if not variant["size_deltas"]:
            continue
        deltas = list(variant["size_deltas"].values())
        variant["mean_delta_percent"] = float(np.mean(deltas))
        weighted_sum = 0.0
        weight_sum = 0.0
        for size in sizes:
            size_key = str(size)
            if size_key not in variant["size_deltas"]:
                continue
            w = float(weights.get(size, 1.0))
            weighted_sum += variant["size_deltas"][size_key] * w
            weight_sum += w
        if weight_sum > 0:
            variant["weighted_delta_percent"] = float(weighted_sum / weight_sum)
        else:
            variant["weighted_delta_percent"] = variant["mean_delta_percent"]
        ranked_variants.append(variant)

    ranked_variants.sort(
        key=lambda item: (
            item["weighted_delta_percent"],
            item["mean_delta_percent"],
        ),
        reverse=True,
    )
    best_variant = ranked_variants[0] if ranked_variants else None

    stop_rule_hits: list[dict[str, Any]] = []
    for variant in ranked_variants:
        for target_size in (1400, 2048):
            size_delta = variant["size_deltas"].get(str(target_size))
            if size_delta is not None and size_delta >= 5.0:
                stop_rule_hits.append(
                    {
                        "variant": variant["name"],
                        "size": target_size,
                        "delta_percent": float(size_delta),
                    }
                )

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sessions": sessions,
            "iterations": iterations,
            "seed": seed,
            "protocol": "10x20 canonical benchmark",
        },
        "environment": {
            "platform": platform.name,
            "device": device.name,
            "driver": device.driver_version,
        },
        "sizes": sizes,
        "results": results,
        "summary": {
            "best_variant": best_variant,
            "ranked_variants": ranked_variants,
            "size_weights": {str(size): float(weights[size]) for size in sizes},
            "stop_rule": {
                "required_variants": 3,
                "target_sizes": [1400, 2048],
                "min_delta_percent": 5.0,
                "triggered": len(ranked_variants) >= 3 and len(stop_rule_hits) == 0,
                "hits": stop_rule_hits,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 2 T1 IO-aware campaign")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1400, 2048, 3072])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t1_io_aware",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_campaign(
        repo_root=repo_root,
        sizes=args.sizes,
        sessions=args.sessions,
        iterations=args.iterations,
        seed=args.seed,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week2_t1_io_campaign_{timestamp}.json"
    md_path = output_dir / f"week2_t1_io_campaign_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T1 campaign JSON: {json_path}")
    print(f"T1 campaign MD:   {md_path}")
    best = report["summary"]["best_variant"]
    if best:
        print(
            "Best variant:",
            best["name"],
            f"(weighted delta {best['weighted_delta_percent']:+.3f}%)",
        )
    if report["summary"]["stop_rule"]["triggered"]:
        print("Stop rule: TRIGGERED")
    else:
        print("Stop rule: not triggered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
