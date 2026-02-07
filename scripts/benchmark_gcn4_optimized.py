#!/usr/bin/env python3
"""
GCN4-oriented benchmark using current OptimizedKernelEngine API.

Compares:
- GEMM_FLOAT4_VEC (baseline in this script)
- GEMM_GCN4_ULTRA
- GEMM_GCN4_STREAMING

Outputs JSON+Markdown reports in results/benchmark_reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Workaround for pyopencl cache warning in specific runtime versions.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.reporting import markdown_table, report_paths, save_json_report, save_markdown_report
from src.optimization_engines.optimized_kernel_engine import KernelType, OptimizedKernelEngine


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


def _bench_kernel(
    engine: OptimizedKernelEngine,
    kernel_type: KernelType,
    A: np.ndarray,
    B: np.ndarray,
    C_ref: np.ndarray,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        engine.gemm(A, B, kernel_type=kernel_type)

    gflops: list[float] = []
    times_ms: list[float] = []
    rel_errors: list[float] = []
    max_abs_errors: list[float] = []

    for _ in range(runs):
        r = engine.gemm(A, B, kernel_type=kernel_type)
        gflops.append(float(r.kernel_metrics.gflops))
        times_ms.append(float(r.kernel_metrics.exec_time_ms))

        diff = r.result - C_ref
        rel_errors.append(float(np.linalg.norm(diff) / (np.linalg.norm(C_ref) + 1e-12)))
        max_abs_errors.append(float(np.max(np.abs(diff))))

    return {
        "gflops": _stats(gflops),
        "kernel_time_ms": _stats(times_ms),
        "relative_error": _stats(rel_errors),
        "max_abs_error": _stats(max_abs_errors),
    }


def _build_markdown(report: dict[str, Any]) -> str:
    rows = []
    for size_key, kdata in report["results"].items():
        for kernel_name in ["float4_vec", "gcn4_ultra", "gcn4_streaming"]:
            data = kdata[kernel_name]
            rows.append(
                (
                    size_key,
                    kernel_name,
                    f"{data['gflops']['mean']:.1f}",
                    f"{data['gflops']['max']:.1f}",
                    f"{data['kernel_time_ms']['mean']:.3f}",
                    f"{data['relative_error']['mean']:.2e}",
                )
            )

    summary = report["summary"]
    return (
        "# GCN4 Optimized Benchmark Report\n\n"
        + markdown_table(
            headers=["Size", "Kernel", "GFLOPS mean", "GFLOPS max", "Kernel ms mean", "Rel error mean"],
            rows=rows,
        )
        + "\n\n"
        + markdown_table(
            headers=["Metric", "Value"],
            rows=[
                ("Peak kernel", summary["peak_kernel"]),
                ("Peak size", summary["peak_size"]),
                ("Peak GFLOPS", f"{summary['peak_gflops']:.1f}"),
                ("Avg improvement vs float4_vec", f"{summary['avg_improvement_vs_float4']:+.2%}"),
            ],
        )
        + "\n"
    )


def run_suite(
    *,
    sizes: list[int],
    runs: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    engine = OptimizedKernelEngine()
    rng = np.random.default_rng(seed)

    kernels = {
        "float4_vec": KernelType.GEMM_FLOAT4_VEC,
        "gcn4_ultra": KernelType.GEMM_GCN4_ULTRA,
        "gcn4_streaming": KernelType.GEMM_GCN4_STREAMING,
    }

    report: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "runs": runs,
            "warmup": warmup,
            "sizes": sizes,
            "device": engine.device.name,
        },
        "results": {},
        "summary": {},
    }

    all_kernel_peaks: list[tuple[float, str, str]] = []
    improvements: list[float] = []

    print("GCN4 benchmark sweep")
    print("size    kernel          gflops_mean  gflops_peak")

    for size in sizes:
        A = rng.standard_normal((size, size), dtype=np.float32)
        B = rng.standard_normal((size, size), dtype=np.float32)
        C_ref = A @ B

        size_key = f"{size}x{size}"
        report["results"][size_key] = {}

        baseline_mean = None
        for label, kernel_type in kernels.items():
            result = _bench_kernel(
                engine=engine,
                kernel_type=kernel_type,
                A=A,
                B=B,
                C_ref=C_ref,
                warmup=warmup,
                runs=runs,
            )
            report["results"][size_key][label] = result

            mean_gflops = result["gflops"]["mean"]
            peak_gflops = result["gflops"]["max"]
            print(f"{size:<7} {label:<15} {mean_gflops:>11.1f} {peak_gflops:>11.1f}")

            all_kernel_peaks.append((peak_gflops, size_key, label))
            if label == "float4_vec":
                baseline_mean = mean_gflops

        if baseline_mean and baseline_mean > 0:
            best_non_baseline = max(
                report["results"][size_key]["gcn4_ultra"]["gflops"]["mean"],
                report["results"][size_key]["gcn4_streaming"]["gflops"]["mean"],
            )
            improvements.append((best_non_baseline - baseline_mean) / baseline_mean)

    peak_gflops, peak_size, peak_kernel = max(all_kernel_peaks, key=lambda x: x[0])
    report["summary"] = {
        "peak_gflops": float(peak_gflops),
        "peak_size": peak_size,
        "peak_kernel": peak_kernel,
        "avg_improvement_vs_float4": float(np.mean(improvements) if improvements else 0.0),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="GCN4 benchmark on current engine")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/benchmark_reports")
    args = parser.parse_args()

    report = run_suite(
        sizes=args.sizes,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
    )

    json_path, md_path = report_paths(prefix="gcn4_optimized_benchmark", output_dir=args.output_dir)
    save_json_report(json_path, report)
    save_markdown_report(md_path, _build_markdown(report))

    print("\nSummary:")
    print(f"  Peak: {report['summary']['peak_gflops']:.1f} GFLOPS ({report['summary']['peak_kernel']} @ {report['summary']['peak_size']})")
    print(f"  Avg improvement vs float4_vec: {report['summary']['avg_improvement_vs_float4']:+.2%}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
