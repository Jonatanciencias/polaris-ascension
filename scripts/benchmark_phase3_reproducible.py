#!/usr/bin/env python3
"""
Phase 3 reproducible performance benchmark.

Measures production kernels with fixed seed and repeated sessions to produce a
stable baseline (mean, std, min/max, p95).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pyopencl as cl

# Make project root importable when invoked as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.reporting import markdown_table, report_paths, save_markdown_report
from test_production_system import benchmark_kernel


KernelCase = Tuple[str, int, int, int, str, str, Tuple[int, int]]


def _build_cases() -> List[KernelCase]:
    tile20 = Path("src/kernels/gemm_tile20_production.cl").read_text()
    tile24 = Path("src/kernels/gemm_tile24_production.cl").read_text()
    return [
        ("1400x1400", 1400, 1400, 1400, tile20, "gemm_tile20_optimized", (10, 10)),
        ("2048x2048", 2048, 2048, 2048, tile24, "gemm_tile24_vectorized", (12, 12)),
        ("512x512", 512, 512, 512, tile24, "gemm_tile24_vectorized", (12, 12)),
    ]


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def run(sessions: int, iterations: int) -> Dict[str, object]:
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    acc: Dict[str, Dict[str, List[float]]] = {}
    for name, *_ in _build_cases():
        acc[name] = {"peak": [], "avg": [], "time_ms": []}

    for _ in range(sessions):
        for name, M, N, K, src, kernel_name, local_size in _build_cases():
            result = benchmark_kernel(
                ctx,
                queue,
                src,
                kernel_name,
                M,
                N,
                K,
                local_size,
                iterations=iterations,
            )
            if not result["success"]:
                raise RuntimeError(f"Benchmark failed for {name}: {result['error']}")
            acc[name]["peak"].append(float(result["gflops_peak"]))
            acc[name]["avg"].append(float(result["gflops_avg"]))
            acc[name]["time_ms"].append(float(result["time_ms"]))

    summary = {}
    for name, bucket in acc.items():
        summary[name] = {
            "peak": _stats(bucket["peak"]),
            "avg": _stats(bucket["avg"]),
            "time_ms": _stats(bucket["time_ms"]),
        }

    return {
        "metadata": {
            "sessions": sessions,
            "iterations_per_session": iterations,
            "seed": 42,
        },
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 reproducible benchmark runner")
    parser.add_argument("--sessions", type=int, default=10, help="Number of repeated sessions")
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Kernel benchmark iterations per session",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path to write JSON report (optional). If omitted, auto timestamp path is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_reports",
        help="Output directory when --output-json is omitted",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="phase3_reproducible_baseline",
        help="Filename prefix when --output-json is omitted",
    )
    args = parser.parse_args()

    report = run(sessions=args.sessions, iterations=args.iterations)
    if args.output_json:
        output_path = Path(args.output_json)
        md_path = output_path.with_suffix(".md")
    else:
        output_path, md_path = report_paths(prefix=args.prefix, output_dir=args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print("Phase 3 reproducible baseline:")
    for name, data in report["summary"].items():
        peak = data["peak"]
        avg = data["avg"]
        print(
            f"  {name:>9} | peak mean {peak['mean']:.1f} GFLOPS "
            f"[{peak['min']:.1f}, {peak['max']:.1f}] | avg mean {avg['mean']:.1f}"
        )
    print(f"\nJSON report written to: {output_path}")

    md_rows = []
    for name, data in report["summary"].items():
        peak = data["peak"]
        avg = data["avg"]
        md_rows.append(
            (
                name,
                f"{peak['mean']:.1f}",
                f"[{peak['min']:.1f}, {peak['max']:.1f}]",
                f"{avg['mean']:.1f}",
                f"{data['time_ms']['mean']:.3f}",
            )
        )
    md = (
        "# Phase 3 Reproducible Benchmark Report\n\n"
        + markdown_table(
            headers=["Size", "Peak mean GFLOPS", "Peak range", "Avg mean GFLOPS", "Kernel ms mean"],
            rows=md_rows,
        )
        + "\n"
    )
    save_markdown_report(md_path, md)
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
