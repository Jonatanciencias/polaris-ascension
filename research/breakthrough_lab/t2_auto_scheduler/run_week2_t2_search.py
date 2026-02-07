#!/usr/bin/env python3
"""
Week 2 T2 bounded-search runner.

Runs an expanded bounded search and enforces strict correctness filtering
before candidate ranking.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.auto_tuner.gemm_auto_tuner import GEMMAutoTuner


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


def _baseline_kernel_for_size(size: int) -> str:
    return "tile20" if size < 1800 else "tile24"


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T2 Week 2 Bounded Search")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        "- Budget: "
        f"{report['metadata']['kernels']} kernels x {report['metadata']['sizes']} sizes x "
        f"{report['metadata']['runs_per_config']} runs"
    )
    lines.append(f"- Correctness threshold: {report['metadata']['correctness_threshold']}")
    lines.append("")
    lines.append("## Search Results")
    lines.append("")
    lines.append(
        "| Rank | Kernel | Size | GFLOPS | Max Error | Delta vs Baseline | Status |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | --- |")

    for idx, item in enumerate(report["search"]["ranked_candidates"], start=1):
        delta = item["delta_vs_baseline_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
        status = "valid" if item["correctness_passed"] else "filtered"
        lines.append(
            f"| {idx} | {item['kernel']} | {item['size']} | {item['gflops']:.3f} | "
            f"{item['max_error']:.6f} | {delta_text} | {status} |"
        )

    lines.append("")
    lines.append("## Replay Summary")
    lines.append("")
    lines.append(
        "| Candidate | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | --- |")

    for item in report["replay"]["candidates"]:
        delta = item["mean_delta_vs_baseline_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
        lines.append(
            f"| {item['kernel']}@{item['size']} | {item['gflops']['mean']:.3f} | "
            f"{item['cv_peak']:.5f} | {item['max_error']['max']:.6f} | {delta_text} | "
            f"{item['correctness_passed']} | {item['stability_passed']} | {item['promotion_gate_passed']} |"
        )

    lines.append("")
    lines.append("## Decision Hint")
    lines.append("")
    lines.append(f"- Suggested decision: `{report['summary']['decision_hint']}`")
    lines.append(f"- Decision rationale: {report['summary']['decision_rationale']}")
    return "\n".join(lines) + "\n"


def run_search(
    *,
    kernels: list[str],
    sizes: list[int],
    runs_per_config: int,
    warmup: int,
    replay_sessions: int,
    replay_runs: int,
    top_k: int,
    correctness_threshold: float,
    seed: int,
) -> dict[str, Any]:
    tuner = GEMMAutoTuner(output_dir="results/auto_tuner", verbose=False)

    raw_candidates: list[dict[str, Any]] = []
    config_idx = 0
    for kernel in kernels:
        if kernel not in tuner.kernels:
            raw_candidates.append(
                {
                    "kernel": kernel,
                    "size": None,
                    "status": "failed",
                    "error": f"kernel '{kernel}' not available",
                }
            )
            continue

        for size in sizes:
            np.random.seed(seed + config_idx)
            config_idx += 1
            result = tuner.benchmark_kernel(
                kernel_name=kernel,
                M=size,
                N=size,
                K=size,
                runs=runs_per_config,
                warmup=warmup,
            )
            if result is None:
                raw_candidates.append(
                    {
                        "kernel": kernel,
                        "size": size,
                        "status": "failed",
                        "error": "benchmark_kernel returned None",
                    }
                )
                continue

            raw_candidates.append(
                {
                    "kernel": kernel,
                    "size": size,
                    "status": "completed",
                    "gflops": float(result.gflops),
                    "avg_time_ms": float(result.avg_time_ms),
                    "max_error": float(result.max_error),
                    "runs": int(result.runs),
                }
            )

    completed = [item for item in raw_candidates if item.get("status") == "completed"]
    baseline_by_size: dict[int, float] = {}
    for size in sizes:
        baseline_kernel = _baseline_kernel_for_size(size)
        baseline_items = [
            item
            for item in completed
            if item["size"] == size and item["kernel"] == baseline_kernel
        ]
        if baseline_items:
            baseline_by_size[size] = baseline_items[0]["gflops"]

    ranked_input: list[dict[str, Any]] = []
    for item in completed:
        size = int(item["size"])
        baseline = baseline_by_size.get(size)
        delta = None
        if baseline and baseline > 0:
            delta = (item["gflops"] - baseline) / baseline * 100.0

        decorated = {
            **item,
            "correctness_passed": item["max_error"] <= correctness_threshold,
            "delta_vs_baseline_percent": None if delta is None else float(delta),
        }
        ranked_input.append(decorated)

    valid = [item for item in ranked_input if item["correctness_passed"]]
    pool = valid if valid else ranked_input
    ranked = sorted(pool, key=lambda item: item["gflops"], reverse=True)
    top_candidates = ranked[:top_k]

    replay_candidates: list[dict[str, Any]] = []
    for idx, candidate in enumerate(top_candidates):
        gflops_values: list[float] = []
        time_values: list[float] = []
        error_values: list[float] = []

        for session_idx in range(replay_sessions):
            np.random.seed(seed + 10_000 + idx * 100 + session_idx)
            replay_result = tuner.benchmark_kernel(
                kernel_name=candidate["kernel"],
                M=candidate["size"],
                N=candidate["size"],
                K=candidate["size"],
                runs=replay_runs,
                warmup=warmup,
            )
            if replay_result is None:
                continue
            gflops_values.append(float(replay_result.gflops))
            time_values.append(float(replay_result.avg_time_ms))
            error_values.append(float(replay_result.max_error))

        if not gflops_values:
            replay_candidates.append(
                {
                    "kernel": candidate["kernel"],
                    "size": candidate["size"],
                    "status": "failed",
                    "error": "no replay samples",
                }
            )
            continue

        size = int(candidate["size"])
        baseline = baseline_by_size.get(size)
        gflops_mean = float(np.mean(gflops_values))
        delta_mean = None
        if baseline and baseline > 0:
            delta_mean = (gflops_mean - baseline) / baseline * 100.0

        cv_peak = float(np.std(gflops_values) / np.mean(gflops_values))
        max_error_max = float(np.max(error_values))
        correctness_passed = max_error_max <= correctness_threshold
        stability_passed = cv_peak <= 0.03
        promotion_passed = (
            delta_mean is not None
            and delta_mean >= 10.0
            and correctness_passed
            and stability_passed
        )

        replay_candidates.append(
            {
                "kernel": candidate["kernel"],
                "size": candidate["size"],
                "status": "completed",
                "gflops": _stats(gflops_values),
                "avg_time_ms": _stats(time_values),
                "max_error": _stats(error_values),
                "cv_peak": cv_peak,
                "mean_delta_vs_baseline_percent": None if delta_mean is None else float(delta_mean),
                "correctness_passed": correctness_passed,
                "stability_passed": stability_passed,
                "promotion_gate_passed": promotion_passed,
            }
        )

    replay_completed = [item for item in replay_candidates if item.get("status") == "completed"]

    if any(item["promotion_gate_passed"] for item in replay_completed):
        decision_hint = "promote"
        rationale = "At least one replayed candidate passed performance, correctness and stability gates."
    elif not valid:
        decision_hint = "refine"
        rationale = "No candidate passed strict correctness filter in search phase."
    elif any(item["correctness_passed"] and item["stability_passed"] for item in replay_completed):
        decision_hint = "iterate"
        rationale = "Correct/stable candidates exist but promotion uplift threshold is not met."
    else:
        decision_hint = "refine"
        rationale = "Replay did not produce stable and correct candidates under strict filter."

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "kernels": kernels,
            "sizes": sizes,
            "runs_per_config": runs_per_config,
            "warmup": warmup,
            "replay_sessions": replay_sessions,
            "replay_runs": replay_runs,
            "top_k": top_k,
            "correctness_threshold": correctness_threshold,
            "seed": seed,
        },
        "search": {
            "raw_candidates": raw_candidates,
            "baseline_by_size": baseline_by_size,
            "ranked_candidates": ranked,
            "valid_count": len(valid),
            "completed_count": len(completed),
        },
        "replay": {
            "candidates": replay_candidates,
        },
        "summary": {
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 2 T2 bounded search")
    parser.add_argument("--kernels", nargs="+", default=["tile20", "tile24"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--runs-per-config", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replay-sessions", type=int, default=5)
    parser.add_argument("--replay-runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t2_auto_scheduler",
    )
    args = parser.parse_args()

    report = run_search(
        kernels=args.kernels,
        sizes=args.sizes,
        runs_per_config=args.runs_per_config,
        warmup=args.warmup,
        replay_sessions=args.replay_sessions,
        replay_runs=args.replay_runs,
        top_k=args.top_k,
        correctness_threshold=args.correctness_threshold,
        seed=args.seed,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"week2_t2_bounded_search_{timestamp}.json"
    md_path = output_dir / f"week2_t2_bounded_search_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T2 search JSON: {json_path}")
    print(f"T2 search MD:   {md_path}")
    print(f"Decision hint:  {report['summary']['decision_hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
