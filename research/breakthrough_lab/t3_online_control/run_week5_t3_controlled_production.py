#!/usr/bin/env python3
"""Week 5 Block 1 runner for T3 controlled production integration.

Runs strict deterministic comparisons between static production auto mode and
the opt-in controlled online policy (`auto_t3_controlled`), then emits a formal
decision artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarking.production_kernel_benchmark import run_production_benchmark


def _load_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = ["policy_id", "scope", "promotion_gate"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Policy missing required fields: {missing}")
    return data


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


def _decision(report: dict[str, Any], gate: dict[str, float]) -> tuple[str, str]:
    summary = report["summary"]
    if int(summary["correctness_failures"]) > int(gate["max_correctness_failures"]):
        return (
            "drop",
            "Correctness guard failed in controlled mode; rollout must be cut.",
        )
    if float(summary["fallback_rate"]) > float(gate["max_fallback_rate"]):
        return (
            "iterate",
            "Fallback rate exceeded gate; keep static path and refine policy.",
        )
    if float(summary["p95_latency_delta_percent"]) > float(gate["max_p95_latency_delta_percent"]):
        return (
            "iterate",
            "Latency gate failed; controlled policy needs refinement before promotion.",
        )
    if float(summary["delta_vs_static_percent"]) < float(gate["min_uplift_percent"]):
        return (
            "iterate",
            "Uplift gate not reached; maintain controlled rollout and gather more data.",
        )
    return (
        "promote",
        "Controlled mode passed uplift, latency, fallback and correctness gates.",
    )


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T3 Week 5 Block 1 - Controlled Production Integration Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['metadata']['policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']} | Seed={report['metadata']['seed']}"
    )
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- Static avg GFLOPS mean: {report['summary']['static_avg_gflops']['mean']:.3f}")
    lines.append(
        f"- Controlled avg GFLOPS mean: {report['summary']['controlled_avg_gflops']['mean']:.3f}"
    )
    lines.append(f"- Delta vs static: {report['summary']['delta_vs_static_percent']:+.3f}%")
    lines.append(f"- P95 latency delta: {report['summary']['p95_latency_delta_percent']:+.3f}%")
    lines.append(f"- Fallback rate: {report['summary']['fallback_rate']:.3f}")
    lines.append(f"- Correctness failures: {report['summary']['correctness_failures']}")
    lines.append(f"- Disable events: {report['summary']['disable_events']}")
    lines.append("")
    lines.append("## Per-Size Summary")
    lines.append("")
    lines.append(
        "| Size | Static Avg GFLOPS | Controlled Avg GFLOPS | Delta | Fallback Rate | Correctness Fails |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["per_size"]:
        lines.append(
            f"| {row['size']} | {row['static_avg_gflops']:.3f} | {row['controlled_avg_gflops']:.3f} | {row['delta_vs_static_percent']:+.3f}% | {row['fallback_rate']:.3f} | {row['correctness_failures']} |"
        )
    lines.append("")
    lines.append("## Gate Evaluation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["gate_evaluation"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    policy_path: Path,
    sizes: list[int],
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    gate = policy["promotion_gate"]
    scope_sizes = [int(x) for x in sizes]

    per_size_rows: list[dict[str, Any]] = []
    detailed_reports: list[dict[str, Any]] = []

    for idx, size in enumerate(scope_sizes):
        size_seed = seed + idx * 10000
        run_report = run_production_benchmark(
            size=size,
            iterations=iterations,
            sessions=sessions,
            kernel="auto_t3_controlled",
            seed=size_seed,
            t3_policy_path=str(policy_path),
        )

        summary = run_report["summary"]
        static = summary["static_reference"]
        decisions = run_report.get("decisions", [])
        correctness_failures = sum(
            1 for d in decisions if str(d.get("fallback_reason")) == "correctness"
        )
        disable_events = sum(1 for d in decisions if bool(d.get("disable_signal", False)))

        per_size_rows.append(
            {
                "size": int(size),
                "static_avg_gflops": float(static["avg_gflops"]["mean"]),
                "controlled_avg_gflops": float(summary["avg_gflops"]["mean"]),
                "static_p95_time_ms": float(static["time_ms"]["p95"]),
                "controlled_p95_time_ms": float(summary["time_ms"]["p95"]),
                "delta_vs_static_percent": float(summary["delta_vs_static_percent"]),
                "fallback_rate": float(summary["fallback_rate"]),
                "fallback_count": int(summary["fallback_count"]),
                "correctness_failures": int(correctness_failures),
                "disable_events": int(disable_events),
                "policy_disabled": bool(summary["policy_disabled"]),
            }
        )
        detailed_reports.append(
            {
                "size": int(size),
                "seed": int(size_seed),
                "report": run_report,
            }
        )

    static_avgs = [float(r["static_avg_gflops"]) for r in per_size_rows]
    controlled_avgs = [float(r["controlled_avg_gflops"]) for r in per_size_rows]
    static_p95 = [float(r["static_p95_time_ms"]) for r in per_size_rows]
    controlled_p95 = [float(r["controlled_p95_time_ms"]) for r in per_size_rows]

    static_mean = float(np.mean(static_avgs)) if static_avgs else 0.0
    controlled_mean = float(np.mean(controlled_avgs)) if controlled_avgs else 0.0
    static_p95_mean = float(np.mean(static_p95)) if static_p95 else 0.0
    controlled_p95_mean = float(np.mean(controlled_p95)) if controlled_p95 else 0.0

    delta_vs_static = (
        ((controlled_mean - static_mean) / static_mean * 100.0) if static_mean > 0 else 0.0
    )
    p95_latency_delta = (
        ((controlled_p95_mean - static_p95_mean) / static_p95_mean * 100.0)
        if static_p95_mean > 0
        else 0.0
    )
    total_fallback_count = int(sum(int(r["fallback_count"]) for r in per_size_rows))
    total_sessions = int(sessions * len(scope_sizes))
    fallback_rate = float(total_fallback_count / max(1, total_sessions))
    correctness_failures = int(sum(int(r["correctness_failures"]) for r in per_size_rows))
    disable_events = int(sum(int(r["disable_events"]) for r in per_size_rows))

    gate_eval = {
        "min_uplift_percent": {
            "threshold": float(gate["min_uplift_percent"]),
            "observed": float(delta_vs_static),
            "comparator": ">=",
            "pass": bool(delta_vs_static >= float(gate["min_uplift_percent"])),
        },
        "max_p95_latency_delta_percent": {
            "threshold": float(gate["max_p95_latency_delta_percent"]),
            "observed": float(p95_latency_delta),
            "comparator": "<=",
            "pass": bool(p95_latency_delta <= float(gate["max_p95_latency_delta_percent"])),
        },
        "max_fallback_rate": {
            "threshold": float(gate["max_fallback_rate"]),
            "observed": float(fallback_rate),
            "comparator": "<=",
            "pass": bool(fallback_rate <= float(gate["max_fallback_rate"])),
        },
        "max_correctness_failures": {
            "threshold": int(gate["max_correctness_failures"]),
            "observed": int(correctness_failures),
            "comparator": "<=",
            "pass": bool(correctness_failures <= int(gate["max_correctness_failures"])),
        },
    }

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_path": str(policy_path),
            "policy_id": str(policy["policy_id"]),
            "sizes": scope_sizes,
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
            "opencl_platform": detailed_reports[0]["report"]["metadata"]["platform"],
            "opencl_device": detailed_reports[0]["report"]["metadata"]["device"],
        },
        "per_size": per_size_rows,
        "summary": {
            "static_avg_gflops": _stats(static_avgs),
            "controlled_avg_gflops": _stats(controlled_avgs),
            "delta_vs_static_percent": float(delta_vs_static),
            "static_p95_time_ms": _stats(static_p95),
            "controlled_p95_time_ms": _stats(controlled_p95),
            "p95_latency_delta_percent": float(p95_latency_delta),
            "fallback_count": int(total_fallback_count),
            "fallback_rate": float(fallback_rate),
            "correctness_failures": int(correctness_failures),
            "disable_events": int(disable_events),
        },
        "gate_evaluation": gate_eval,
        "details": detailed_reports,
    }
    decision, rationale = _decision(report, gate)
    report["decision"] = {
        "decision": decision,
        "rationale": rationale,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week 5 Block 1 T3 controlled production integration campaign."
    )
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/t3_online_control/policy_controlled_block1.json",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1200, 1280, 1400, 1536, 1600, 1792, 1920, 2048],
    )
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t3_online_control",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    policy_path = (repo_root / args.policy_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_campaign(
        policy_path=policy_path,
        sizes=list(args.sizes),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week5_t3_controlled_production_{timestamp}.json"
    md_path = output_dir / f"week5_t3_controlled_production_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T3 controlled JSON: {json_path}")
    print(f"T3 controlled MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(f"Delta vs static: {report['summary']['delta_vs_static_percent']:+.3f}%")
    print(f"Fallback rate: {report['summary']['fallback_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
