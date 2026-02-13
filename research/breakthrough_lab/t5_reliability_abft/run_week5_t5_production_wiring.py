#!/usr/bin/env python3
"""Week 5 Block 3 - T5 production wiring with auto-disable guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarking.production_kernel_benchmark import run_production_benchmark


def _load_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = ["policy_id", "scope", "runtime_guardrails", "stress_evidence"]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"T5 policy missing required fields: {missing}")
    return data


def _evaluate_guardrails(
    *,
    report: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    guardrails = policy["runtime_guardrails"]
    evidence = policy.get("stress_evidence", {})
    summary = report["summary"]

    checks = {
        "false_positive_rate": {
            "observed": float(summary["false_positive_rate"]),
            "threshold": float(guardrails["disable_if_false_positive_rate_gt"]),
            "comparator": "<=",
        },
        "effective_overhead_percent": {
            "observed": float(summary["effective_overhead_percent"]),
            "threshold": float(guardrails["disable_if_effective_overhead_percent_gt"]),
            "comparator": "<=",
        },
        "correctness_error": {
            "observed": float(summary["max_error"]),
            "threshold": float(guardrails["disable_if_correctness_error_gt"]),
            "comparator": "<=",
        },
        "uniform_recall_reference": {
            "observed": float(evidence.get("uniform_recall", 0.0)),
            "threshold": float(guardrails["disable_if_uniform_recall_lt"]),
            "comparator": ">=",
        },
        "critical_recall_reference": {
            "observed": float(evidence.get("critical_recall", 0.0)),
            "threshold": float(guardrails["disable_if_critical_recall_lt"]),
            "comparator": ">=",
        },
    }
    for payload in checks.values():
        if payload["comparator"] == "<=":
            payload["pass"] = bool(payload["observed"] <= payload["threshold"])
        else:
            payload["pass"] = bool(payload["observed"] >= payload["threshold"])

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    return {
        "checks": checks,
        "all_passed": len(failed) == 0,
        "failed_guardrails": failed,
        "disable_signal": len(failed) > 0,
        "fallback_action": (
            "auto_disable_t5_abft_runtime"
            if len(failed) > 0
            else "keep_t5_abft_runtime_guarded"
        ),
    }


def _decision(report: dict[str, Any]) -> tuple[str, str]:
    guardrail_eval = report["guardrails"]
    summary = report["summary"]
    if int(summary["disable_events"]) > 0:
        return (
            "iterate",
            "Auto-disable triggered during controlled wiring; keep fallback path and refine thresholds.",
        )
    if not guardrail_eval["all_passed"]:
        return (
            "iterate",
            "One or more T5 guardrails failed in production wiring rerun.",
        )
    return (
        "promote",
        "T5 guarded runtime wiring passed all guardrails with zero auto-disable events.",
    )


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T5 Week 5 Block 3 - Production Wiring Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['metadata']['policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions/size={report['metadata']['sessions']} | Iterations/session={report['metadata']['iterations']} | Seed={report['metadata']['seed']}"
    )
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- Kernel avg GFLOPS mean: {report['summary']['kernel_avg_gflops_mean']:.3f}")
    lines.append(
        f"- Effective overhead percent: {report['summary']['effective_overhead_percent']:.3f}"
    )
    lines.append(f"- False positive rate: {report['summary']['false_positive_rate']:.3f}")
    lines.append(f"- Correctness max error: {report['summary']['max_error']:.7f}")
    lines.append(f"- Disable events: {report['summary']['disable_events']}")
    lines.append("")
    lines.append("## Per-Size")
    lines.append("")
    lines.append(
        "| Size | Kernel Avg GFLOPS | Overhead % | False Pos Rate | Max Error | Disable Events |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in report["per_size"]:
        lines.append(
            f"| {item['size']} | {item['kernel_avg_gflops_mean']:.3f} | {item['effective_overhead_percent']:.3f} | {item['false_positive_rate']:.3f} | {item['max_error']:.7f} | {item['disable_events']} |"
        )
    lines.append("")
    lines.append("## Guardrail Evaluation")
    lines.append("")
    lines.append("| Guardrail | Observed | Threshold | Comparator | Pass |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for name, payload in report["guardrails"]["checks"].items():
        lines.append(
            f"| {name} | {payload['observed']:.6f} | {payload['threshold']:.6f} | {payload['comparator']} | {payload['pass']} |"
        )
    lines.append("")
    lines.append(f"- Disable signal: {report['guardrails']['disable_signal']}")
    lines.append(f"- Fallback action: `{report['guardrails']['fallback_action']}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_wiring(
    *,
    policy_path: Path,
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    sizes = [int(x) for x in policy["scope"]["sizes"]]
    per_size: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    for idx, size in enumerate(sizes):
        size_seed = seed + idx * 10_000
        result = run_production_benchmark(
            size=size,
            iterations=iterations,
            sessions=sessions,
            kernel="auto_t5_guarded",
            seed=size_seed,
            t5_policy_path=str(policy_path),
        )
        t5 = result["summary"]["t5_abft"]
        per_size.append(
            {
                "size": int(size),
                "kernel_avg_gflops_mean": float(result["summary"]["avg_gflops"]["mean"]),
                "effective_overhead_percent": float(t5["effective_overhead_percent"]),
                "false_positive_rate": float(t5["false_positive_rate"]),
                "max_error": float(result["summary"]["max_error"]["max"]),
                "disable_events": int(t5["disable_events"]),
            }
        )
        detailed.append(
            {
                "size": int(size),
                "seed": int(size_seed),
                "report": result,
            }
        )

    total_disable_events = int(sum(item["disable_events"] for item in per_size))
    mean_kernel_gflops = float(
        sum(item["kernel_avg_gflops_mean"] for item in per_size) / max(1, len(per_size))
    )
    mean_overhead = float(
        sum(item["effective_overhead_percent"] for item in per_size) / max(1, len(per_size))
    )
    mean_fp_rate = float(
        sum(item["false_positive_rate"] for item in per_size) / max(1, len(per_size))
    )
    max_error = float(max(item["max_error"] for item in per_size)) if per_size else 0.0

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_path": str(policy_path),
            "policy_id": str(policy["policy_id"]),
            "sizes": sizes,
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "per_size": per_size,
        "summary": {
            "kernel_avg_gflops_mean": mean_kernel_gflops,
            "effective_overhead_percent": mean_overhead,
            "false_positive_rate": mean_fp_rate,
            "max_error": max_error,
            "disable_events": total_disable_events,
        },
        "details": detailed,
    }
    report["guardrails"] = _evaluate_guardrails(report=report, policy=policy)
    decision, rationale = _decision(report)
    report["decision"] = {"decision": decision, "rationale": rationale}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T5 Week 5 Block 3 production wiring.")
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json",
    )
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t5_reliability_abft",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    policy_path = (repo_root / args.policy_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_wiring(
        policy_path=policy_path,
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week5_t5_production_wiring_{timestamp}.json"
    md_path = output_dir / f"week5_t5_production_wiring_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"T5 wiring JSON: {json_path}")
    print(f"T5 wiring MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(f"Guardrails passed: {report['guardrails']['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
