#!/usr/bin/env python3
"""Evaluate Week 11 weekly replay artifacts against formal SLO policy."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _key(kernel: str, size: int) -> str:
    return f"{kernel}:{size}"


def _aggregate_runs(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    ok_runs = [r for r in report.get("runs", []) if r.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in ok_runs:
        key = _key(str(run["kernel"]), int(run["size"]))
        grouped.setdefault(key, []).append(run)

    rows: dict[str, dict[str, Any]] = {}
    for key, entries in grouped.items():
        ordered = sorted(entries, key=lambda e: int(e.get("snapshot", 0)))
        avg_values = [float(e["metrics"]["avg_mean_gflops"]) for e in ordered]
        p95_values = [float(e["metrics"]["p95_time_ms"]) for e in ordered]
        first_avg = float(ordered[0]["metrics"]["avg_mean_gflops"])
        last_avg = float(ordered[-1]["metrics"]["avg_mean_gflops"])
        first_p95 = float(ordered[0]["metrics"]["p95_time_ms"])
        last_p95 = float(ordered[-1]["metrics"]["p95_time_ms"])
        drift_throughput = 0.0 if first_avg == 0.0 else (last_avg - first_avg) / first_avg * 100.0
        drift_p95 = 0.0 if first_p95 == 0.0 else (last_p95 - first_p95) / first_p95 * 100.0
        rows[key] = {
            "kernel": str(ordered[0]["kernel"]),
            "size": int(ordered[0]["size"]),
            "samples": int(len(ordered)),
            "avg_gflops_mean": float(statistics.mean(avg_values)),
            "p95_time_ms_mean": float(statistics.mean(p95_values)),
            "throughput_drift_percent": float(drift_throughput),
            "p95_drift_percent": float(drift_p95),
            "t3_fallback_max": float(
                max(float(e["metrics"].get("t3_fallback_rate", 0.0)) for e in ordered)
            ),
            "t5_overhead_max": float(
                max(float(e["metrics"].get("t5_overhead_percent", 0.0)) for e in ordered)
            ),
            "t5_disable_events_total": int(
                sum(int(e["metrics"].get("t5_disable_events", 0)) for e in ordered)
            ),
        }

    summary = {
        "max_correctness_error": float(
            max((float(r["metrics"]["max_error_max"]) for r in ok_runs), default=999.0)
        ),
        "max_t3_fallback_rate": float(
            max(
                (
                    float(r["metrics"].get("t3_fallback_rate", 0.0))
                    for r in ok_runs
                    if str(r.get("kernel")) == "auto_t3_controlled"
                ),
                default=0.0,
            )
        ),
        "max_t5_overhead_percent": float(
            max(
                (
                    float(r["metrics"].get("t5_overhead_percent", 0.0))
                    for r in ok_runs
                    if str(r.get("kernel")) == "auto_t5_guarded"
                ),
                default=0.0,
            )
        ),
        "t5_disable_events_total": int(
            sum(
                int(r["metrics"].get("t5_disable_events", 0))
                for r in ok_runs
                if str(r.get("kernel")) == "auto_t5_guarded"
            )
        ),
    }
    return rows, summary


def _evaluate(report: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    rows, summary = _aggregate_runs(report)
    guardrails = policy["weekly_slo"]["global_guardrails"]
    per_key = policy["weekly_slo"]["per_kernel_size"]
    checks: dict[str, dict[str, Any]] = {}

    executed_snapshots = int(report["metadata"]["executed_snapshots"])
    checks["minimum_snapshots"] = {
        "observed": executed_snapshots,
        "required_min": int(guardrails["minimum_snapshots"]),
        "pass": executed_snapshots >= int(guardrails["minimum_snapshots"]),
    }
    checks["correctness_error_bound"] = {
        "observed_max": float(summary["max_correctness_error"]),
        "required_max": float(guardrails["max_correctness_error"]),
        "pass": float(summary["max_correctness_error"]) <= float(guardrails["max_correctness_error"]),
    }
    checks["t3_fallback_bound"] = {
        "observed_max": float(summary["max_t3_fallback_rate"]),
        "required_max": float(guardrails["max_t3_fallback_rate"]),
        "pass": float(summary["max_t3_fallback_rate"]) <= float(guardrails["max_t3_fallback_rate"]),
    }
    checks["t5_disable_events_total_bound"] = {
        "observed": int(summary["t5_disable_events_total"]),
        "required_max": int(guardrails["max_t5_disable_events_total"]),
        "pass": int(summary["t5_disable_events_total"]) <= int(guardrails["max_t5_disable_events_total"]),
    }
    checks["t5_overhead_bound"] = {
        "observed_max": float(summary["max_t5_overhead_percent"]),
        "required_max": float(guardrails["max_t5_overhead_percent"]),
        "pass": float(summary["max_t5_overhead_percent"]) <= float(guardrails["max_t5_overhead_percent"]),
    }

    missing_slo_rows: list[str] = []
    per_row_checks: list[dict[str, Any]] = []
    for row_key, slo in per_key.items():
        row = rows.get(row_key)
        if row is None:
            missing_slo_rows.append(row_key)
            continue
        avg_pass = float(row["avg_gflops_mean"]) >= float(slo["min_avg_gflops"])
        p95_pass = float(row["p95_time_ms_mean"]) <= float(slo["max_p95_time_ms"])
        per_row_checks.append(
            {
                "key": row_key,
                "observed_avg_gflops": float(row["avg_gflops_mean"]),
                "required_min_avg_gflops": float(slo["min_avg_gflops"]),
                "avg_pass": bool(avg_pass),
                "observed_p95_time_ms": float(row["p95_time_ms_mean"]),
                "required_max_p95_time_ms": float(slo["max_p95_time_ms"]),
                "p95_pass": bool(p95_pass),
                "pass": bool(avg_pass and p95_pass),
            }
        )

    checks["all_policy_rows_present"] = {
        "missing_rows": missing_slo_rows,
        "pass": len(missing_slo_rows) == 0,
    }
    checks["per_kernel_size_slo"] = {
        "rows": per_row_checks,
        "pass": all(bool(r["pass"]) for r in per_row_checks) and len(per_row_checks) > 0,
    }

    max_abs_throughput_drift = max(
        (abs(float(r["throughput_drift_percent"])) for r in rows.values()),
        default=999.0,
    )
    max_positive_p95_drift = max((float(r["p95_drift_percent"]) for r in rows.values()), default=999.0)
    checks["throughput_drift_abs_bound"] = {
        "observed_max_abs_percent": float(max_abs_throughput_drift),
        "required_max_abs_percent": float(guardrails["max_abs_throughput_drift_percent"]),
        "pass": float(max_abs_throughput_drift) <= float(guardrails["max_abs_throughput_drift_percent"]),
    }
    checks["p95_drift_bound"] = {
        "observed_max_percent": float(max_positive_p95_drift),
        "required_max_percent": float(guardrails["max_p95_drift_percent"]),
        "pass": float(max_positive_p95_drift) <= float(guardrails["max_p95_drift_percent"]),
    }

    failed_checks = [name for name, payload in checks.items() if not bool(payload.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds."
        if decision == "promote"
        else "Weekly replay violated one or more formal SLO checks; keep iterate."
    )

    return {
        "summary": summary,
        "rows": rows,
        "checks": checks,
        "failed_checks": failed_checks,
        "decision": decision,
        "rationale": rationale,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 11 Block 4 - Weekly Replay Evaluation")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append(f"- Canary: `{report['metadata']['canary_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = sorted(
        report["evaluation"]["rows"].values(),
        key=lambda r: (str(r["kernel"]), int(r["size"])),
    )
    for row in rows:
        lines.append(
            f"| {row['kernel']} | {row['size']} | {row['avg_gflops_mean']:.3f} | "
            f"{row['p95_time_ms_mean']:.3f} | {row['throughput_drift_percent']:.3f} | "
            f"{row['p95_drift_percent']:.3f} | {row['t3_fallback_max']:.4f} | "
            f"{row['t5_overhead_max']:.4f} | {row['t5_disable_events_total']} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {report['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate weekly canary replay against formal SLO policy.")
    parser.add_argument("--canary-path", required=True, help="Path to weekly canary JSON artifact.")
    parser.add_argument("--policy-path", required=True, help="Path to formal SLO policy JSON.")
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week11_controlled_rollout",
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument("--output-prefix", default="week11_block4_weekly_replay_eval")
    args = parser.parse_args()

    canary_path = Path(args.canary_path).resolve()
    policy_path = Path(args.policy_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canary = _read_json(canary_path)
    policy = _read_json(policy_path)
    evaluation = _evaluate(canary, policy)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "canary_path": str(canary_path),
            "policy_path": str(policy_path),
            "policy_id": policy.get("policy_id"),
        },
        "evaluation": evaluation,
    }
    json_path = output_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_to_markdown(report) + "\n")

    print(f"Weekly replay eval JSON: {json_path}")
    print(f"Weekly replay eval MD:   {md_path}")
    print(f"Decision: {evaluation['decision']}")
    print(f"Failed checks: {evaluation['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
