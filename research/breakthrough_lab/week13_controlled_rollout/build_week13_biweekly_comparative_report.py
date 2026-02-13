#!/usr/bin/env python3
"""Build Week 13 biweekly comparative report from two canary artifacts."""

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


def _mean_rows(canary: dict[str, Any]) -> dict[str, dict[str, float]]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_key(str(run["kernel"]), int(run["size"])), []).append(run)
    out: dict[str, dict[str, float]] = {}
    for key, entries in grouped.items():
        out[key] = {
            "avg_gflops_mean": float(
                statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
            ),
            "p95_time_ms_mean": float(
                statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
            ),
            "max_error_max": float(max(float(e["metrics"]["max_error_max"]) for e in entries)),
            "t3_fallback_max": float(
                max(float(e["metrics"].get("t3_fallback_rate", 0.0)) for e in entries)
            ),
            "t5_overhead_max": float(
                max(float(e["metrics"].get("t5_overhead_percent", 0.0)) for e in entries)
            ),
            "t5_disable_events_total": int(
                sum(int(e["metrics"].get("t5_disable_events", 0)) for e in entries)
            ),
        }
    return out


def _guardrail_summary(canary: dict[str, Any]) -> dict[str, Any]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    return {
        "decision": str(canary.get("evaluation", {}).get("decision")),
        "snapshots": int(canary.get("metadata", {}).get("executed_snapshots", 0)),
        "rollback_triggered": bool(canary.get("rollback_event") is not None),
        "failed_checks": list(canary.get("evaluation", {}).get("failed_checks", [])),
        "max_error": float(max((float(r["metrics"]["max_error_max"]) for r in runs), default=999.0)),
        "t3_fallback_max": float(
            max(
                (
                    float(r["metrics"].get("t3_fallback_rate", 0.0))
                    for r in runs
                    if str(r.get("kernel")) == "auto_t3_controlled"
                ),
                default=0.0,
            )
        ),
        "t5_overhead_max": float(
            max(
                (
                    float(r["metrics"].get("t5_overhead_percent", 0.0))
                    for r in runs
                    if str(r.get("kernel")) == "auto_t5_guarded"
                ),
                default=0.0,
            )
        ),
        "t5_disable_events_total": int(
            sum(
                int(r["metrics"].get("t5_disable_events", 0))
                for r in runs
                if str(r.get("kernel")) == "auto_t5_guarded"
            )
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 1 - Biweekly Comparative Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline canary: `{report['metadata']['baseline_canary']}`")
    lines.append(f"- Current canary: `{report['metadata']['current_canary']}`")
    lines.append("")
    lines.append("## Guardrail Summary")
    lines.append("")
    lines.append("| Window | Decision | Snapshots | Rollback | Max error | T3 fallback max | T5 overhead max % | T5 disable total |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for name in ("baseline", "current"):
        row = report["guardrails"][name]
        lines.append(
            f"| {name} | {row['decision']} | {row['snapshots']} | {row['rollback_triggered']} | "
            f"{row['max_error']:.9f} | {row['t3_fallback_max']:.4f} | {row['t5_overhead_max']:.4f} | {row['t5_disable_events_total']} |"
        )
    lines.append("")
    lines.append("## Per-Key Deltas (current vs baseline)")
    lines.append("")
    lines.append("| Key | Baseline avg GFLOPS | Current avg GFLOPS | Delta % | Baseline p95 ms | Current p95 ms | Delta % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["rows"]:
        lines.append(
            f"| {row['key']} | {row['baseline_avg_gflops']:.3f} | {row['current_avg_gflops']:.3f} | "
            f"{row['avg_gflops_delta_percent']:.3f} | {row['baseline_p95_ms']:.3f} | "
            f"{row['current_p95_ms']:.3f} | {row['p95_delta_percent']:.3f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']}`")
    lines.append(f"- Failed checks: {report['failed_checks']}")
    lines.append(f"- Rationale: {report['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build biweekly comparative report for Week 13.")
    parser.add_argument("--baseline-canary", required=True)
    parser.add_argument("--current-canary", required=True)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week13_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week13_block1_biweekly_comparative")
    parser.add_argument("--max-throughput-drop-percent", type=float, default=3.0)
    parser.add_argument("--max-p95-increase-percent", type=float, default=5.0)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_canary).resolve()
    current_path = Path(args.current_canary).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _read_json(baseline_path)
    current = _read_json(current_path)
    baseline_rows = _mean_rows(baseline)
    current_rows = _mean_rows(current)
    common = sorted(set(baseline_rows).intersection(current_rows))

    rows: list[dict[str, Any]] = []
    for key in common:
        b = baseline_rows[key]
        c = current_rows[key]
        avg_delta = 0.0 if b["avg_gflops_mean"] == 0.0 else (
            (c["avg_gflops_mean"] - b["avg_gflops_mean"]) / b["avg_gflops_mean"] * 100.0
        )
        p95_delta = 0.0 if b["p95_time_ms_mean"] == 0.0 else (
            (c["p95_time_ms_mean"] - b["p95_time_ms_mean"]) / b["p95_time_ms_mean"] * 100.0
        )
        rows.append(
            {
                "key": key,
                "baseline_avg_gflops": float(b["avg_gflops_mean"]),
                "current_avg_gflops": float(c["avg_gflops_mean"]),
                "avg_gflops_delta_percent": float(avg_delta),
                "baseline_p95_ms": float(b["p95_time_ms_mean"]),
                "current_p95_ms": float(c["p95_time_ms_mean"]),
                "p95_delta_percent": float(p95_delta),
            }
        )

    failed_checks: list[str] = []
    if current.get("evaluation", {}).get("decision") != "promote":
        failed_checks.append("current_canary_not_promote")
    if any(float(r["avg_gflops_delta_percent"]) < -float(args.max_throughput_drop_percent) for r in rows):
        failed_checks.append("throughput_drop_exceeds_limit")
    if any(float(r["p95_delta_percent"]) > float(args.max_p95_increase_percent) for r in rows):
        failed_checks.append("p95_increase_exceeds_limit")

    guardrails = {
        "baseline": _guardrail_summary(baseline),
        "current": _guardrail_summary(current),
    }
    if guardrails["current"]["max_error"] > 0.001:
        failed_checks.append("correctness_error_exceeds_limit")
    if guardrails["current"]["t5_disable_events_total"] > 0:
        failed_checks.append("t5_disable_events_nonzero")

    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Extended controlled production remains stable with bounded deltas and healthy guardrails."
        if decision == "promote"
        else "Biweekly comparison detected one or more regressions/guardrail concerns."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_canary": str(baseline_path),
            "current_canary": str(current_path),
            "max_throughput_drop_percent": float(args.max_throughput_drop_percent),
            "max_p95_increase_percent": float(args.max_p95_increase_percent),
        },
        "guardrails": guardrails,
        "rows": rows,
        "decision": decision,
        "failed_checks": failed_checks,
        "rationale": rationale,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report) + "\n")

    print(f"Week13 comparative JSON: {json_path}")
    print(f"Week13 comparative MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
