#!/usr/bin/env python3
"""Week 13 Block 4: build biweekly operational consolidation package."""

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


def _canary_summary(canary: dict[str, Any]) -> dict[str, Any]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    t3_runs = [r for r in runs if str(r.get("kernel")) == "auto_t3_controlled"]
    t5_runs = [r for r in runs if str(r.get("kernel")) == "auto_t5_guarded"]
    return {
        "decision": str(canary.get("evaluation", {}).get("decision", "unknown")),
        "executed_snapshots": int(canary.get("metadata", {}).get("executed_snapshots", 0)),
        "rollback_triggered": canary.get("rollback_event") is not None,
        "failed_checks": list(canary.get("evaluation", {}).get("failed_checks", [])),
        "correctness_max": float(
            max((float(r["metrics"]["max_error_max"]) for r in runs), default=999.0)
        ),
        "t3_fallback_max": float(
            max((float(r["metrics"].get("t3_fallback_rate", 0.0)) for r in t3_runs), default=0.0)
        ),
        "t5_overhead_max": float(
            max((float(r["metrics"].get("t5_overhead_percent", 0.0)) for r in t5_runs), default=0.0)
        ),
        "t5_disable_events_total": int(
            sum(int(r["metrics"].get("t5_disable_events", 0)) for r in t5_runs)
        ),
    }


def _per_key_means(canary: dict[str, Any]) -> dict[str, dict[str, float]]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_key(str(run["kernel"]), int(run["size"])), []).append(run)
    out: dict[str, dict[str, float]] = {}
    for row_key, entries in grouped.items():
        out[row_key] = {
            "avg_gflops_mean": float(
                statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
            ),
            "p95_time_ms_mean": float(
                statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
            ),
        }
    return out


def _split_ratio_min(split_eval: dict[str, Any]) -> float:
    rows = (
        split_eval.get("evaluation", {})
        .get("checks", {})
        .get("rusticl_ratio_vs_clover", {})
        .get("rows", [])
    )
    ratios = [float(r.get("ratio_rusticl_vs_clover", 0.0)) for r in rows]
    return float(min(ratios)) if ratios else 0.0


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 4 - Operational Consolidation Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline canary: `{payload['metadata']['baseline_canary_path']}`")
    lines.append(f"- Current canary: `{payload['metadata']['current_canary_path']}`")
    lines.append(f"- Policy eval v2: `{payload['metadata']['policy_eval_v2_path']}`")
    lines.append(f"- Split eval: `{payload['metadata']['split_eval_path']}`")
    lines.append("")
    lines.append("## Block Summaries")
    lines.append("")
    lines.append("| Window | Decision | Snapshots | Rollback | Correctness max | T3 fallback max | T5 overhead max % | T5 disable total |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for name in ("baseline", "current"):
        row = payload["windows"][name]
        lines.append(
            f"| {name} | {row['decision']} | {row['executed_snapshots']} | {row['rollback_triggered']} | "
            f"{row['correctness_max']:.9f} | {row['t3_fallback_max']:.4f} | {row['t5_overhead_max']:.4f} | {row['t5_disable_events_total']} |"
        )
    lines.append("")
    lines.append("## Comparative Deltas (current vs baseline)")
    lines.append("")
    lines.append("| Key | Baseline avg GFLOPS | Current avg GFLOPS | Delta % | Baseline p95 ms | Current p95 ms | Delta % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["comparison_rows"]:
        lines.append(
            f"| {row['key']} | {row['baseline_avg_gflops']:.3f} | {row['current_avg_gflops']:.3f} | "
            f"{row['avg_gflops_delta_percent']:.3f} | {row['baseline_p95_ms']:.3f} | {row['current_p95_ms']:.3f} | "
            f"{row['p95_delta_percent']:.3f} |"
        )
    lines.append("")
    lines.append("## Drift v2 and Split Status")
    lines.append("")
    lines.append(f"- Drift v2 decision: `{payload['drift_status_v2']['decision']}`")
    lines.append(f"- Drift v2 failed checks: {payload['drift_status_v2']['failed_checks']}")
    lines.append(f"- Split eval decision: `{payload['split_status']['decision']}`")
    lines.append(f"- Split ratio rusticl/clover min: `{payload['split_status']['ratio_min']:.6f}`")
    lines.append("")
    lines.append("## Package Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['package_decision']}`")
    lines.append(f"- Failed checks: {payload['failed_checks']}")
    lines.append(f"- Rationale: {payload['package_rationale']}")
    lines.append("")
    return "\n".join(lines)


def _drift_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 4 - Drift Status v2")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Policy eval path: `{payload['metadata']['policy_eval_v2_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in payload["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['decision']}`")
    lines.append(f"- Failed checks: {payload['failed_checks']}")
    lines.append(f"- Rationale: {payload['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week13 Block4 operational consolidation package.")
    parser.add_argument("--baseline-canary", required=True)
    parser.add_argument("--current-canary", required=True)
    parser.add_argument("--policy-eval-v2", required=True)
    parser.add_argument("--split-eval", required=True)
    parser.add_argument("--block3-report", required=False, default=None)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week13_controlled_rollout",
    )
    parser.add_argument("--dashboard-prefix", default="week13_block4_operational_dashboard")
    parser.add_argument("--drift-prefix", default="week13_block4_drift_status_v2")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_canary).resolve()
    current_path = Path(args.current_canary).resolve()
    policy_eval_path = Path(args.policy_eval_v2).resolve()
    split_eval_path = Path(args.split_eval).resolve()
    block3_report_path = Path(args.block3_report).resolve() if args.block3_report else None
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_canary = _read_json(baseline_path)
    current_canary = _read_json(current_path)
    policy_eval_v2 = _read_json(policy_eval_path)
    split_eval = _read_json(split_eval_path)
    block3_report = _read_json(block3_report_path) if block3_report_path else None

    baseline_summary = _canary_summary(baseline_canary)
    current_summary = _canary_summary(current_canary)

    baseline_means = _per_key_means(baseline_canary)
    current_means = _per_key_means(current_canary)
    common_keys = sorted(set(baseline_means).intersection(current_means))
    comparison_rows: list[dict[str, Any]] = []
    for row_key in common_keys:
        b = baseline_means[row_key]
        c = current_means[row_key]
        avg_delta = 0.0 if b["avg_gflops_mean"] == 0.0 else (
            (c["avg_gflops_mean"] - b["avg_gflops_mean"]) / b["avg_gflops_mean"] * 100.0
        )
        p95_delta = 0.0 if b["p95_time_ms_mean"] == 0.0 else (
            (c["p95_time_ms_mean"] - b["p95_time_ms_mean"]) / b["p95_time_ms_mean"] * 100.0
        )
        comparison_rows.append(
            {
                "key": row_key,
                "baseline_avg_gflops": float(b["avg_gflops_mean"]),
                "current_avg_gflops": float(c["avg_gflops_mean"]),
                "avg_gflops_delta_percent": float(avg_delta),
                "baseline_p95_ms": float(b["p95_time_ms_mean"]),
                "current_p95_ms": float(c["p95_time_ms_mean"]),
                "p95_delta_percent": float(p95_delta),
            }
        )

    drift_status_v2 = {
        "decision": str(policy_eval_v2.get("evaluation", {}).get("decision", "unknown")),
        "failed_checks": list(policy_eval_v2.get("evaluation", {}).get("failed_checks", [])),
        "rationale": str(policy_eval_v2.get("evaluation", {}).get("rationale", "")),
    }
    split_status = {
        "decision": str(split_eval.get("evaluation", {}).get("decision", "unknown")),
        "failed_checks": list(split_eval.get("evaluation", {}).get("failed_checks", [])),
        "ratio_min": float(_split_ratio_min(split_eval)),
    }

    failed_checks: list[str] = []
    if current_summary["decision"] != "promote":
        failed_checks.append("current_canary_not_promote")
    if current_summary["rollback_triggered"]:
        failed_checks.append("rollback_triggered")
    if drift_status_v2["decision"] != "promote":
        failed_checks.append("drift_v2_not_promote")
    if split_status["decision"] != "promote":
        failed_checks.append("split_eval_not_promote")
    if split_status["ratio_min"] < 0.85:
        failed_checks.append("split_ratio_below_floor")

    package_decision = "promote" if not failed_checks else "iterate"
    package_rationale = (
        "Quincenal operational package is stable under policy v2 with healthy split compatibility."
        if package_decision == "promote"
        else "One or more operational checks failed; keep iterate and refine before promotion."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_canary_path": str(baseline_path),
            "current_canary_path": str(current_path),
            "policy_eval_v2_path": str(policy_eval_path),
            "split_eval_path": str(split_eval_path),
            "block3_report_path": str(block3_report_path) if block3_report_path else None,
        },
        "windows": {
            "baseline": baseline_summary,
            "current": current_summary,
        },
        "comparison_rows": comparison_rows,
        "drift_status_v2": drift_status_v2,
        "split_status": split_status,
        "block3_summary": (
            block3_report.get("evaluation", {}).get("summary", {}) if block3_report else {}
        ),
        "package_decision": package_decision,
        "package_rationale": package_rationale,
        "failed_checks": failed_checks,
    }

    drift_report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_eval_v2_path": str(policy_eval_path),
        },
        "checks": policy_eval_v2.get("evaluation", {}).get("checks", {}),
        "decision": drift_status_v2["decision"],
        "failed_checks": drift_status_v2["failed_checks"],
        "rationale": drift_status_v2["rationale"],
    }

    dashboard_json = out_dir / f"{args.dashboard_prefix}_{stamp}.json"
    dashboard_md = out_dir / f"{args.dashboard_prefix}_{stamp}.md"
    drift_json = out_dir / f"{args.drift_prefix}_{stamp}.json"
    drift_md = out_dir / f"{args.drift_prefix}_{stamp}.md"
    dashboard_json.write_text(json.dumps(dashboard, indent=2) + "\n")
    dashboard_md.write_text(_dashboard_md(dashboard) + "\n")
    drift_json.write_text(json.dumps(drift_report, indent=2) + "\n")
    drift_md.write_text(_drift_md(drift_report) + "\n")

    print(f"Week13 block4 dashboard JSON: {dashboard_json}")
    print(f"Week13 block4 dashboard MD:   {dashboard_md}")
    print(f"Week13 block4 drift JSON:     {drift_json}")
    print(f"Week13 block4 drift MD:       {drift_md}")
    print(f"Package decision: {package_decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if package_decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
