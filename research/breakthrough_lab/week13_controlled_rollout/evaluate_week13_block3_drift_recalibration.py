#!/usr/bin/env python3
"""Week 13 Block 3: biweekly drift review and conservative SLO recalibration."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_cv(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_val = statistics.mean(values)
    if mean_val == 0.0:
        return 0.0
    return float(statistics.pstdev(values) / mean_val)


def _aggregate_rows(eval_reports: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    per_key: dict[str, dict[str, Any]] = {}
    global_drift_abs_max = 0.0
    global_p95_drift_max = 0.0
    decisions: list[str] = []

    for idx, report in enumerate(eval_reports):
        evaluation = report.get("evaluation", {})
        decisions.append(str(evaluation.get("decision", "unknown")))
        checks = evaluation.get("checks", {})
        global_drift_abs_max = max(
            global_drift_abs_max,
            float(checks.get("throughput_drift_abs_bound", {}).get("observed_max_abs_percent", 0.0)),
        )
        global_p95_drift_max = max(
            global_p95_drift_max,
            float(checks.get("p95_drift_bound", {}).get("observed_max_percent", 0.0)),
        )

        for row_key, row in evaluation.get("rows", {}).items():
            slot = per_key.setdefault(
                row_key,
                {
                    "kernel": str(row.get("kernel")),
                    "size": int(row.get("size")),
                    "avg_gflops_mean": [],
                    "p95_time_ms_mean": [],
                    "throughput_drift_percent": [],
                    "p95_drift_percent": [],
                    "t3_fallback_max": [],
                    "t5_overhead_max": [],
                    "t5_disable_events_total": [],
                    "windows": [],
                },
            )
            slot["avg_gflops_mean"].append(float(row.get("avg_gflops_mean", 0.0)))
            slot["p95_time_ms_mean"].append(float(row.get("p95_time_ms_mean", 0.0)))
            slot["throughput_drift_percent"].append(float(row.get("throughput_drift_percent", 0.0)))
            slot["p95_drift_percent"].append(float(row.get("p95_drift_percent", 0.0)))
            slot["t3_fallback_max"].append(float(row.get("t3_fallback_max", 0.0)))
            slot["t5_overhead_max"].append(float(row.get("t5_overhead_max", 0.0)))
            slot["t5_disable_events_total"].append(int(row.get("t5_disable_events_total", 0)))
            slot["windows"].append(idx)

    summary = {
        "windows_analyzed": int(len(eval_reports)),
        "all_window_decisions_promote": all(d == "promote" for d in decisions),
        "window_decisions": decisions,
        "global_max_abs_throughput_drift_percent": float(global_drift_abs_max),
        "global_max_p95_drift_percent": float(global_p95_drift_max),
    }
    return per_key, summary


def _row_stats(values: dict[str, Any]) -> dict[str, Any]:
    avg_vals = [float(v) for v in values["avg_gflops_mean"]]
    p95_vals = [float(v) for v in values["p95_time_ms_mean"]]
    thr_vals = [float(v) for v in values["throughput_drift_percent"]]
    p95d_vals = [float(v) for v in values["p95_drift_percent"]]
    return {
        "samples": len(avg_vals),
        "avg_gflops_mean": float(statistics.mean(avg_vals)) if avg_vals else 0.0,
        "avg_gflops_min": float(min(avg_vals)) if avg_vals else 0.0,
        "avg_gflops_cv": float(_safe_cv(avg_vals)),
        "p95_time_ms_mean": float(statistics.mean(p95_vals)) if p95_vals else 0.0,
        "p95_time_ms_max": float(max(p95_vals)) if p95_vals else 0.0,
        "p95_time_ms_cv": float(_safe_cv(p95_vals)),
        "throughput_drift_abs_max": float(max((abs(x) for x in thr_vals), default=0.0)),
        "p95_drift_max": float(max(p95d_vals, default=0.0)),
        "t3_fallback_max": float(max((float(x) for x in values["t3_fallback_max"]), default=0.0)),
        "t5_overhead_max": float(max((float(x) for x in values["t5_overhead_max"]), default=0.0)),
        "t5_disable_events_total": int(sum(int(x) for x in values["t5_disable_events_total"])),
    }


def _recalibrate_policy(
    base_policy: dict[str, Any],
    per_key_stats: dict[str, dict[str, Any]],
    *,
    tighten_headroom_fraction: float,
    observed_global_abs_drift_max: float,
    observed_global_p95_drift_max: float,
) -> dict[str, Any]:
    out = json.loads(json.dumps(base_policy))
    out["policy_id"] = "week13-block3-weekly-slo-v2-2026-02-09"
    out["status"] = "candidate_weekly_monitoring_policy_recalibrated"
    out["inherits_from"] = "research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json"
    out["recalibration"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "method": "conservative_headroom_tightening",
        "tighten_headroom_fraction": float(tighten_headroom_fraction),
        "evidence_source": "week11_block4 + week13_block1 + week13_block2",
    }

    guardrails = out["weekly_slo"]["global_guardrails"]
    current_abs = float(guardrails["max_abs_throughput_drift_percent"])
    current_p95 = float(guardrails["max_p95_drift_percent"])

    # Keep rollback-safe defaults while reducing slack based on observed stable behavior.
    candidate_abs = max(2.5, round(observed_global_abs_drift_max * 1.5, 6))
    candidate_p95 = max(5.0, round(observed_global_p95_drift_max * 2.0, 6))
    guardrails["max_abs_throughput_drift_percent"] = float(min(current_abs, candidate_abs))
    guardrails["max_p95_drift_percent"] = float(min(current_p95, candidate_p95))

    for row_key, cfg in out["weekly_slo"]["per_kernel_size"].items():
        stats = per_key_stats.get(row_key)
        if not stats:
            continue
        current_min_avg = float(cfg["min_avg_gflops"])
        current_max_p95 = float(cfg["max_p95_time_ms"])
        observed_min_avg = float(stats["avg_gflops_min"])
        observed_max_p95 = float(stats["p95_time_ms_max"])

        if observed_min_avg > current_min_avg:
            new_min_avg = current_min_avg + (observed_min_avg - current_min_avg) * float(
                tighten_headroom_fraction
            )
            cfg["min_avg_gflops"] = float(round(new_min_avg, 6))

        if observed_max_p95 < current_max_p95:
            new_max_p95 = current_max_p95 - (current_max_p95 - observed_max_p95) * float(
                tighten_headroom_fraction
            )
            cfg["max_p95_time_ms"] = float(round(new_max_p95, 6))

        cfg["notes"] = (
            "Recalibrated in Week 13 Block 3 with sustained evidence; conservative headroom tightening."
        )

    return out


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 3 - Drift Review and Recalibration")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Base policy: `{report['metadata']['base_policy_path']}`")
    lines.append(f"- Recalibrated policy: `{report['artifacts']['recalibrated_policy_path']}`")
    lines.append("")
    lines.append("## Sustained Evidence Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Per-Key Summary")
    lines.append("")
    lines.append("| Key | Samples | Avg GFLOPS min | Avg GFLOPS cv | P95 max ms | P95 cv | Max abs thr drift % | Max p95 drift % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(report["evaluation"]["per_key"]):
        row = report["evaluation"]["per_key"][key]
        lines.append(
            f"| {key} | {row['samples']} | {row['avg_gflops_min']:.3f} | {row['avg_gflops_cv']:.4f} | "
            f"{row['p95_time_ms_max']:.3f} | {row['p95_time_ms_cv']:.4f} | {row['throughput_drift_abs_max']:.3f} | "
            f"{row['p95_drift_max']:.3f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Recalibration action: `{report['evaluation']['recalibration_action']}`")
    lines.append(f"- Failed checks: {report['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Week13 Block3 drift review + conservative recalibration.")
    parser.add_argument("--base-policy-path", required=True)
    parser.add_argument("--eval-artifacts", nargs="+", required=True)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--max-global-abs-drift-for-recalibration", type=float, default=2.0)
    parser.add_argument("--max-global-p95-drift-for-recalibration", type=float, default=1.0)
    parser.add_argument("--max-avg-cv-for-recalibration", type=float, default=0.01)
    parser.add_argument("--max-p95-cv-for-recalibration", type=float, default=0.01)
    parser.add_argument("--tighten-headroom-fraction", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week13_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week13_block3_drift_recalibration")
    parser.add_argument(
        "--policy-output-path",
        default="research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json",
    )
    args = parser.parse_args()

    base_policy_path = Path(args.base_policy_path).resolve()
    eval_paths = [Path(p).resolve() for p in args.eval_artifacts]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_output_path = Path(args.policy_output_path).resolve()
    policy_output_path.parent.mkdir(parents=True, exist_ok=True)

    base_policy = _read_json(base_policy_path)
    eval_reports = [_read_json(path) for path in eval_paths]
    per_key_raw, summary = _aggregate_rows(eval_reports)

    per_key: dict[str, dict[str, Any]] = {}
    for key, values in per_key_raw.items():
        per_key[key] = _row_stats(values)

    policy_rows = set(base_policy.get("weekly_slo", {}).get("per_kernel_size", {}).keys())
    failed_checks: list[str] = []

    checks: dict[str, dict[str, Any]] = {}
    checks["all_windows_promote"] = {
        "observed": summary["window_decisions"],
        "required": "all promote",
        "pass": bool(summary["all_window_decisions_promote"]),
    }
    checks["global_abs_drift_stable"] = {
        "observed_max": float(summary["global_max_abs_throughput_drift_percent"]),
        "required_max": float(args.max_global_abs_drift_for_recalibration),
        "pass": float(summary["global_max_abs_throughput_drift_percent"])
        <= float(args.max_global_abs_drift_for_recalibration),
    }
    checks["global_p95_drift_stable"] = {
        "observed_max": float(summary["global_max_p95_drift_percent"]),
        "required_max": float(args.max_global_p95_drift_for_recalibration),
        "pass": float(summary["global_max_p95_drift_percent"])
        <= float(args.max_global_p95_drift_for_recalibration),
    }

    per_row_checks: list[dict[str, Any]] = []
    for key in sorted(policy_rows):
        row = per_key.get(key)
        if row is None:
            per_row_checks.append(
                {
                    "key": key,
                    "pass": False,
                    "reason": "missing_row",
                }
            )
            continue
        row_pass = (
            int(row["samples"]) >= int(args.min_windows)
            and float(row["avg_gflops_cv"]) <= float(args.max_avg_cv_for_recalibration)
            and float(row["p95_time_ms_cv"]) <= float(args.max_p95_cv_for_recalibration)
        )
        per_row_checks.append(
            {
                "key": key,
                "samples": int(row["samples"]),
                "avg_gflops_cv": float(row["avg_gflops_cv"]),
                "p95_time_ms_cv": float(row["p95_time_ms_cv"]),
                "required_samples_min": int(args.min_windows),
                "required_avg_cv_max": float(args.max_avg_cv_for_recalibration),
                "required_p95_cv_max": float(args.max_p95_cv_for_recalibration),
                "pass": bool(row_pass),
            }
        )
    checks["policy_rows_sustained_stability"] = {
        "rows": per_row_checks,
        "pass": all(bool(r["pass"]) for r in per_row_checks) and len(per_row_checks) > 0,
    }

    for check_name, payload in checks.items():
        if not bool(payload.get("pass")):
            failed_checks.append(check_name)

    recalibration_allowed = len(failed_checks) == 0
    if recalibration_allowed:
        recalibrated = _recalibrate_policy(
            base_policy,
            per_key,
            tighten_headroom_fraction=float(args.tighten_headroom_fraction),
            observed_global_abs_drift_max=float(summary["global_max_abs_throughput_drift_percent"]),
            observed_global_p95_drift_max=float(summary["global_max_p95_drift_percent"]),
        )
        recalibration_action = "applied"
        rationale = (
            "Sustained evidence conditions passed across windows; conservative tightening was applied."
        )
    else:
        recalibrated = json.loads(json.dumps(base_policy))
        recalibrated["recalibration"] = {
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "method": "hold",
            "reason": "sustained_evidence_not_met",
            "failed_checks": failed_checks,
        }
        recalibration_action = "held"
        rationale = "Sustained evidence criteria were not fully met; thresholds were kept unchanged."

    policy_output_path.write_text(json.dumps(recalibrated, indent=2) + "\n")

    decision = "promote" if summary["all_window_decisions_promote"] else "iterate"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_policy_path": str(base_policy_path),
            "eval_artifacts": [str(p) for p in eval_paths],
            "min_windows": int(args.min_windows),
        },
        "artifacts": {
            "recalibrated_policy_path": str(policy_output_path),
        },
        "evaluation": {
            "summary": summary,
            "checks": checks,
            "per_key": per_key,
            "decision": decision,
            "recalibration_action": recalibration_action,
            "failed_checks": failed_checks,
            "rationale": rationale,
        },
    }

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report) + "\n")

    print(f"Week13 block3 JSON: {json_path}")
    print(f"Week13 block3 MD:   {md_path}")
    print(f"Recalibrated policy: {policy_output_path}")
    print(f"Decision: {decision}")
    print(f"Recalibration action: {recalibration_action}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
