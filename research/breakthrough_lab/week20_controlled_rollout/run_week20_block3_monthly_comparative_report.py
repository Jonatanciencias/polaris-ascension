#!/usr/bin/env python3
"""Week 20 Block 3: monthly comparative report + operational debt review."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_prefixed_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        row = line.strip()
        if row.startswith(prefix):
            return row[len(prefix) :].strip()
    return None


def _latest_glob(pattern: str) -> Path | None:
    candidates = sorted(REPO_ROOT.glob(pattern))
    return candidates[-1] if candidates else None


def _safe_delta_percent(current: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 3 - Monthly Comparative Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline report: `{payload['metadata']['baseline_report_path']}`")
    lines.append(f"- Current report: `{payload['metadata']['current_report_path']}`")
    lines.append("")
    lines.append("## Comparative Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in payload["rows"]:
        lines.append(
            f"| {row['metric']} | {row['baseline_value']:.6f} | {row['current_value']:.6f} | {row['delta_percent']:.6f} |"
        )
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append(f"- Baseline decision: `{payload['decisions']['baseline_decision']}`")
    lines.append(f"- Current block1 decision: `{payload['decisions']['current_block1_decision']}`")
    lines.append(f"- Current block2 decision: `{payload['decisions']['current_block2_decision']}`")
    lines.append(f"- Block2 alerts decision: `{payload['decisions']['block2_alerts_decision']}`")
    lines.append("")
    lines.append("## Debt Summary")
    lines.append("")
    lines.append(
        f"- Open debts: `{payload['debt_summary']['open_total']}`"
    )
    lines.append(
        f"- High/Critical open debts: `{payload['debt_summary']['high_or_critical_open_total']}`"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 3 - Monthly Comparative Report + Debt Review")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- Baseline decision: `{payload['highlights']['baseline_decision']}`")
    lines.append(f"- Current block1 decision: `{payload['highlights']['current_block1_decision']}`")
    lines.append(f"- Current block2 decision: `{payload['highlights']['current_block2_decision']}`")
    lines.append(f"- Block2 alerts decision: `{payload['highlights']['block2_alerts_decision']}`")
    lines.append(
        f"- split_ratio_min delta %: `{float(payload['highlights']['split_ratio_delta_percent']):.6f}`"
    )
    lines.append(
        f"- t5_overhead_max delta %: `{float(payload['highlights']['t5_overhead_delta_percent']):.6f}`"
    )
    lines.append(
        f"- t5_disable_total delta: `{int(payload['highlights']['t5_disable_delta'])}`"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week20 Block3 monthly comparative report and debt review."
    )
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--baseline-report-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_20260211_015638.json",
    )
    parser.add_argument(
        "--baseline-decision-path",
        default="research/breakthrough_lab/week19_block1_weekly_split_maintenance_decision.json",
    )
    parser.add_argument(
        "--current-block1-report-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block1_monthly_cycle_20260211_023053.json",
    )
    parser.add_argument(
        "--current-block1-decision-path",
        default="research/breakthrough_lab/week20_block1_monthly_full_cycle_decision.json",
    )
    parser.add_argument(
        "--current-block2-report-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_20*.json",
    )
    parser.add_argument(
        "--current-block2-alerts-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_*.json",
    )
    parser.add_argument(
        "--week20-block1-debt-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK1_MONTHLY_CYCLE_LIVE_DEBT_MATRIX.json",
    )
    parser.add_argument(
        "--week20-block2-debt-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_operational_debt_*.json",
    )
    parser.add_argument("--max-overhead-regression-percent", type=float, default=25.0)
    parser.add_argument("--min-rusticl-ratio-floor", type=float, default=0.85)
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week20_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week20_block3_monthly_comparative_report")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    baseline_report_path = _resolve(args.baseline_report_path)
    baseline_decision_path = _resolve(args.baseline_decision_path)
    current_block1_report_path = _resolve(args.current_block1_report_path)
    current_block1_decision_path = _resolve(args.current_block1_decision_path)
    if "*" in args.current_block2_report_path:
        current_block2_report_path = _latest_glob(args.current_block2_report_path)
    else:
        current_block2_report_path = _resolve(args.current_block2_report_path)
    if "*" in args.current_block2_alerts_path:
        current_block2_alerts_path = _latest_glob(args.current_block2_alerts_path)
    else:
        current_block2_alerts_path = _resolve(args.current_block2_alerts_path)
    week20_block1_debt_path = _resolve(args.week20_block1_debt_path)
    if "*" in args.week20_block2_debt_path:
        week20_block2_debt_path = _latest_glob(args.week20_block2_debt_path)
    else:
        week20_block2_debt_path = _resolve(args.week20_block2_debt_path)
    preprod_dir = _resolve(args.preprod_signoff_dir)
    output_dir = _resolve(args.output_dir)
    preprod_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest = _read_json(stable_manifest_path) if stable_manifest_path.exists() else {}
    stable_tag = str(stable_manifest.get("stable_tag", "unknown"))
    baseline_report = _read_json(baseline_report_path) if baseline_report_path.exists() else {}
    baseline_decision = _read_json(baseline_decision_path) if baseline_decision_path.exists() else {}
    current_block1_report = (
        _read_json(current_block1_report_path) if current_block1_report_path.exists() else {}
    )
    current_block1_decision = (
        _read_json(current_block1_decision_path) if current_block1_decision_path.exists() else {}
    )
    current_block2_report = (
        _read_json(current_block2_report_path)
        if current_block2_report_path and current_block2_report_path.exists()
        else {}
    )
    current_block2_alerts = (
        _read_json(current_block2_alerts_path)
        if current_block2_alerts_path and current_block2_alerts_path.exists()
        else {}
    )

    baseline_block_decision = str(baseline_decision.get("block_decision", "unknown"))
    current_block1_block_decision = str(current_block1_decision.get("block_decision", "unknown"))
    current_block2_block_decision = str(
        current_block2_report.get("evaluation", {}).get("decision", "unknown")
    )
    current_block2_alerts_decision = str(
        current_block2_alerts.get("summary", {}).get("decision", "unknown")
    )

    baseline_ratio = float(baseline_report.get("highlights", {}).get("split_ratio_min", 0.0))
    current_ratio = float(current_block1_report.get("highlights", {}).get("split_ratio_min", 0.0))
    baseline_t5_overhead = float(
        baseline_report.get("highlights", {}).get("split_t5_overhead_max", 999.0)
    )
    current_t5_overhead = float(
        current_block1_report.get("highlights", {}).get("split_t5_overhead_max", 999.0)
    )
    baseline_t5_disable = int(
        baseline_report.get("highlights", {}).get("split_t5_disable_total", 999)
    )
    current_t5_disable = int(
        current_block1_report.get("highlights", {}).get("split_t5_disable_total", 999)
    )

    rows = [
        {
            "metric": "split_ratio_min",
            "baseline_value": baseline_ratio,
            "current_value": current_ratio,
            "delta_percent": _safe_delta_percent(current_ratio, baseline_ratio),
        },
        {
            "metric": "t5_overhead_max",
            "baseline_value": baseline_t5_overhead,
            "current_value": current_t5_overhead,
            "delta_percent": _safe_delta_percent(current_t5_overhead, baseline_t5_overhead),
        },
        {
            "metric": "t5_disable_total",
            "baseline_value": float(baseline_t5_disable),
            "current_value": float(current_t5_disable),
            "delta_percent": _safe_delta_percent(float(current_t5_disable), float(baseline_t5_disable))
            if baseline_t5_disable != 0
            else 0.0,
        },
    ]

    debt_matrices: list[dict[str, Any]] = []
    for debt_path in (week20_block1_debt_path, week20_block2_debt_path):
        if debt_path and debt_path.exists():
            debt_matrices.append(_read_json(debt_path))
    debts: list[dict[str, Any]] = []
    for matrix in debt_matrices:
        debts.extend(list(matrix.get("debts", [])))
    debt_summary = {
        "matrices_loaded": len(debt_matrices),
        "open_total": sum(1 for debt in debts if debt.get("status") == "open"),
        "high_or_critical_open_total": sum(
            1
            for debt in debts
            if debt.get("status") == "open" and debt.get("severity") in {"high", "critical"}
        ),
    }
    debt_review_payload = {
        "review_id": "week20-block3-operational-debt-review-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_matrices": [str(p) for p in (week20_block1_debt_path, week20_block2_debt_path)],
        "summary": debt_summary,
        "debts": debts,
    }

    pre_gate_cmd = [
        "./venv/bin/python",
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.report_dir),
    ]
    pre_gate = _run(pre_gate_cmd)
    pre_gate_json = _extract_prefixed_line(pre_gate["stdout"], "Wrote JSON report:")
    pre_gate_decision = "unknown"
    pre_gate_pytest_green = False
    if pre_gate_json:
        pre_payload = _read_json(_resolve(pre_gate_json))
        pre_gate_decision = str(pre_payload.get("evaluation", {}).get("decision", "unknown"))
        pre_gate_pytest_green = bool(
            pre_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report_path": str(baseline_report_path),
            "current_report_path": str(current_block1_report_path),
        },
        "rows": rows,
        "decisions": {
            "baseline_decision": baseline_block_decision,
            "current_block1_decision": current_block1_block_decision,
            "current_block2_decision": current_block2_block_decision,
            "block2_alerts_decision": current_block2_alerts_decision,
        },
        "debt_summary": debt_summary,
    }
    dashboard_json = output_dir / f"week20_block3_monthly_comparative_dashboard_{stamp}.json"
    dashboard_md = output_dir / f"week20_block3_monthly_comparative_dashboard_{stamp}.md"
    _write_json(dashboard_json, dashboard_payload)
    dashboard_md.write_text(_dashboard_md(dashboard_payload))

    debt_review_json = preprod_dir / "WEEK20_BLOCK3_OPERATIONAL_DEBT_REVIEW.json"
    _write_json(debt_review_json, debt_review_payload)

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    post_gate_pytest_green = False
    if post_gate_json:
        post_payload = _read_json(_resolve(post_gate_json))
        post_gate_decision = str(post_payload.get("evaluation", {}).get("decision", "unknown"))
        post_gate_pytest_green = bool(
            post_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
        )

    ratio_delta = _safe_delta_percent(current_ratio, baseline_ratio)
    t5_overhead_delta = _safe_delta_percent(current_t5_overhead, baseline_t5_overhead)
    t5_disable_delta = current_t5_disable - baseline_t5_disable

    checks: dict[str, dict[str, Any]] = {}
    checks["stable_manifest_exists"] = {
        "observed": stable_manifest_path.exists(),
        "required": True,
        "pass": stable_manifest_path.exists(),
    }
    checks["stable_tag_v0_15_0"] = {
        "observed": stable_tag,
        "required": "v0.15.0",
        "pass": stable_tag == "v0.15.0",
    }
    checks["baseline_block_decision_promote"] = {
        "observed": baseline_block_decision,
        "required": "promote",
        "pass": baseline_block_decision == "promote",
    }
    checks["current_block1_decision_promote"] = {
        "observed": current_block1_block_decision,
        "required": "promote",
        "pass": current_block1_block_decision == "promote",
    }
    checks["current_block2_decision_promote"] = {
        "observed": current_block2_block_decision,
        "required": "promote",
        "pass": current_block2_block_decision == "promote",
    }
    checks["current_block2_alerts_promote"] = {
        "observed": current_block2_alerts_decision,
        "required": "promote",
        "pass": current_block2_alerts_decision == "promote",
    }
    checks["current_split_ratio_floor"] = {
        "observed": current_ratio,
        "required_min": float(args.min_rusticl_ratio_floor),
        "pass": current_ratio >= float(args.min_rusticl_ratio_floor),
    }
    checks["t5_overhead_regression_within_limit"] = {
        "observed_delta_percent": t5_overhead_delta,
        "required_max_delta_percent": float(args.max_overhead_regression_percent),
        "pass": t5_overhead_delta <= float(args.max_overhead_regression_percent),
    }
    checks["t5_disable_events_not_regressed"] = {
        "observed_delta": int(t5_disable_delta),
        "required_max_delta": 0,
        "pass": int(t5_disable_delta) <= 0,
    }
    checks["comparative_dashboard_written"] = {
        "observed": dashboard_json.exists() and dashboard_md.exists(),
        "required": True,
        "pass": dashboard_json.exists() and dashboard_md.exists(),
    }
    checks["debt_review_written"] = {
        "observed": debt_review_json.exists(),
        "required": True,
        "pass": debt_review_json.exists(),
    }
    checks["block2_inputs_exist"] = {
        "observed": bool(current_block2_report_path and current_block2_alerts_path),
        "required": True,
        "pass": bool(current_block2_report_path and current_block2_alerts_path),
    }
    checks["no_high_critical_open_debt"] = {
        "observed": int(debt_summary["high_or_critical_open_total"]),
        "required_max": 0,
        "pass": int(debt_summary["high_or_critical_open_total"]) <= 0,
    }
    checks["pre_gate_promote"] = {
        "observed": pre_gate_decision,
        "required": "promote",
        "pass": pre_gate_decision == "promote",
    }
    checks["pre_gate_pytest_tier_green"] = {
        "observed": bool(pre_gate_pytest_green),
        "required": True,
        "pass": bool(pre_gate_pytest_green),
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }
    checks["post_gate_pytest_tier_green"] = {
        "observed": bool(post_gate_pytest_green),
        "required": True,
        "pass": bool(post_gate_pytest_green),
    }

    failed_checks = [key for key, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Monthly comparative review confirms stable continuation with controlled debt and green gates."
        if decision == "promote"
        else "Monthly comparative/debt review found unresolved regressions or governance blockers."
    )

    report_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "baseline_report_path": str(baseline_report_path),
            "current_block1_report_path": str(current_block1_report_path),
            "current_block2_report_path": str(current_block2_report_path),
            "current_block2_alerts_path": str(current_block2_alerts_path),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "pre_gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "artifacts": {
            "dashboard_json": str(dashboard_json),
            "dashboard_md": str(dashboard_md),
            "debt_review_json": str(debt_review_json),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "baseline_decision": baseline_block_decision,
            "current_block1_decision": current_block1_block_decision,
            "current_block2_decision": current_block2_block_decision,
            "block2_alerts_decision": current_block2_alerts_decision,
            "split_ratio_delta_percent": ratio_delta,
            "t5_overhead_delta_percent": t5_overhead_delta,
            "t5_disable_delta": int(t5_disable_delta),
            "debt_open_total": int(debt_summary["open_total"]),
            "debt_high_critical_open_total": int(debt_summary["high_or_critical_open_total"]),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    report_json = output_dir / f"{args.output_prefix}_{stamp}.json"
    report_md = output_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(report_json, report_payload)
    report_md.write_text(_report_md(report_payload))

    print(f"Week20 block3 JSON: {report_json}")
    print(f"Week20 block3 MD:   {report_md}")
    print(f"Dashboard JSON:     {dashboard_json}")
    print(f"Dashboard MD:       {dashboard_md}")
    print(f"Debt review JSON:   {debt_review_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
