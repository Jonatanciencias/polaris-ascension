#!/usr/bin/env python3
"""Week 21 Block 3: second monthly comparative + formal platform decision."""

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


def _platform_recommendation(
    *,
    split_eval: dict[str, Any],
    cycle_report: dict[str, Any],
    dual_go_ratio_min: float,
    ratio_floor: float,
    max_t5_overhead_for_dual_go: float,
) -> dict[str, Any]:
    checks = split_eval.get("evaluation", {}).get("checks", {})
    ratio_rows = checks.get("rusticl_ratio_vs_clover", {}).get("rows", [])
    ratios = [float(row.get("ratio_rusticl_vs_clover", 0.0)) for row in ratio_rows]
    ratio_min = min(ratios) if ratios else 0.0
    split_decision = str(split_eval.get("evaluation", {}).get("decision", "unknown"))
    cycle_decision = str(cycle_report.get("evaluation", {}).get("decision", "unknown"))
    t5_overhead = float(cycle_report.get("highlights", {}).get("split_t5_overhead_max", 999.0))
    t5_disable = int(cycle_report.get("highlights", {}).get("split_t5_disable_total", 999))

    if (
        split_decision == "promote"
        and cycle_decision == "promote"
        and ratio_min >= dual_go_ratio_min
        and t5_overhead <= max_t5_overhead_for_dual_go
        and t5_disable == 0
    ):
        policy = "dual_go_clover_rusticl"
        rationale = "Rusticl/Clover ratio and guardrails support dual-platform production operation."
    elif split_decision == "promote" and ratio_min >= ratio_floor and t5_disable == 0:
        policy = "clover_primary_rusticl_canary"
        rationale = "Rusticl remains healthy but not yet strong enough for full dual-go envelope."
    else:
        policy = "clover_primary_rusticl_shadow_only"
        rationale = "Conservative fallback: keep Clover primary until ratio/guardrails improve."

    by_environment = {
        "production": policy,
        "staging": "dual_go_clover_rusticl" if split_decision == "promote" else "clover_primary_rusticl_shadow_only",
        "development": "dual_go_clover_rusticl",
    }
    return {
        "policy_id": "week21-block3-platform-policy-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "rationale": rationale,
        "ratio_min_observed": ratio_min,
        "split_decision": split_decision,
        "cycle_decision": cycle_decision,
        "t5_overhead_max_observed": t5_overhead,
        "t5_disable_total_observed": t5_disable,
        "by_environment": by_environment,
    }


def _platform_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 21 Block 3 - Platform Recommendation")
    lines.append("")
    lines.append(f"- Timestamp: {payload['timestamp_utc']}")
    lines.append(f"- Policy: `{payload['policy']}`")
    lines.append(f"- ratio_min_observed: `{float(payload['ratio_min_observed']):.6f}`")
    lines.append(f"- t5_overhead_max_observed: `{float(payload['t5_overhead_max_observed']):.6f}`")
    lines.append(f"- t5_disable_total_observed: `{int(payload['t5_disable_total_observed'])}`")
    lines.append("")
    lines.append("## By Environment")
    lines.append("")
    lines.append(f"- production: `{payload['by_environment']['production']}`")
    lines.append(f"- staging: `{payload['by_environment']['staging']}`")
    lines.append(f"- development: `{payload['by_environment']['development']}`")
    lines.append("")
    lines.append("## Rationale")
    lines.append("")
    lines.append(f"- {payload['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 21 Block 3 - Second Monthly Comparative Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline cycle: `{payload['metadata']['baseline_cycle_report_path']}`")
    lines.append(f"- Current cycle: `{payload['metadata']['current_cycle_report_path']}`")
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
    lines.append(f"- Baseline decision: `{payload['decisions']['baseline_cycle_decision']}`")
    lines.append(f"- Current block1 decision: `{payload['decisions']['current_block1_decision']}`")
    lines.append(f"- Current block2 decision: `{payload['decisions']['current_block2_decision']}`")
    lines.append(f"- Platform policy: `{payload['decisions']['platform_policy']}`")
    lines.append("")
    lines.append("## Debt Summary")
    lines.append("")
    lines.append(f"- Open debts: `{payload['debt_summary']['open_total']}`")
    lines.append(f"- High/Critical open debts: `{payload['debt_summary']['high_or_critical_open_total']}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 21 Block 3 - Second Monthly Comparative + Platform Decision")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- Baseline cycle decision: `{report['highlights']['baseline_cycle_decision']}`")
    lines.append(f"- Current block1 decision: `{report['highlights']['current_block1_decision']}`")
    lines.append(f"- Current block2 decision: `{report['highlights']['current_block2_decision']}`")
    lines.append(f"- Platform policy: `{report['highlights']['platform_policy']}`")
    lines.append(
        f"- split_ratio_delta_percent: `{float(report['highlights']['split_ratio_delta_percent']):.6f}`"
    )
    lines.append(
        f"- t5_overhead_delta_percent: `{float(report['highlights']['t5_overhead_delta_percent']):.6f}`"
    )
    lines.append(f"- t5_disable_delta: `{int(report['highlights']['t5_disable_delta'])}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in report["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {report['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week21 Block3 second monthly comparative and platform recommendation."
    )
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--baseline-comparative-report-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block3_monthly_comparative_report_20260211_140926.json",
    )
    parser.add_argument(
        "--current-block1-report-path",
        default="research/breakthrough_lab/week21_controlled_rollout/week21_block1_monthly_continuity_20260211_142611.json",
    )
    parser.add_argument(
        "--current-block2-report-path",
        default="research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_20*.json",
    )
    parser.add_argument(
        "--current-split-eval-path",
        default="research/breakthrough_lab/week21_controlled_rollout/week21_block1_monthly_continuity_split_eval_20260211_142611.json",
    )
    parser.add_argument(
        "--week20-debt-review-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK3_OPERATIONAL_DEBT_REVIEW.json",
    )
    parser.add_argument(
        "--week21-block1-debt-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK21_BLOCK1_MONTHLY_CONTINUITY_LIVE_DEBT_MATRIX.json",
    )
    parser.add_argument(
        "--week21-block2-debt-path",
        default="research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_operational_debt_*.json",
    )
    parser.add_argument("--ratio-floor", type=float, default=0.85)
    parser.add_argument("--dual-go-ratio-min", type=float, default=0.92)
    parser.add_argument("--max-t5-overhead-for-dual-go", type=float, default=2.0)
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
        default="research/breakthrough_lab/week21_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week21_block3_second_monthly_comparative")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    baseline_comp_path = _resolve(args.baseline_comparative_report_path)
    current_block1_path = _resolve(args.current_block1_report_path)
    current_block2_path = (
        _latest_glob(args.current_block2_report_path)
        if "*" in args.current_block2_report_path
        else _resolve(args.current_block2_report_path)
    )
    split_eval_path = _resolve(args.current_split_eval_path)
    week20_debt_path = _resolve(args.week20_debt_review_path)
    week21_block1_debt_path = _resolve(args.week21_block1_debt_path)
    week21_block2_debt_path = (
        _latest_glob(args.week21_block2_debt_path)
        if "*" in args.week21_block2_debt_path
        else _resolve(args.week21_block2_debt_path)
    )
    preprod_dir = _resolve(args.preprod_signoff_dir)
    output_dir = _resolve(args.output_dir)
    preprod_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest = _read_json(stable_manifest_path) if stable_manifest_path.exists() else {}
    stable_tag = str(stable_manifest.get("stable_tag", "unknown"))
    baseline_comp = _read_json(baseline_comp_path) if baseline_comp_path.exists() else {}
    current_block1 = _read_json(current_block1_path) if current_block1_path.exists() else {}
    current_block2 = _read_json(current_block2_path) if current_block2_path and current_block2_path.exists() else {}
    split_eval = _read_json(split_eval_path) if split_eval_path.exists() else {}

    baseline_cycle_path_raw = str(baseline_comp.get("metadata", {}).get("current_block1_report_path", ""))
    baseline_cycle_path = _resolve(baseline_cycle_path_raw) if baseline_cycle_path_raw else None
    baseline_cycle = _read_json(baseline_cycle_path) if baseline_cycle_path and baseline_cycle_path.exists() else {}

    baseline_decision = str(baseline_cycle.get("evaluation", {}).get("decision", "unknown"))
    current_block1_decision = str(current_block1.get("evaluation", {}).get("decision", "unknown"))
    current_block2_decision = str(current_block2.get("evaluation", {}).get("decision", "unknown"))

    baseline_ratio = float(baseline_cycle.get("highlights", {}).get("split_ratio_min", 0.0))
    current_ratio = float(current_block1.get("highlights", {}).get("split_ratio_min", 0.0))
    baseline_t5_overhead = float(baseline_cycle.get("highlights", {}).get("split_t5_overhead_max", 999.0))
    current_t5_overhead = float(current_block1.get("highlights", {}).get("split_t5_overhead_max", 999.0))
    baseline_t5_disable = int(baseline_cycle.get("highlights", {}).get("split_t5_disable_total", 999))
    current_t5_disable = int(current_block1.get("highlights", {}).get("split_t5_disable_total", 999))

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

    platform_policy = _platform_recommendation(
        split_eval=split_eval,
        cycle_report=current_block1,
        dual_go_ratio_min=float(args.dual_go_ratio_min),
        ratio_floor=float(args.ratio_floor),
        max_t5_overhead_for_dual_go=float(args.max_t5_overhead_for_dual_go),
    )
    platform_json = preprod_dir / "WEEK21_BLOCK3_PLATFORM_POLICY_DECISION.json"
    platform_md = preprod_dir / "WEEK21_BLOCK3_PLATFORM_POLICY_DECISION.md"
    _write_json(platform_json, platform_policy)
    platform_md.write_text(_platform_md(platform_policy))

    debt_matrices: list[dict[str, Any]] = []
    for debt_path in (week20_debt_path, week21_block1_debt_path, week21_block2_debt_path):
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
    debt_review_json = preprod_dir / "WEEK21_BLOCK3_OPERATIONAL_DEBT_REVIEW.json"
    _write_json(
        debt_review_json,
        {
            "review_id": "week21-block3-operational-debt-review-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_matrices": [str(p) for p in (week20_debt_path, week21_block1_debt_path, week21_block2_debt_path) if p],
            "summary": debt_summary,
            "debts": debts,
        },
    )

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
            "baseline_cycle_report_path": str(baseline_cycle_path),
            "current_cycle_report_path": str(current_block1_path),
        },
        "rows": rows,
        "decisions": {
            "baseline_cycle_decision": baseline_decision,
            "current_block1_decision": current_block1_decision,
            "current_block2_decision": current_block2_decision,
            "platform_policy": str(platform_policy.get("policy", "unknown")),
        },
        "debt_summary": debt_summary,
    }
    dashboard_json = output_dir / f"week21_block3_second_monthly_dashboard_{stamp}.json"
    dashboard_md = output_dir / f"week21_block3_second_monthly_dashboard_{stamp}.md"
    _write_json(dashboard_json, dashboard_payload)
    dashboard_md.write_text(_dashboard_md(dashboard_payload))

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

    split_ratio_delta = _safe_delta_percent(current_ratio, baseline_ratio)
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
    checks["baseline_cycle_report_exists"] = {
        "observed": bool(baseline_cycle_path and baseline_cycle_path.exists()),
        "required": True,
        "pass": bool(baseline_cycle_path and baseline_cycle_path.exists()),
    }
    checks["current_block1_promote"] = {
        "observed": current_block1_decision,
        "required": "promote",
        "pass": current_block1_decision == "promote",
    }
    checks["current_block2_promote"] = {
        "observed": current_block2_decision,
        "required": "promote",
        "pass": current_block2_decision == "promote",
    }
    checks["split_eval_promote"] = {
        "observed": str(split_eval.get("evaluation", {}).get("decision", "unknown")),
        "required": "promote",
        "pass": str(split_eval.get("evaluation", {}).get("decision", "unknown")) == "promote",
    }
    checks["platform_policy_written"] = {
        "observed": platform_json.exists() and platform_md.exists(),
        "required": True,
        "pass": platform_json.exists() and platform_md.exists(),
    }
    checks["platform_policy_not_shadow_only"] = {
        "observed": str(platform_policy.get("policy")),
        "required_not": "clover_primary_rusticl_shadow_only",
        "pass": str(platform_policy.get("policy")) != "clover_primary_rusticl_shadow_only",
    }
    checks["debt_review_written"] = {
        "observed": debt_review_json.exists(),
        "required": True,
        "pass": debt_review_json.exists(),
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
        "Second monthly comparative confirms continuity and emits a production-capable platform policy."
        if decision == "promote"
        else "Second comparative found unresolved policy, debt, or gate issues."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "baseline_comparative_report_path": str(baseline_comp_path),
            "baseline_cycle_report_path": str(baseline_cycle_path),
            "current_block1_report_path": str(current_block1_path),
            "current_block2_report_path": str(current_block2_path),
            "current_split_eval_path": str(split_eval_path),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "pre_gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "artifacts": {
            "dashboard_json": str(dashboard_json),
            "dashboard_md": str(dashboard_md),
            "platform_policy_json": str(platform_json),
            "platform_policy_md": str(platform_md),
            "debt_review_json": str(debt_review_json),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "baseline_cycle_decision": baseline_decision,
            "current_block1_decision": current_block1_decision,
            "current_block2_decision": current_block2_decision,
            "platform_policy": str(platform_policy.get("policy", "unknown")),
            "split_ratio_delta_percent": split_ratio_delta,
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
    _write_json(report_json, report)
    report_md.write_text(_report_md(report))

    print(f"Week21 block3 JSON: {report_json}")
    print(f"Week21 block3 MD:   {report_md}")
    print(f"Dashboard JSON:     {dashboard_json}")
    print(f"Dashboard MD:       {dashboard_md}")
    print(f"Platform policy:    {platform_json}")
    print(f"Debt review JSON:   {debt_review_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
