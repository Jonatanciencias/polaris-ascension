#!/usr/bin/env python3
"""Week 20 Block 2: scheduled monthly automation with retention and alerts."""

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


def _alerts_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 2 - Monthly Automation Alerts")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Source report: `{payload['metadata']['source_report']}`")
    lines.append(f"- Decision: `{payload['summary']['decision']}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- min_rusticl_ratio: `{payload['thresholds']['min_rusticl_ratio']}`")
    lines.append(f"- max_t5_overhead_percent: `{payload['thresholds']['max_t5_overhead_percent']}`")
    lines.append(f"- max_t5_disable_events: `{payload['thresholds']['max_t5_disable_events']}`")
    lines.append("")
    lines.append("## Alerts")
    lines.append("")
    if payload["alerts"]:
        lines.append("| Severity | Code | Observed | Threshold | Message |")
        lines.append("| --- | --- | --- | --- | --- |")
        for alert in payload["alerts"]:
            lines.append(
                f"| {alert['severity']} | {alert['code']} | "
                f"{alert.get('observed')} | {alert.get('threshold')} | {alert['message']} |"
            )
    else:
        lines.append("- No alerts triggered.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 2 - Monthly Scheduler Automation")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append(f"- Workflow path: `{payload['metadata']['workflow_path']}`")
    lines.append(f"- Artifact retention days: `{payload['metadata']['artifact_retention_days']}`")
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
    lines.append(f"- Cycle decision: `{payload['highlights']['cycle_decision']}`")
    lines.append(f"- Alerts decision: `{payload['highlights']['alerts_decision']}`")
    lines.append(
        f"- split_ratio_min: `{float(payload['highlights']['split_ratio_min']):.6f}`"
    )
    lines.append(
        f"- split_t5_overhead_max: `{float(payload['highlights']['split_t5_overhead_max']):.6f}`"
    )
    lines.append(
        f"- split_t5_disable_total: `{int(payload['highlights']['split_t5_disable_total'])}`"
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


def _operational_debt_payload() -> dict[str, Any]:
    debts = [
        {
            "debt_id": "week20_block2_scheduler_dispatch_health",
            "area": "automation",
            "severity": "medium",
            "status": "open",
            "owner": "ops-automation",
            "description": "Continuar monitoreo de fiabilidad del trigger schedule mensual en CI.",
            "mitigation": "Agregar verificacion automatica del ultimo run schedule + alerta de ausencia.",
            "target_window": "week20_block3",
        },
        {
            "debt_id": "week20_block2_external_alert_bridge",
            "area": "operations",
            "severity": "medium",
            "status": "open",
            "owner": "ops-observability",
            "description": "Conectar alert summary de Block2 con canal externo (webhook/chatops).",
            "mitigation": "Publicar adaptador minimo de webhook para eventos critical/high.",
            "target_window": "week21_block1",
        },
    ]
    return {
        "matrix_id": "week20-block2-monthly-scheduler-automation-debt-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": debts,
        "summary": {
            "total": len(debts),
            "open": sum(1 for debt in debts if debt["status"] == "open"),
            "high_or_critical_open": sum(
                1
                for debt in debts
                if debt["status"] == "open" and debt["severity"] in {"high", "critical"}
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week20 Block2 monthly scheduler automation with retention and alerts."
    )
    parser.add_argument(
        "--mode",
        choices=("local", "ci"),
        default="local",
    )
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--workflow-path",
        default=".github/workflows/week20-monthly-cycle.yml",
    )
    parser.add_argument("--artifact-retention-days", type=int, default=45)
    parser.add_argument(
        "--cycle-runner-path",
        default="research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py",
    )
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json",
    )
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json",
    )
    parser.add_argument(
        "--baseline-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_canary_20260211_015619.json",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--kernels", nargs="+", default=["auto_t3_controlled", "auto_t5_guarded"])
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses", type=int, default=2)
    parser.add_argument("--weekly-seed", type=int, default=20011)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[211, 509])
    parser.add_argument("--min-rusticl-ratio", type=float, default=0.85)
    parser.add_argument("--max-t5-overhead-percent", type=float, default=5.0)
    parser.add_argument("--max-t5-disable-events", type=int, default=0)
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week20_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week20_block2_monthly_scheduler_automation")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    workflow_path = _resolve(args.workflow_path)
    cycle_runner_path = _resolve(args.cycle_runner_path)
    policy_path = _resolve(args.policy_path)
    t5_policy_path = _resolve(args.t5_policy_path)
    baseline_path = _resolve(args.baseline_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest: dict[str, Any] = {}
    stable_tag = "unknown"
    if stable_manifest_path.exists():
        stable_manifest = _read_json(stable_manifest_path)
        stable_tag = str(stable_manifest.get("stable_tag", "unknown"))

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

    compile_cmd = ["./venv/bin/python", "-m", "py_compile", str(cycle_runner_path)]
    compile_runner = _run(compile_cmd)

    cycle_cmd = [
        "./venv/bin/python",
        str(cycle_runner_path),
        "--policy-path",
        str(policy_path),
        "--t5-policy-path",
        str(t5_policy_path),
        "--baseline-path",
        str(baseline_path),
        "--sizes",
        *[str(size) for size in args.sizes],
        "--kernels",
        *[str(kernel) for kernel in args.kernels],
        "--snapshots",
        str(args.snapshots),
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--pressure-size",
        str(args.pressure_size),
        "--pressure-iterations",
        str(args.pressure_iterations),
        "--pressure-pulses",
        str(args.pressure_pulses),
        "--weekly-seed",
        str(args.weekly_seed),
        "--split-seeds",
        *[str(seed) for seed in args.split_seeds],
        "--min-rusticl-ratio",
        str(args.min_rusticl_ratio),
        "--report-dir",
        str(args.report_dir),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_cycle",
    ]
    cycle_run = _run(cycle_cmd)
    cycle_report_json = _extract_prefixed_line(cycle_run["stdout"], "Week20 block1 JSON:")
    cycle_report_md = _extract_prefixed_line(cycle_run["stdout"], "Week20 block1 MD:")
    cycle_dashboard_json = _extract_prefixed_line(cycle_run["stdout"], "Dashboard JSON:")
    cycle_dashboard_md = _extract_prefixed_line(cycle_run["stdout"], "Dashboard MD:")
    cycle_manifest = _extract_prefixed_line(cycle_run["stdout"], "Manifest:")
    cycle_decision = str(_extract_prefixed_line(cycle_run["stdout"], "Decision:") or "unknown")

    cycle_payload = _read_json(_resolve(cycle_report_json)) if cycle_report_json else {}
    split_ratio_min = float(cycle_payload.get("highlights", {}).get("split_ratio_min", 0.0))
    split_t5_overhead_max = float(
        cycle_payload.get("highlights", {}).get("split_t5_overhead_max", 999.0)
    )
    split_t5_disable_total = int(
        cycle_payload.get("highlights", {}).get("split_t5_disable_total", 999)
    )
    cycle_pre_gate = str(cycle_payload.get("evaluation", {}).get("checks", {}).get("pre_gate_promote", {}).get("observed", "unknown"))
    cycle_post_gate = str(cycle_payload.get("evaluation", {}).get("checks", {}).get("post_gate_promote", {}).get("observed", "unknown"))

    alerts: list[dict[str, Any]] = []
    if cycle_decision.lower() != "promote":
        alerts.append(
            {
                "severity": "critical",
                "code": "cycle_decision_not_promote",
                "observed": cycle_decision.lower(),
                "threshold": "promote",
                "message": "Monthly cycle did not finish in promote.",
            }
        )
    if split_ratio_min < float(args.min_rusticl_ratio):
        alerts.append(
            {
                "severity": "high",
                "code": "rusticl_ratio_below_floor",
                "observed": split_ratio_min,
                "threshold": float(args.min_rusticl_ratio),
                "message": "Rusticl/Clover split ratio is below floor.",
            }
        )
    if split_t5_overhead_max > float(args.max_t5_overhead_percent):
        alerts.append(
            {
                "severity": "high",
                "code": "t5_overhead_above_limit",
                "observed": split_t5_overhead_max,
                "threshold": float(args.max_t5_overhead_percent),
                "message": "T5 overhead exceeds operational ceiling.",
            }
        )
    if split_t5_disable_total > int(args.max_t5_disable_events):
        alerts.append(
            {
                "severity": "critical",
                "code": "t5_disable_events_nonzero",
                "observed": split_t5_disable_total,
                "threshold": int(args.max_t5_disable_events),
                "message": "T5 disable events exceeded allowed value.",
            }
        )
    if cycle_pre_gate != "promote" or cycle_post_gate != "promote":
        alerts.append(
            {
                "severity": "high",
                "code": "cycle_canonical_gate_not_promote",
                "observed": f"pre={cycle_pre_gate}, post={cycle_post_gate}",
                "threshold": "pre=promote, post=promote",
                "message": "Canonical gate inside cycle is not fully green.",
            }
        )

    alerts_decision = "promote" if not alerts else "iterate"
    alerts_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_report": cycle_report_json,
            "stable_tag": stable_tag,
        },
        "thresholds": {
            "min_rusticl_ratio": float(args.min_rusticl_ratio),
            "max_t5_overhead_percent": float(args.max_t5_overhead_percent),
            "max_t5_disable_events": int(args.max_t5_disable_events),
        },
        "alerts": alerts,
        "summary": {
            "alerts_total": len(alerts),
            "critical_total": sum(1 for alert in alerts if alert["severity"] == "critical"),
            "decision": alerts_decision,
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    alerts_json = output_dir / f"{args.output_prefix}_alerts_{stamp}.json"
    alerts_md = output_dir / f"{args.output_prefix}_alerts_{stamp}.md"
    _write_json(alerts_json, alerts_payload)
    alerts_md.write_text(_alerts_md(alerts_payload))

    workflow_text = workflow_path.read_text() if workflow_path.exists() else ""
    workflow_has_schedule = "schedule:" in workflow_text and "cron:" in workflow_text
    workflow_has_retention = f"retention-days: {int(args.artifact_retention_days)}" in workflow_text
    workflow_has_alert_summary = "Alert summary gate" in workflow_text

    scheduler_spec = {
        "scheduler_id": "week20-block2-monthly-cycle-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_path": str(workflow_path),
        "schedule_cron_utc": "0 4 1 * *",
        "artifact_retention_days": int(args.artifact_retention_days),
        "alert_thresholds": {
            "min_rusticl_ratio": float(args.min_rusticl_ratio),
            "max_t5_overhead_percent": float(args.max_t5_overhead_percent),
            "max_t5_disable_events": int(args.max_t5_disable_events),
        },
    }
    scheduler_spec_json = output_dir / f"{args.output_prefix}_scheduler_spec_{stamp}.json"
    _write_json(scheduler_spec_json, scheduler_spec)

    debt_payload = _operational_debt_payload()
    debt_json = output_dir / f"{args.output_prefix}_operational_debt_{stamp}.json"
    _write_json(debt_json, debt_payload)

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
    checks["workflow_exists"] = {
        "observed": workflow_path.exists(),
        "required": True,
        "pass": workflow_path.exists(),
    }
    checks["workflow_has_schedule"] = {
        "observed": workflow_has_schedule,
        "required": True,
        "pass": workflow_has_schedule,
    }
    checks["workflow_has_retention_days"] = {
        "observed": workflow_has_retention,
        "required": True,
        "pass": workflow_has_retention,
    }
    checks["workflow_has_alert_summary_step"] = {
        "observed": workflow_has_alert_summary,
        "required": True,
        "pass": workflow_has_alert_summary,
    }
    checks["cycle_runner_compiles"] = {
        "observed_returncode": int(compile_runner["returncode"]),
        "required_returncode": 0,
        "pass": int(compile_runner["returncode"]) == 0,
    }
    checks["cycle_report_exists"] = {
        "observed": bool(cycle_report_json),
        "required": True,
        "pass": bool(cycle_report_json),
    }
    checks["cycle_decision_promote"] = {
        "observed": cycle_decision.lower(),
        "required": "promote",
        "pass": cycle_decision.lower() == "promote",
    }
    checks["alerts_decision_promote"] = {
        "observed": alerts_decision,
        "required": "promote",
        "pass": alerts_decision == "promote",
    }
    checks["no_high_critical_open_debt"] = {
        "observed": int(debt_payload.get("summary", {}).get("high_or_critical_open", 99)),
        "required_max": 0,
        "pass": int(debt_payload.get("summary", {}).get("high_or_critical_open", 99)) <= 0,
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
        "Monthly scheduler automation, retention policy, and alerting remained fully green."
        if decision == "promote"
        else "Scheduler automation found unresolved issues in workflow, cycle execution, alerts, or gates."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "workflow_path": str(workflow_path),
            "artifact_retention_days": int(args.artifact_retention_days),
            "mode": args.mode,
        },
        "commands": {
            "pre_gate": pre_gate,
            "compile_runner": compile_runner,
            "cycle_run": cycle_run,
            "post_gate": post_gate,
            "pre_gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
            "cycle_cmd_pretty": " ".join(shlex.quote(x) for x in cycle_cmd),
        },
        "artifacts": {
            "cycle_report_json": cycle_report_json,
            "cycle_report_md": cycle_report_md,
            "cycle_dashboard_json": cycle_dashboard_json,
            "cycle_dashboard_md": cycle_dashboard_md,
            "cycle_manifest": cycle_manifest,
            "alerts_json": str(alerts_json),
            "alerts_md": str(alerts_md),
            "scheduler_spec_json": str(scheduler_spec_json),
            "operational_debt_json": str(debt_json),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "cycle_decision": cycle_decision,
            "alerts_decision": alerts_decision,
            "split_ratio_min": split_ratio_min,
            "split_t5_overhead_max": split_t5_overhead_max,
            "split_t5_disable_total": split_t5_disable_total,
            "alerts_total": len(alerts),
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

    print(f"Week20 block2 JSON: {report_json}")
    print(f"Week20 block2 MD:   {report_md}")
    print(f"Alerts JSON:        {alerts_json}")
    print(f"Alerts MD:          {alerts_md}")
    print(f"Scheduler spec:     {scheduler_spec_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
