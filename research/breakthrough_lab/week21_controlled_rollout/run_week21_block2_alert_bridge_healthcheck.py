#!/usr/bin/env python3
"""Week 21 Block 2: external alert bridge + monthly scheduler health-check."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
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


def _scheduler_health(workflow_text: str, scheduler_spec: dict[str, Any]) -> dict[str, Any]:
    expected_cron = str(scheduler_spec.get("schedule_cron_utc", ""))
    expected_retention = int(scheduler_spec.get("artifact_retention_days", 0))
    has_schedule = "schedule:" in workflow_text and "cron:" in workflow_text
    has_expected_cron = expected_cron in workflow_text if expected_cron else False
    has_upload_artifact = "actions/upload-artifact@v4" in workflow_text
    has_expected_retention = f"retention-days: {expected_retention}" in workflow_text
    has_alert_gate = "Alert summary gate" in workflow_text
    return {
        "expected_cron_utc": expected_cron,
        "expected_retention_days": expected_retention,
        "checks": {
            "has_schedule": has_schedule,
            "has_expected_cron": has_expected_cron,
            "has_upload_artifact_step": has_upload_artifact,
            "has_expected_retention_days": has_expected_retention,
            "has_alert_summary_gate": has_alert_gate,
        },
    }


def _dispatch_payload(
    *,
    payload: dict[str, Any],
    endpoint: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run or not endpoint:
        return {
            "mode": "dry_run",
            "sent": False,
            "http_status": None,
            "error": None,
        }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {
                "mode": "live",
                "sent": True,
                "http_status": int(response.status),
                "error": None,
            }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "mode": "live",
            "sent": False,
            "http_status": None,
            "error": str(exc),
        }


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 21 Block 2 - Alert Bridge + Scheduler Health-check")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Workflow path: `{report['metadata']['workflow_path']}`")
    lines.append(f"- Bridge mode: `{report['highlights']['dispatch_mode']}`")
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
    lines.append(f"- Source cycle decision: `{report['highlights']['source_cycle_decision']}`")
    lines.append(f"- Source alerts decision: `{report['highlights']['source_alerts_decision']}`")
    lines.append(f"- Bridged alerts count: `{int(report['highlights']['bridged_alerts_count'])}`")
    lines.append(f"- Heartbeat emitted: `{bool(report['highlights']['heartbeat_emitted'])}`")
    lines.append(f"- Dispatch sent: `{bool(report['highlights']['dispatch_sent'])}`")
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
        description="Run Week21 Block2 alert bridge and monthly scheduler health-check."
    )
    parser.add_argument("--mode", choices=("local", "ci"), default="local")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--workflow-path",
        default=".github/workflows/week20-monthly-cycle.yml",
    )
    parser.add_argument(
        "--scheduler-spec-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_scheduler_spec_20260211_140738.json",
    )
    parser.add_argument(
        "--source-cycle-report-path",
        default="research/breakthrough_lab/week21_controlled_rollout/week21_block1_monthly_continuity_20260211_142611.json",
    )
    parser.add_argument(
        "--source-alerts-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_20260211_140738.json",
    )
    parser.add_argument(
        "--bridge-endpoint-env",
        default="WEEKLY_ALERT_WEBHOOK_URL",
    )
    parser.add_argument(
        "--bridge-channel",
        default="chatops-webhook",
    )
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week21_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week21_block2_alert_bridge_healthcheck")
    args = parser.parse_args()

    workflow_path = _resolve(args.workflow_path)
    scheduler_spec_path = _resolve(args.scheduler_spec_path)
    source_cycle_path = _resolve(args.source_cycle_report_path)
    source_alerts_path = _resolve(args.source_alerts_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    scheduler_spec = _read_json(scheduler_spec_path) if scheduler_spec_path.exists() else {}
    source_cycle = _read_json(source_cycle_path) if source_cycle_path.exists() else {}
    source_alerts = _read_json(source_alerts_path) if source_alerts_path.exists() else {}
    workflow_text = workflow_path.read_text() if workflow_path.exists() else ""

    scheduler_health = _scheduler_health(workflow_text, scheduler_spec)
    scheduler_checks = scheduler_health["checks"]
    scheduler_healthy = all(bool(value) for value in scheduler_checks.values())

    source_cycle_decision = str(source_cycle.get("evaluation", {}).get("decision", "unknown"))
    source_alerts_decision = str(source_alerts.get("summary", {}).get("decision", "unknown"))
    bridged_alerts = list(source_alerts.get("alerts", []))
    heartbeat_emitted = False
    if not bridged_alerts:
        heartbeat_emitted = True
        bridged_alerts.append(
            {
                "severity": "info",
                "code": "heartbeat_no_active_alerts",
                "message": "No active alerts in source payload; heartbeat emitted.",
            }
        )

    bridge_payload = {
        "bridge_id": "week21-block2-alert-bridge-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "channel": args.bridge_channel,
        "mode": args.mode,
        "source_cycle_report": str(source_cycle_path),
        "source_cycle_decision": source_cycle_decision,
        "source_alerts_path": str(source_alerts_path),
        "source_alerts_decision": source_alerts_decision,
        "alerts": bridged_alerts,
    }

    endpoint = os.environ.get(args.bridge_endpoint_env)
    dispatch_result = _dispatch_payload(payload=bridge_payload, endpoint=endpoint, dry_run=args.dry_run)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bridge_payload_json = output_dir / f"{args.output_prefix}_bridge_payload_{stamp}.json"
    _write_json(bridge_payload_json, bridge_payload)
    dispatch_json = output_dir / f"{args.output_prefix}_dispatch_{stamp}.json"
    _write_json(
        dispatch_json,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint_env": args.bridge_endpoint_env,
            "endpoint_present": bool(endpoint),
            "result": dispatch_result,
        },
    )
    scheduler_health_json = output_dir / f"{args.output_prefix}_scheduler_health_{stamp}.json"
    _write_json(
        scheduler_health_json,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow_path": str(workflow_path),
            "scheduler_spec_path": str(scheduler_spec_path),
            "health": scheduler_health,
        },
    )

    debt_payload = {
        "matrix_id": "week21-block2-alert-bridge-debt-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": [
            {
                "debt_id": "week21_block2_webhook_live_cutover",
                "area": "operations",
                "severity": "medium",
                "status": "open",
                "owner": "ops-observability",
                "description": "Completar cutover de bridge de alertas de dry-run a endpoint real.",
                "mitigation": "Configurar `WEEKLY_ALERT_WEBHOOK_URL` en entorno de despliegue controlado.",
                "target_window": "week22_block1",
            }
        ],
        "summary": {
            "total": 1,
            "open": 1,
            "high_or_critical_open": 0,
        },
    }
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
    checks["workflow_exists"] = {
        "observed": workflow_path.exists(),
        "required": True,
        "pass": workflow_path.exists(),
    }
    checks["scheduler_spec_exists"] = {
        "observed": scheduler_spec_path.exists(),
        "required": True,
        "pass": scheduler_spec_path.exists(),
    }
    checks["scheduler_health_all_checks"] = {
        "observed": scheduler_checks,
        "required": "all_true",
        "pass": scheduler_healthy,
    }
    checks["source_cycle_promote"] = {
        "observed": source_cycle_decision,
        "required": "promote",
        "pass": source_cycle_decision == "promote",
    }
    checks["source_alerts_promote"] = {
        "observed": source_alerts_decision,
        "required": "promote",
        "pass": source_alerts_decision == "promote",
    }
    checks["bridge_payload_written"] = {
        "observed": bridge_payload_json.exists(),
        "required": True,
        "pass": bridge_payload_json.exists(),
    }
    checks["dispatch_record_written"] = {
        "observed": dispatch_json.exists(),
        "required": True,
        "pass": dispatch_json.exists(),
    }
    checks["dispatch_successful_or_dry_run"] = {
        "observed": dispatch_result,
        "required": "dry_run_or_sent_true",
        "pass": (dispatch_result.get("mode") == "dry_run") or bool(dispatch_result.get("sent")),
    }
    checks["no_high_critical_open_debt"] = {
        "observed": int(debt_payload["summary"]["high_or_critical_open"]),
        "required_max": 0,
        "pass": int(debt_payload["summary"]["high_or_critical_open"]) <= 0,
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
        "Alert bridge dry-run and scheduler health-check are stable with canonical gates green."
        if decision == "promote"
        else "Bridge health-check found unresolved scheduler, dispatch, or validation issues."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow_path": str(workflow_path),
            "scheduler_spec_path": str(scheduler_spec_path),
            "source_cycle_report_path": str(source_cycle_path),
            "source_alerts_path": str(source_alerts_path),
            "mode": args.mode,
            "dry_run": bool(args.dry_run),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "pre_gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "artifacts": {
            "bridge_payload_json": str(bridge_payload_json),
            "dispatch_json": str(dispatch_json),
            "scheduler_health_json": str(scheduler_health_json),
            "operational_debt_json": str(debt_json),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "source_cycle_decision": source_cycle_decision,
            "source_alerts_decision": source_alerts_decision,
            "bridged_alerts_count": len(bridged_alerts),
            "heartbeat_emitted": heartbeat_emitted,
            "dispatch_mode": str(dispatch_result.get("mode")),
            "dispatch_sent": bool(dispatch_result.get("sent")),
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
    report_md.write_text(_md(report))

    print(f"Week21 block2 JSON: {report_json}")
    print(f"Week21 block2 MD:   {report_md}")
    print(f"Bridge payload:     {bridge_payload_json}")
    print(f"Dispatch record:    {dispatch_json}")
    print(f"Scheduler health:   {scheduler_health_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
