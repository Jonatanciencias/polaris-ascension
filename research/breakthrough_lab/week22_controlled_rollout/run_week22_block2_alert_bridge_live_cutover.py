#!/usr/bin/env python3
"""Week 22 Block 2: live alert-bridge cutover + explicit rollback path."""

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


def _dispatch_payload(*, payload: dict[str, Any], endpoint: str | None, dispatch_mode: str) -> dict[str, Any]:
    if dispatch_mode == "dry_run":
        return {
            "mode": "dry_run",
            "sent": False,
            "http_status": None,
            "error": None,
        }

    if not endpoint:
        return {
            "mode": "live",
            "sent": False,
            "http_status": None,
            "error": "endpoint_missing",
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
            status = int(response.status)
            return {
                "mode": "live",
                "sent": 200 <= status < 300,
                "http_status": status,
                "error": None if 200 <= status < 300 else f"http_status_{status}",
            }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "mode": "live",
            "sent": False,
            "http_status": None,
            "error": str(exc),
        }


def _build_rollback_record(
    *,
    bridge_payload: dict[str, Any],
    dispatch_result: dict[str, Any],
    rollback_enabled: bool,
    rollback_channel: str,
    output_dir: Path,
    output_prefix: str,
    stamp: str,
) -> tuple[dict[str, Any], Path | None]:
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rollback_enabled": bool(rollback_enabled),
        "rollback_channel": rollback_channel,
        "dispatch_mode": str(dispatch_result.get("mode", "unknown")),
        "dispatch_sent": bool(dispatch_result.get("sent", False)),
        "triggered": False,
        "action": "none",
        "reason": "not_required",
        "spool_payload_json": None,
    }
    spool_path: Path | None = None

    if rollback_enabled and dispatch_result.get("mode") == "live" and not bool(dispatch_result.get("sent")):
        spool_path = output_dir / f"{output_prefix}_rollback_spool_{stamp}.json"
        _write_json(spool_path, bridge_payload)
        record.update(
            {
                "triggered": True,
                "action": "spool_payload_for_retry",
                "reason": "live_dispatch_failed",
                "spool_payload_json": str(spool_path),
            }
        )

    return record, spool_path


def _operational_debt_payload(*, dispatch_result: dict[str, Any], rollback_record: dict[str, Any]) -> dict[str, Any]:
    live_cutover_closed = bool(dispatch_result.get("mode") == "live" and dispatch_result.get("sent"))
    rollback_path_closed = bool(
        rollback_record.get("rollback_enabled")
        and (
            rollback_record.get("dispatch_sent")
            or rollback_record.get("triggered")
            or rollback_record.get("reason") == "not_required"
        )
    )

    debts = [
        {
            "debt_id": "week22_block2_webhook_live_cutover",
            "area": "operations",
            "severity": "medium",
            "status": "closed" if live_cutover_closed else "open",
            "owner": "ops-observability",
            "description": "Completar cutover del bridge de alertas de dry-run a endpoint webhook real.",
            "mitigation": "Mantener endpoint controlado y monitorear delivery success 2xx.",
            "target_window": "week22_block2",
        },
        {
            "debt_id": "week22_block2_rollback_explicit_path",
            "area": "operations",
            "severity": "medium",
            "status": "closed" if rollback_path_closed else "open",
            "owner": "ops-observability",
            "description": "Definir y validar rollback explícito de canal ante falla de dispatch live.",
            "mitigation": "Registrar rollback record y spool de payload para retry seguro.",
            "target_window": "week22_block2",
        },
    ]
    open_count = sum(1 for debt in debts if debt["status"] == "open")
    high_critical_open = sum(
        1 for debt in debts if debt["status"] == "open" and debt["severity"] in {"high", "critical"}
    )
    return {
        "matrix_id": "week22-block2-alert-bridge-live-cutover-debt-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": debts,
        "summary": {
            "total": len(debts),
            "open": open_count,
            "high_or_critical_open": high_critical_open,
        },
    }


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 22 Block 2 - Alert Bridge Live Cutover + Rollback")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Dispatch mode: `{report['metadata']['dispatch_mode']}`")
    lines.append(f"- Rollback channel: `{report['metadata']['rollback_channel']}`")
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
    lines.append(f"- source_cycle_decision: `{report['highlights']['source_cycle_decision']}`")
    lines.append(f"- source_alerts_decision: `{report['highlights']['source_alerts_decision']}`")
    lines.append(f"- bridged_alerts_count: `{int(report['highlights']['bridged_alerts_count'])}`")
    lines.append(f"- dispatch_mode: `{report['highlights']['dispatch_mode']}`")
    lines.append(f"- dispatch_sent: `{bool(report['highlights']['dispatch_sent'])}`")
    lines.append(f"- rollback_triggered: `{bool(report['highlights']['rollback_triggered'])}`")
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
        description="Run Week22 Block2 alert bridge live cutover with explicit rollback.",
    )
    parser.add_argument("--mode", choices=("local", "ci"), default="local")
    parser.add_argument("--dispatch-mode", choices=("dry_run", "live"), default="live")
    parser.add_argument("--rollback-on-dispatch-failure", action="store_true", default=True)
    parser.add_argument("--rollback-channel", default="spool-fallback")
    parser.add_argument("--bridge-endpoint-env", default="WEEKLY_ALERT_WEBHOOK_URL")
    parser.add_argument("--bridge-endpoint", default="")
    parser.add_argument("--bridge-channel", default="chatops-webhook")
    parser.add_argument("--workflow-path", default=".github/workflows/week20-monthly-cycle.yml")
    parser.add_argument(
        "--scheduler-spec-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_scheduler_spec_20260211_140738.json",
    )
    parser.add_argument(
        "--source-cycle-report-path",
        default="research/breakthrough_lab/week22_controlled_rollout/week22_block1_monthly_continuity_20260211_155815.json",
    )
    parser.add_argument(
        "--source-alerts-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_20260211_140738.json",
    )
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week22_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week22_block2_alert_bridge_live_cutover")
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
            pre_payload.get("evaluation", {}).get("checks", {}).get("pytest_tier_green", {}).get("pass", False)
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
        "bridge_id": "week22-block2-alert-bridge-live-cutover-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "channel": args.bridge_channel,
        "mode": args.mode,
        "dispatch_mode": args.dispatch_mode,
        "source_cycle_report": str(source_cycle_path),
        "source_cycle_decision": source_cycle_decision,
        "source_alerts_path": str(source_alerts_path),
        "source_alerts_decision": source_alerts_decision,
        "alerts": bridged_alerts,
    }

    endpoint = args.bridge_endpoint or os.environ.get(args.bridge_endpoint_env, "")
    endpoint = endpoint.strip()
    dispatch_result = _dispatch_payload(payload=bridge_payload, endpoint=endpoint or None, dispatch_mode=args.dispatch_mode)

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
            "endpoint": endpoint,
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

    rollback_record, rollback_spool_path = _build_rollback_record(
        bridge_payload=bridge_payload,
        dispatch_result=dispatch_result,
        rollback_enabled=args.rollback_on_dispatch_failure,
        rollback_channel=args.rollback_channel,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        stamp=stamp,
    )
    rollback_json = output_dir / f"{args.output_prefix}_rollback_{stamp}.json"
    _write_json(rollback_json, rollback_record)

    debt_payload = _operational_debt_payload(dispatch_result=dispatch_result, rollback_record=rollback_record)
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
            post_payload.get("evaluation", {}).get("checks", {}).get("pytest_tier_green", {}).get("pass", False)
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
    checks["dispatch_mode_live"] = {
        "observed": args.dispatch_mode,
        "required": "live",
        "pass": args.dispatch_mode == "live",
    }
    checks["endpoint_present_for_live"] = {
        "observed": bool(endpoint),
        "required": True,
        "pass": bool(endpoint),
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
    checks["dispatch_live_success"] = {
        "observed": dispatch_result,
        "required": "sent_true_http_2xx",
        "pass": bool(dispatch_result.get("sent")) and dispatch_result.get("mode") == "live",
    }
    checks["rollback_record_written"] = {
        "observed": rollback_json.exists(),
        "required": True,
        "pass": rollback_json.exists(),
    }
    checks["rollback_triggered_on_failure_or_not_needed"] = {
        "observed": {
            "dispatch_sent": bool(dispatch_result.get("sent")),
            "rollback_triggered": bool(rollback_record.get("triggered")),
            "rollback_reason": rollback_record.get("reason"),
        },
        "required": "dispatch_success_or_triggered_fallback",
        "pass": bool(dispatch_result.get("sent")) or bool(rollback_record.get("triggered")),
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
        "Live webhook cutover succeeded with explicit rollback path ready and canonical gates green."
        if decision == "promote"
        else "Live cutover found unresolved dispatch, rollback, scheduler, or validation failures."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow_path": str(workflow_path),
            "scheduler_spec_path": str(scheduler_spec_path),
            "source_cycle_report_path": str(source_cycle_path),
            "source_alerts_path": str(source_alerts_path),
            "mode": args.mode,
            "dispatch_mode": args.dispatch_mode,
            "rollback_channel": args.rollback_channel,
            "endpoint_env": args.bridge_endpoint_env,
            "endpoint_present": bool(endpoint),
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
            "rollback_json": str(rollback_json),
            "rollback_spool_json": str(rollback_spool_path) if rollback_spool_path else "",
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
            "dispatch_http_status": dispatch_result.get("http_status"),
            "rollback_triggered": bool(rollback_record.get("triggered")),
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

    print(f"Week22 block2 JSON: {report_json}")
    print(f"Week22 block2 MD:   {report_md}")
    print(f"Bridge payload:     {bridge_payload_json}")
    print(f"Dispatch record:    {dispatch_json}")
    print(f"Rollback record:    {rollback_json}")
    print(f"Scheduler health:   {scheduler_health_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
