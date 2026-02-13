#!/usr/bin/env python3
"""Week 23 Block 2: live alert bridge hardening (retry/backoff + delivery health-check)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


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
    checks = {
        "has_schedule": "schedule:" in workflow_text and "cron:" in workflow_text,
        "has_expected_cron": expected_cron in workflow_text if expected_cron else False,
        "has_upload_artifact_step": "actions/upload-artifact@v4" in workflow_text,
        "has_expected_retention_days": f"retention-days: {expected_retention}" in workflow_text,
        "has_alert_summary_gate": "Alert summary gate" in workflow_text,
    }
    return {
        "expected_cron_utc": expected_cron,
        "expected_retention_days": expected_retention,
        "checks": checks,
    }


def _build_health_url(endpoint: str, healthcheck_path: str) -> str:
    parsed = urlparse(endpoint)
    return urlunparse((parsed.scheme, parsed.netloc, healthcheck_path, "", "", ""))


def _http_get(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            return {
                "url": url,
                "status": status,
                "ok": 200 <= status < 300,
                "error": None,
            }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "url": url,
            "status": None,
            "ok": False,
            "error": str(exc),
        }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            return {
                "sent": 200 <= status < 300,
                "http_status": status,
                "error": None if 200 <= status < 300 else f"http_status_{status}",
            }
    except urllib.error.HTTPError as exc:  # pragma: no cover
        return {
            "sent": False,
            "http_status": int(exc.code),
            "error": str(exc),
        }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "sent": False,
            "http_status": None,
            "error": str(exc),
        }


def _dispatch_with_retry(
    *,
    endpoint: str,
    payload: dict[str, Any],
    retry_attempts: int,
    backoff_seconds: float,
    backoff_multiplier: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    wait_seconds = float(backoff_seconds)
    final_sent = False
    final_status: int | None = None
    final_error: str | None = None

    for attempt in range(1, retry_attempts + 1):
        start = time.monotonic()
        outcome = _post_json(endpoint, payload, timeout_seconds)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        attempt_row = {
            "attempt": attempt,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sent": bool(outcome.get("sent")),
            "http_status": outcome.get("http_status"),
            "error": outcome.get("error"),
            "elapsed_ms": elapsed_ms,
            "backoff_applied_seconds": 0.0,
        }

        attempts.append(attempt_row)
        if bool(outcome.get("sent")):
            final_sent = True
            final_status = int(outcome.get("http_status") or 0)
            final_error = None
            break

        final_sent = False
        final_status = int(outcome["http_status"]) if outcome.get("http_status") is not None else None
        final_error = str(outcome.get("error")) if outcome.get("error") else "unknown_error"

        if attempt < retry_attempts:
            attempts[-1]["backoff_applied_seconds"] = float(wait_seconds)
            time.sleep(wait_seconds)
            wait_seconds *= float(backoff_multiplier)

    return {
        "mode": "live",
        "sent": final_sent,
        "http_status": final_status,
        "error": final_error,
        "attempts": attempts,
        "attempts_executed": len(attempts),
        "retries_used": len(attempts) > 1,
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
        "dispatch_sent": bool(dispatch_result.get("sent", False)),
        "triggered": False,
        "action": "none",
        "reason": "not_required",
        "spool_payload_json": None,
    }
    spool_path: Path | None = None

    if rollback_enabled and not bool(dispatch_result.get("sent")):
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


def _operational_debt_payload(
    *,
    dispatch_result: dict[str, Any],
    pre_health: dict[str, Any],
    post_health: dict[str, Any],
    retries_used: bool,
) -> dict[str, Any]:
    debts = [
        {
            "debt_id": "week23_block2_retry_backoff_hardening",
            "area": "operations",
            "severity": "medium",
            "status": "closed" if retries_used else "open",
            "owner": "ops-observability",
            "description": "Validar ruta de retry/backoff en dispatch live ante falla transitoria.",
            "mitigation": "Mantener retry determinista con backoff exponencial acotado.",
            "target_window": "week23_block2",
        },
        {
            "debt_id": "week23_block2_delivery_healthcheck",
            "area": "operations",
            "severity": "medium",
            "status": "closed" if bool(pre_health.get("ok")) and bool(post_health.get("ok")) else "open",
            "owner": "ops-observability",
            "description": "Monitorear salud del endpoint de delivery antes y despues del dispatch.",
            "mitigation": "Agregar health-check GET pre/post al bridge y alertar si falla.",
            "target_window": "week23_block2",
        },
        {
            "debt_id": "week23_block2_live_dispatch_reliability",
            "area": "operations",
            "severity": "medium",
            "status": "closed" if bool(dispatch_result.get("sent")) else "open",
            "owner": "ops-observability",
            "description": "Asegurar entrega live del bridge con resultado 2xx.",
            "mitigation": "Continuar monitoreo de success ratio y fallback controlado.",
            "target_window": "week23_block2",
        },
    ]

    open_count = sum(1 for debt in debts if debt["status"] == "open")
    high_critical_open = sum(
        1 for debt in debts if debt["status"] == "open" and debt["severity"] in {"high", "critical"}
    )
    return {
        "matrix_id": "week23-block2-alert-bridge-hardening-debt-v1",
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
    lines.append("# Week 23 Block 2 - Alert Bridge Hardening")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Endpoint: `{report['metadata']['endpoint']}`")
    lines.append(f"- Health URL: `{report['metadata']['health_url']}`")
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
    lines.append(f"- dispatch_sent: `{bool(report['highlights']['dispatch_sent'])}`")
    lines.append(f"- attempts_executed: `{int(report['highlights']['attempts_executed'])}`")
    lines.append(f"- retries_used: `{bool(report['highlights']['retries_used'])}`")
    lines.append(f"- pre_health_ok: `{bool(report['highlights']['pre_health_ok'])}`")
    lines.append(f"- post_health_ok: `{bool(report['highlights']['post_health_ok'])}`")
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
        description="Run Week23 Block2 alert bridge hardening with retry/backoff and delivery health-check.",
    )
    parser.add_argument("--mode", choices=("local", "ci"), default="local")
    parser.add_argument("--bridge-endpoint-env", default="WEEKLY_ALERT_WEBHOOK_URL")
    parser.add_argument("--bridge-endpoint", default="")
    parser.add_argument("--healthcheck-path", default="/health")
    parser.add_argument("--request-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=0.4)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--rollback-on-dispatch-failure", action="store_true", default=True)
    parser.add_argument("--rollback-channel", default="spool-fallback")
    parser.add_argument("--bridge-channel", default="chatops-webhook")
    parser.add_argument("--workflow-path", default=".github/workflows/week20-monthly-cycle.yml")
    parser.add_argument(
        "--scheduler-spec-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_scheduler_spec_20260211_140738.json",
    )
    parser.add_argument(
        "--source-cycle-report-path",
        default="research/breakthrough_lab/week23_controlled_rollout/week23_block1_monthly_continuity_20260212_003814.json",
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
        default="research/breakthrough_lab/week23_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week23_block2_alert_bridge_hardening")
    args = parser.parse_args()

    if args.retry_attempts < 1:
        raise SystemExit("--retry-attempts must be >= 1")

    workflow_path = _resolve(args.workflow_path)
    scheduler_spec_path = _resolve(args.scheduler_spec_path)
    source_cycle_path = _resolve(args.source_cycle_report_path)
    source_alerts_path = _resolve(args.source_alerts_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = (args.bridge_endpoint or os.environ.get(args.bridge_endpoint_env, "")).strip()

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

    alerts = list(source_alerts.get("alerts", []))
    heartbeat_emitted = False
    if not alerts:
        heartbeat_emitted = True
        alerts.append(
            {
                "severity": "info",
                "code": "heartbeat_no_active_alerts",
                "message": "No active alerts in source payload; heartbeat emitted.",
            }
        )

    bridge_payload = {
        "bridge_id": "week23-block2-alert-bridge-hardening-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "channel": args.bridge_channel,
        "mode": args.mode,
        "source_cycle_report": str(source_cycle_path),
        "source_cycle_decision": source_cycle_decision,
        "source_alerts_path": str(source_alerts_path),
        "source_alerts_decision": source_alerts_decision,
        "alerts": alerts,
    }

    health_url = _build_health_url(endpoint, args.healthcheck_path) if endpoint else ""
    pre_health = _http_get(health_url, args.request_timeout_seconds) if health_url else {
        "url": "",
        "status": None,
        "ok": False,
        "error": "endpoint_missing",
    }

    dispatch_result = (
        _dispatch_with_retry(
            endpoint=endpoint,
            payload=bridge_payload,
            retry_attempts=int(args.retry_attempts),
            backoff_seconds=float(args.backoff_seconds),
            backoff_multiplier=float(args.backoff_multiplier),
            timeout_seconds=float(args.request_timeout_seconds),
        )
        if endpoint
        else {
            "mode": "live",
            "sent": False,
            "http_status": None,
            "error": "endpoint_missing",
            "attempts": [],
            "attempts_executed": 0,
            "retries_used": False,
        }
    )

    post_health = _http_get(health_url, args.request_timeout_seconds) if health_url else {
        "url": "",
        "status": None,
        "ok": False,
        "error": "endpoint_missing",
    }

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
            "health_url": health_url,
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
            "delivery_health_pre": pre_health,
            "delivery_health_post": post_health,
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

    debt_payload = _operational_debt_payload(
        dispatch_result=dispatch_result,
        pre_health=pre_health,
        post_health=post_health,
        retries_used=bool(dispatch_result.get("retries_used", False)),
    )
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
    checks["endpoint_present_for_live"] = {
        "observed": bool(endpoint),
        "required": True,
        "pass": bool(endpoint),
    }
    checks["delivery_healthcheck_pre_ok"] = {
        "observed": pre_health,
        "required": "http_2xx",
        "pass": bool(pre_health.get("ok")),
    }
    checks["delivery_healthcheck_post_ok"] = {
        "observed": post_health,
        "required": "http_2xx",
        "pass": bool(post_health.get("ok")),
    }
    checks["retry_configured"] = {
        "observed": int(args.retry_attempts),
        "required_min": 2,
        "pass": int(args.retry_attempts) >= 2,
    }
    checks["retry_path_exercised"] = {
        "observed": bool(dispatch_result.get("retries_used", False)),
        "required": True,
        "pass": bool(dispatch_result.get("retries_used", False)),
    }
    checks["dispatch_live_success"] = {
        "observed": dispatch_result,
        "required": "sent_true_http_2xx",
        "pass": bool(dispatch_result.get("sent")),
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
        "Alert bridge hardening validated retry/backoff and delivery health checks with canonical gates green."
        if decision == "promote"
        else "Hardening block found unresolved dispatch reliability, health-check, retry, or validation failures."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow_path": str(workflow_path),
            "scheduler_spec_path": str(scheduler_spec_path),
            "source_cycle_report_path": str(source_cycle_path),
            "source_alerts_path": str(source_alerts_path),
            "mode": args.mode,
            "endpoint": endpoint,
            "health_url": health_url,
            "retry_attempts": int(args.retry_attempts),
            "backoff_seconds": float(args.backoff_seconds),
            "backoff_multiplier": float(args.backoff_multiplier),
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
            "bridged_alerts_count": len(alerts),
            "heartbeat_emitted": heartbeat_emitted,
            "dispatch_sent": bool(dispatch_result.get("sent")),
            "dispatch_http_status": dispatch_result.get("http_status"),
            "attempts_executed": int(dispatch_result.get("attempts_executed", 0)),
            "retries_used": bool(dispatch_result.get("retries_used", False)),
            "pre_health_ok": bool(pre_health.get("ok")),
            "post_health_ok": bool(post_health.get("ok")),
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

    print(f"Week23 block2 JSON: {report_json}")
    print(f"Week23 block2 MD:   {report_md}")
    print(f"Bridge payload:     {bridge_payload_json}")
    print(f"Dispatch record:    {dispatch_json}")
    print(f"Scheduler health:   {scheduler_health_json}")
    print(f"Rollback record:    {rollback_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
