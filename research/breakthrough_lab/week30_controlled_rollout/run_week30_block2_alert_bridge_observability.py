#!/usr/bin/env python3
"""Week 30 Block 2: alert bridge observability (success ratio + latency + degradation alerts)."""

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
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            status = int(response.status)
            return {
                "url": url,
                "status": status,
                "ok": 200 <= status < 300,
                "elapsed_ms": elapsed_ms,
                "error": None,
            }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "url": url,
            "status": None,
            "ok": False,
            "elapsed_ms": None,
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
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            status = int(response.status)
            return {
                "sent": 200 <= status < 300,
                "http_status": status,
                "elapsed_ms": elapsed_ms,
                "error": None if 200 <= status < 300 else f"http_status_{status}",
            }
    except urllib.error.HTTPError as exc:  # pragma: no cover
        return {
            "sent": False,
            "http_status": int(exc.code),
            "elapsed_ms": None,
            "error": str(exc),
        }
    except urllib.error.URLError as exc:  # pragma: no cover
        return {
            "sent": False,
            "http_status": None,
            "elapsed_ms": None,
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
        outcome = _post_json(endpoint, payload, timeout_seconds)
        attempt_row = {
            "attempt": attempt,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sent": bool(outcome.get("sent")),
            "http_status": outcome.get("http_status"),
            "error": outcome.get("error"),
            "elapsed_ms": outcome.get("elapsed_ms"),
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

    success_latency_ms = None
    for row in reversed(attempts):
        if bool(row.get("sent")) and row.get("elapsed_ms") is not None:
            success_latency_ms = float(row["elapsed_ms"])
            break

    return {
        "sent": final_sent,
        "http_status": final_status,
        "error": final_error,
        "attempts": attempts,
        "attempts_executed": len(attempts),
        "retries_used": len(attempts) > 1,
        "success_latency_ms": success_latency_ms,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 30 Block 2 - Alert Bridge Observability")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Endpoint: `{report['metadata']['endpoint']}`")
    lines.append(f"- Health URL: `{report['metadata']['health_url']}`")
    lines.append(f"- Cycles: `{int(report['metadata']['cycles'])}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Observability Metrics")
    lines.append("")
    m = report["observability_metrics"]
    lines.append(f"- cycle_success_ratio: `{float(m['cycle_success_ratio']):.6f}`")
    lines.append(f"- attempt_success_ratio: `{float(m['attempt_success_ratio']):.6f}`")
    lines.append(f"- dispatch_success_latency_p95_ms: `{m['dispatch_success_latency_p95_ms']}`")
    lines.append(f"- dispatch_success_latency_max_ms: `{m['dispatch_success_latency_max_ms']}`")
    lines.append(f"- retries_rate: `{float(m['retries_rate']):.6f}`")
    lines.append("")
    lines.append("## Alerts")
    lines.append("")
    for alert in report["alerts"]:
        lines.append(f"- [{alert['severity']}] `{alert['code']}`: {alert['message']}")
    if not report["alerts"]:
        lines.append("- none")
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
        description="Run Week30 Block2 bridge observability with success ratio, latency, and degradation alerts.",
    )
    parser.add_argument("--mode", choices=("local", "ci"), default="local")
    parser.add_argument("--bridge-endpoint-env", default="WEEKLY_ALERT_WEBHOOK_URL")
    parser.add_argument("--bridge-endpoint", default="")
    parser.add_argument("--healthcheck-path", default="/health")
    parser.add_argument("--request-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=0.3)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--success-ratio-min", type=float, default=0.95)
    parser.add_argument("--latency-p95-max-ms", type=float, default=50.0)
    parser.add_argument("--latency-max-ms", type=float, default=120.0)
    parser.add_argument("--bridge-channel", default="chatops-webhook")
    parser.add_argument("--workflow-path", default=".github/workflows/week20-monthly-cycle.yml")
    parser.add_argument(
        "--scheduler-spec-path",
        default="research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_scheduler_spec_20260211_140738.json",
    )
    parser.add_argument(
        "--source-cycle-report-path",
        default="research/breakthrough_lab/week30_controlled_rollout/week30_block1_monthly_continuity_20260213_013243.json",
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
        default="research/breakthrough_lab/week30_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week30_block2_alert_bridge_observability")
    args = parser.parse_args()

    if args.retry_attempts < 1:
        raise SystemExit("--retry-attempts must be >= 1")
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")

    workflow_path = _resolve(args.workflow_path)
    scheduler_spec_path = _resolve(args.scheduler_spec_path)
    source_cycle_path = _resolve(args.source_cycle_report_path)
    source_alerts_path = _resolve(args.source_alerts_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = (args.bridge_endpoint or os.environ.get(args.bridge_endpoint_env, "")).strip()
    health_url = _build_health_url(endpoint, args.healthcheck_path) if endpoint else ""

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

    alerts_payload = list(source_alerts.get("alerts", []))
    heartbeat_emitted = False
    if not alerts_payload:
        heartbeat_emitted = True
        alerts_payload.append(
            {
                "severity": "info",
                "code": "heartbeat_no_active_alerts",
                "message": "No active alerts in source payload; heartbeat emitted.",
            }
        )

    cycle_records: list[dict[str, Any]] = []
    success_cycles = 0
    retries_used_cycles = 0
    attempt_total = 0
    attempt_success_total = 0
    success_latencies: list[float] = []
    max_success_latency: float | None = None

    for cycle_idx in range(1, int(args.cycles) + 1):
        payload = {
            "bridge_id": "week30-block2-alert-bridge-observability-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle_idx,
            "channel": args.bridge_channel,
            "mode": args.mode,
            "source_cycle_report": str(source_cycle_path),
            "source_cycle_decision": source_cycle_decision,
            "source_alerts_path": str(source_alerts_path),
            "source_alerts_decision": source_alerts_decision,
            "alerts": alerts_payload,
        }
        pre_health = _http_get(health_url, args.request_timeout_seconds) if health_url else {
            "url": "",
            "status": None,
            "ok": False,
            "elapsed_ms": None,
            "error": "endpoint_missing",
        }
        dispatch = (
            _dispatch_with_retry(
                endpoint=endpoint,
                payload=payload,
                retry_attempts=int(args.retry_attempts),
                backoff_seconds=float(args.backoff_seconds),
                backoff_multiplier=float(args.backoff_multiplier),
                timeout_seconds=float(args.request_timeout_seconds),
            )
            if endpoint
            else {
                "sent": False,
                "http_status": None,
                "error": "endpoint_missing",
                "attempts": [],
                "attempts_executed": 0,
                "retries_used": False,
                "success_latency_ms": None,
            }
        )
        post_health = _http_get(health_url, args.request_timeout_seconds) if health_url else {
            "url": "",
            "status": None,
            "ok": False,
            "elapsed_ms": None,
            "error": "endpoint_missing",
        }

        if bool(dispatch.get("sent")):
            success_cycles += 1
        if bool(dispatch.get("retries_used")):
            retries_used_cycles += 1
        attempt_total += int(dispatch.get("attempts_executed", 0))
        attempt_success_total += sum(1 for row in dispatch.get("attempts", []) if bool(row.get("sent")))
        if dispatch.get("success_latency_ms") is not None:
            lat = float(dispatch["success_latency_ms"])
            success_latencies.append(lat)
            max_success_latency = lat if max_success_latency is None else max(max_success_latency, lat)

        cycle_records.append(
            {
                "cycle": cycle_idx,
                "pre_health": pre_health,
                "dispatch": dispatch,
                "post_health": post_health,
            }
        )

    cycle_success_ratio = float(success_cycles / int(args.cycles))
    attempt_success_ratio = float(attempt_success_total / attempt_total) if attempt_total else 0.0
    retries_rate = float(retries_used_cycles / int(args.cycles))
    latency_p95 = _percentile(success_latencies, 0.95)

    observability_metrics = {
        "cycles_total": int(args.cycles),
        "cycles_success": int(success_cycles),
        "cycle_success_ratio": cycle_success_ratio,
        "attempts_total": int(attempt_total),
        "attempts_success": int(attempt_success_total),
        "attempt_success_ratio": attempt_success_ratio,
        "dispatch_success_latency_p95_ms": latency_p95,
        "dispatch_success_latency_max_ms": max_success_latency,
        "retries_rate": retries_rate,
    }

    alerts: list[dict[str, Any]] = []
    if cycle_success_ratio < float(args.success_ratio_min):
        alerts.append(
            {
                "severity": "high",
                "code": "delivery_success_ratio_below_floor",
                "message": f"Observed cycle success ratio {cycle_success_ratio:.6f} below floor {float(args.success_ratio_min):.6f}.",
            }
        )
    if latency_p95 is not None and float(latency_p95) > float(args.latency_p95_max_ms):
        alerts.append(
            {
                "severity": "medium",
                "code": "delivery_latency_p95_high",
                "message": f"Observed p95 latency {float(latency_p95):.3f}ms above threshold {float(args.latency_p95_max_ms):.3f}ms.",
            }
        )
    if max_success_latency is not None and float(max_success_latency) > float(args.latency_max_ms):
        alerts.append(
            {
                "severity": "medium",
                "code": "delivery_latency_max_high",
                "message": f"Observed max latency {float(max_success_latency):.3f}ms above threshold {float(args.latency_max_ms):.3f}ms.",
            }
        )
    health_ok_all = all(bool(c["pre_health"].get("ok")) and bool(c["post_health"].get("ok")) for c in cycle_records)
    if not health_ok_all:
        alerts.append(
            {
                "severity": "high",
                "code": "delivery_healthcheck_failed",
                "message": "Pre or post delivery health-check failed in at least one cycle.",
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bridge_payload_json = output_dir / f"{args.output_prefix}_bridge_payload_{stamp}.json"
    _write_json(
        bridge_payload_json,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "health_url": health_url,
            "cycles": cycle_records,
        },
    )

    dispatch_json = output_dir / f"{args.output_prefix}_dispatch_{stamp}.json"
    _write_json(
        dispatch_json,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint_env": args.bridge_endpoint_env,
            "endpoint_present": bool(endpoint),
            "endpoint": endpoint,
            "health_url": health_url,
            "observability_metrics": observability_metrics,
            "cycles": cycle_records,
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

    alerts_json = output_dir / f"{args.output_prefix}_alerts_{stamp}.json"
    _write_json(
        alerts_json,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "alerts": alerts,
            "metrics": observability_metrics,
        },
    )

    debt_payload = {
        "matrix_id": "week30-block2-alert-bridge-observability-debt-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": [
            {
                "debt_id": "week30_block2_delivery_success_ratio_monitoring",
                "area": "operations",
                "severity": "medium",
                "status": "closed" if cycle_success_ratio >= float(args.success_ratio_min) else "open",
                "owner": "ops-observability",
                "description": "Monitorear success ratio de delivery en bridge live.",
                "mitigation": "Alertar si ratio cae bajo threshold y activar rollback operativo.",
                "target_window": "week30_block2",
            },
            {
                "debt_id": "week30_block2_delivery_latency_monitoring",
                "area": "operations",
                "severity": "medium",
                "status": "closed"
                if (latency_p95 is not None and float(latency_p95) <= float(args.latency_p95_max_ms))
                else "open",
                "owner": "ops-observability",
                "description": "Monitorear latencia p95 de delivery para detectar degradación.",
                "mitigation": "Publicar métricas por ciclo y alertar si p95 excede policy.",
                "target_window": "week30_block2",
            },
            {
                "debt_id": "week30_block2_degradation_alerting",
                "area": "operations",
                "severity": "medium",
                "status": "closed" if not any(a["severity"] == "high" for a in alerts) else "open",
                "owner": "ops-observability",
                "description": "Emitir alertas de degradación para bridge live.",
                "mitigation": "Mantener catálogo de alertas y acciones operativas de contención.",
                "target_window": "week30_block2",
            },
        ],
    }
    debt_payload["summary"] = {
        "total": len(debt_payload["debts"]),
        "open": sum(1 for d in debt_payload["debts"] if d["status"] == "open"),
        "high_or_critical_open": sum(
            1
            for d in debt_payload["debts"]
            if d["status"] == "open" and d["severity"] in {"high", "critical"}
        ),
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
    checks["endpoint_present"] = {
        "observed": bool(endpoint),
        "required": True,
        "pass": bool(endpoint),
    }
    checks["cycle_success_ratio_threshold"] = {
        "observed": cycle_success_ratio,
        "required_min": float(args.success_ratio_min),
        "pass": cycle_success_ratio >= float(args.success_ratio_min),
    }
    checks["latency_p95_threshold"] = {
        "observed": latency_p95,
        "required_max": float(args.latency_p95_max_ms),
        "pass": latency_p95 is not None and float(latency_p95) <= float(args.latency_p95_max_ms),
    }
    checks["degradation_high_alerts_none"] = {
        "observed_high_alerts": [a for a in alerts if a.get("severity") == "high"],
        "required": "empty",
        "pass": not any(a.get("severity") == "high" for a in alerts),
    }
    checks["retry_path_exercised"] = {
        "observed": retries_used_cycles,
        "required_min": 1,
        "pass": retries_used_cycles >= 1,
    }
    checks["artifacts_written"] = {
        "observed": all(p.exists() for p in [bridge_payload_json, dispatch_json, scheduler_health_json, alerts_json, debt_json]),
        "required": True,
        "pass": all(p.exists() for p in [bridge_payload_json, dispatch_json, scheduler_health_json, alerts_json, debt_json]),
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
        "Bridge observability metrics stay healthy with degradation alerts under threshold and canonical gates green."
        if decision == "promote"
        else "Observability block found unresolved delivery reliability, latency, alerting, or validation issues."
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
            "cycles": int(args.cycles),
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
            "alerts_json": str(alerts_json),
            "operational_debt_json": str(debt_json),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "observability_metrics": observability_metrics,
        "alerts": alerts,
        "highlights": {
            "source_cycle_decision": source_cycle_decision,
            "source_alerts_decision": source_alerts_decision,
            "cycle_success_ratio": cycle_success_ratio,
            "dispatch_success_latency_p95_ms": latency_p95,
            "dispatch_success_latency_max_ms": max_success_latency,
            "retries_rate": retries_rate,
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

    print(f"Week30 block2 JSON: {report_json}")
    print(f"Week30 block2 MD:   {report_md}")
    print(f"Dispatch JSON:       {dispatch_json}")
    print(f"Alerts JSON:         {alerts_json}")
    print(f"Scheduler health:    {scheduler_health_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
