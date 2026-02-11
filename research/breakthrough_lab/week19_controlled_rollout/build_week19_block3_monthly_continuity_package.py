#!/usr/bin/env python3
"""Week 19 Block 3: monthly operational continuity package."""

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


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 19 Block 3 - Monthly Continuity Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append(f"- Policy: `{payload['metadata']['policy_path']}`")
    lines.append("")
    lines.append("## Block Inputs")
    lines.append("")
    lines.append("| Input | Decision |")
    lines.append("| --- | --- |")
    lines.append(f"| week19_block1 | {payload['inputs']['block1_decision']} |")
    lines.append(f"| week19_block2 | {payload['inputs']['block2_decision']} |")
    lines.append("")
    lines.append("## Operational Metrics")
    lines.append("")
    lines.append(
        f"- rusticl/clover ratio min (Block1): `{float(payload['metrics']['block1_split_ratio_min']):.6f}`"
    )
    lines.append(
        f"- split T5 overhead max (Block1): `{float(payload['metrics']['block1_t5_overhead_max']):.6f}`"
    )
    lines.append(
        f"- split T5 disable total (Block1): `{int(payload['metrics']['block1_t5_disable_total'])}`"
    )
    lines.append(
        f"- global abs throughput drift max (Block2): `{float(payload['metrics']['block2_global_abs_drift_max']):.6f}`"
    )
    lines.append(
        f"- global p95 drift max (Block2): `{float(payload['metrics']['block2_global_p95_drift_max']):.6f}`"
    )
    lines.append(f"- recalibration action (Block2): `{payload['metrics']['block2_recalibration_action']}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _runbook_md(policy_path: str, dashboard_path: str) -> str:
    lines: list[str] = []
    lines.append("# Week 19 Block 3 - Monthly Continuity Runbook")
    lines.append("")
    lines.append("- Cadence: mensual (primer dia habil de cada mes).")
    lines.append("- Scope: replay semanal + split Clover/rusticl + consolidacion de drift.")
    lines.append(f"- Active policy: `{policy_path}`")
    lines.append(f"- Dashboard reference: `{dashboard_path}`")
    lines.append("")
    lines.append("## Mandatory Flow")
    lines.append("")
    lines.append("1. Run canonical gate pre (`run_validation_suite.py --tier canonical --driver-smoke`).")
    lines.append("2. Run weekly replay automation on stable baseline.")
    lines.append("3. Run Clover/rusticl split canary and policy evaluation.")
    lines.append("4. Run biweekly drift review/recalibration package.")
    lines.append("5. Run canonical gate post and close acta + decision.")
    lines.append("")
    lines.append("## Rollback Rules")
    lines.append("")
    lines.append("- If `t5_disable_total > 0`, stop promotion and rollback to last known good policy.")
    lines.append("- If rusticl/clover ratio falls below policy floor, stop expansion.")
    lines.append("- If canonical gate is not promote, block closure.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _checklist_md() -> str:
    lines: list[str] = []
    lines.append("# Week 19 Block 3 - Monthly Continuity Checklist")
    lines.append("")
    lines.append("- [ ] Week 19 Block 1 closed in promote.")
    lines.append("- [ ] Week 19 Block 2 closed in promote.")
    lines.append("- [ ] Recalibrated policy exists and is traceable.")
    lines.append("- [ ] Canonical gate pre is promote.")
    lines.append("- [ ] Canonical gate post is promote.")
    lines.append("- [ ] Dashboard updated with monthly continuity metrics.")
    lines.append("- [ ] Debt matrix reviewed (no high/critical open).")
    lines.append("")
    return "\n".join(lines) + "\n"


def _debt_matrix() -> dict[str, Any]:
    debts = [
        {
            "debt_id": "monthly_continuity_ci_schedule_hardening",
            "area": "automation",
            "severity": "medium",
            "status": "open",
            "owner": "ops-automation",
            "description": "Consolidate monthly continuity runner as scheduled CI workflow.",
            "mitigation": "Add scheduled workflow with artifact retention and failure escalation.",
            "target_window": "week20_block1"
        },
        {
            "debt_id": "external_alerting_bridge",
            "area": "operations",
            "severity": "medium",
            "status": "open",
            "owner": "ops-observability",
            "description": "Connect continuity failure events to external alerting channel.",
            "mitigation": "Bridge canonical/split failures to alert endpoint (mail/webhook/chatops).",
            "target_window": "week20_block2"
        }
    ]
    return {
        "matrix_id": "week19-block3-monthly-continuity-debt-v1-2026-02-11",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": debts,
        "summary": {
            "total": len(debts),
            "open": sum(1 for d in debts if d["status"] == "open"),
            "high_or_critical_open": sum(
                1 for d in debts if d["status"] == "open" and d["severity"] in {"high", "critical"}
            ),
        },
    }


def _report_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 19 Block 3 - Monthly Continuity Package")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, chk in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {chk['pass']} |")
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
    parser = argparse.ArgumentParser(description="Build Week19 Block3 monthly continuity package.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--block1-report-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_20260211_015638.json",
    )
    parser.add_argument(
        "--block1-decision-path",
        default="research/breakthrough_lab/week19_block1_weekly_split_maintenance_decision.json",
    )
    parser.add_argument(
        "--block2-report-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block2_biweekly_drift_recalibration_package_20260211_020443.json",
    )
    parser.add_argument(
        "--block2-decision-path",
        default="research/breakthrough_lab/week19_block2_biweekly_drift_recalibration_decision.json",
    )
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json",
    )
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
        default="research/breakthrough_lab/week19_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week19_block3_monthly_continuity_package")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    block1_report_path = _resolve(args.block1_report_path)
    block1_decision_path = _resolve(args.block1_decision_path)
    block2_report_path = _resolve(args.block2_report_path)
    block2_decision_path = _resolve(args.block2_decision_path)
    policy_path = _resolve(args.policy_path)
    preprod_dir = _resolve(args.preprod_signoff_dir)
    output_dir = _resolve(args.output_dir)
    preprod_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest = _read_json(stable_manifest_path) if stable_manifest_path.exists() else {}
    block1_report = _read_json(block1_report_path) if block1_report_path.exists() else {}
    block1_decision = _read_json(block1_decision_path) if block1_decision_path.exists() else {}
    block2_report = _read_json(block2_report_path) if block2_report_path.exists() else {}
    block2_decision = _read_json(block2_decision_path) if block2_decision_path.exists() else {}

    stable_tag = str(stable_manifest.get("stable_tag", "unknown"))
    block1_decision_value = str(block1_decision.get("block_decision", "unknown"))
    block2_decision_value = str(block2_decision.get("block_decision", "unknown"))
    block2_package_decision = str(block2_report.get("evaluation", {}).get("decision", "unknown"))

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
    if pre_gate_json:
        pre_gate_decision = str(
            _read_json(_resolve(pre_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    dashboard_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "policy_path": str(policy_path),
            "block1_report_path": str(block1_report_path),
            "block2_report_path": str(block2_report_path),
        },
        "inputs": {
            "block1_decision": block1_decision_value,
            "block2_decision": block2_decision_value,
            "block2_package_decision": block2_package_decision,
        },
        "metrics": {
            "block1_split_ratio_min": float(
                block1_report.get("highlights", {}).get("split_ratio_min", 0.0)
            ),
            "block1_t5_overhead_max": float(
                block1_report.get("highlights", {}).get("split_t5_overhead_max", 999.0)
            ),
            "block1_t5_disable_total": int(
                block1_report.get("highlights", {}).get("split_t5_disable_total", 999)
            ),
            "block2_global_abs_drift_max": float(
                block2_report.get("highlights", {}).get("global_max_abs_throughput_drift_percent", 999.0)
            ),
            "block2_global_p95_drift_max": float(
                block2_report.get("highlights", {}).get("global_max_p95_drift_percent", 999.0)
            ),
            "block2_recalibration_action": str(
                block2_report.get("highlights", {}).get("recalibration_action", "unknown")
            ),
        },
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard_json = output_dir / f"week19_block3_monthly_continuity_dashboard_{stamp}.json"
    dashboard_md = output_dir / f"week19_block3_monthly_continuity_dashboard_{stamp}.md"
    _write_json(dashboard_json, dashboard_payload)
    dashboard_md.write_text(_dashboard_md(dashboard_payload))

    runbook_path = preprod_dir / "WEEK19_BLOCK3_MONTHLY_CONTINUITY_RUNBOOK.md"
    checklist_path = preprod_dir / "WEEK19_BLOCK3_MONTHLY_CONTINUITY_CHECKLIST.md"
    debt_path = preprod_dir / "WEEK19_BLOCK3_MONTHLY_LIVE_DEBT_MATRIX.json"
    manifest_path = preprod_dir / "WEEK19_BLOCK3_MONTHLY_CONTINUITY_MANIFEST.json"

    runbook_path.write_text(_runbook_md(str(policy_path), str(dashboard_json)))
    checklist_path.write_text(_checklist_md())
    debt_payload = _debt_matrix()
    _write_json(debt_path, debt_payload)

    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stable_tag": stable_tag,
        "dashboard_json": str(dashboard_json),
        "dashboard_md": str(dashboard_md),
        "runbook_path": str(runbook_path),
        "checklist_path": str(checklist_path),
        "debt_matrix_path": str(debt_path),
        "policy_path": str(policy_path),
        "status": "monthly_continuity_candidate",
    }
    _write_json(manifest_path, manifest_payload)

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
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
    checks["week19_block1_promote"] = {
        "observed": block1_decision_value,
        "required": "promote",
        "pass": block1_decision_value == "promote",
    }
    checks["week19_block2_promote"] = {
        "observed": block2_decision_value,
        "required": "promote",
        "pass": block2_decision_value == "promote",
    }
    checks["week19_block2_package_promote"] = {
        "observed": block2_package_decision,
        "required": "promote",
        "pass": block2_package_decision == "promote",
    }
    checks["recalibrated_policy_exists"] = {
        "observed": policy_path.exists(),
        "required": True,
        "pass": policy_path.exists(),
    }
    checks["dashboard_written"] = {
        "observed": dashboard_json.exists() and dashboard_md.exists(),
        "required": True,
        "pass": dashboard_json.exists() and dashboard_md.exists(),
    }
    checks["continuity_docs_written"] = {
        "observed": runbook_path.exists() and checklist_path.exists() and debt_path.exists() and manifest_path.exists(),
        "required": True,
        "pass": runbook_path.exists() and checklist_path.exists() and debt_path.exists() and manifest_path.exists(),
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
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }

    failed_checks = [key for key, chk in checks.items() if not bool(chk.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Monthly continuity package is complete with stable promote chain and operational artifacts published."
        if decision == "promote"
        else "Monthly continuity package has unresolved checks and cannot be promoted."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "policy_path": str(policy_path),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "artifacts": {
            "dashboard_json": str(dashboard_json),
            "dashboard_md": str(dashboard_md),
            "runbook_path": str(runbook_path),
            "checklist_path": str(checklist_path),
            "debt_matrix_path": str(debt_path),
            "manifest_path": str(manifest_path),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
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

    print(f"Week19 block3 JSON: {report_json}")
    print(f"Week19 block3 MD:   {report_md}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
