#!/usr/bin/env python3
"""Week 13 Block 5: build operational continuity package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _cadence_json(
    *,
    policy_path: str,
    validation_cmd: str,
    weekly_replay_cmd: str,
    split_cmd: str,
    continuity_report_cmd: str,
) -> dict[str, Any]:
    return {
        "cadence_id": "week13-block5-weekly-cadence-v1-2026-02-09",
        "timezone": "America/New_York",
        "policy_path": policy_path,
        "windows": {
            "weekly_window": {
                "day_of_week": "Monday",
                "start_local": "08:00",
                "duration_minutes": 120,
                "steps": [
                    {"id": "gate_pre", "command": validation_cmd},
                    {"id": "weekly_replay", "command": weekly_replay_cmd},
                    {"id": "platform_split", "command": split_cmd},
                    {"id": "continuity_consolidation", "command": continuity_report_cmd},
                    {"id": "gate_post", "command": validation_cmd},
                ],
            },
            "daily_smoke": {
                "day_of_week": "Mon-Fri",
                "start_local": "07:30",
                "duration_minutes": 20,
                "steps": [{"id": "driver_smoke", "command": validation_cmd}],
            },
        },
        "sla": {
            "sev1_response_minutes": 10,
            "sev2_response_minutes": 30,
            "sev3_response_minutes": 120,
            "rollback_sla_reference": "research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md",
        },
    }


def _monthly_audit_md(*, policy_path: str, cadence_path: str) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 5 - Monthly Audit Window")
    lines.append("")
    lines.append("- Audit mode: monthly continuity audit")
    lines.append("- Frequency: first business day of each month")
    lines.append("- Duration target: 180 minutes")
    lines.append(f"- Active policy: `{policy_path}`")
    lines.append(f"- Weekly cadence baseline: `{cadence_path}`")
    lines.append("")
    lines.append("## Mandatory Audit Checklist")
    lines.append("")
    lines.append("- Validate canonical gate before audit execution.")
    lines.append("- Replay weekly profile on latest policy version.")
    lines.append("- Execute Clover/rusticl split and verify ratio floor.")
    lines.append("- Compare monthly drift against previous monthly window.")
    lines.append("- Confirm rollback drill command remains executable.")
    lines.append("- Capture go/no-go result and open debts in live matrix.")
    lines.append("")
    lines.append("## Escalation Rules")
    lines.append("")
    lines.append("- Any correctness violation: immediate rollback and SEV1 escalation.")
    lines.append("- Any `t5_disable_events_total > 0`: hold promotion and SEV2 escalation.")
    lines.append("- Any split ratio below floor: hold promotion and SEV2 escalation.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _live_debt_matrix(
    *,
    block4_decision: str,
    gate_decision: str,
) -> dict[str, Any]:
    debts: list[dict[str, Any]] = []
    debts.append(
        {
            "debt_id": "ops_push_authentication_pending",
            "area": "release_operations",
            "severity": "medium",
            "status": "open",
            "owner": "ops",
            "description": "Remote push still requires local GitHub auth configuration.",
            "mitigation": "Configure PAT/SSH and validate branch push path.",
            "due_window": "week14_block1",
        }
    )
    debts.append(
        {
            "debt_id": "policy_v2_extended_horizon_confirmation",
            "area": "performance_policy",
            "severity": "medium",
            "status": "open",
            "owner": "performance",
            "description": "Policy v2 was validated on current horizon; extended-horizon replay still pending.",
            "mitigation": "Execute week14_block1 replay/split cycle with policy v2 and compare against block13 baselines.",
            "due_window": "week14_block1",
        }
    )
    debts.append(
        {
            "debt_id": "monthly_audit_first_dry_run",
            "area": "governance",
            "severity": "low",
            "status": "open",
            "owner": "qa_ops",
            "description": "Monthly audit window template created, first dry run pending execution.",
            "mitigation": "Run first monthly audit simulation and update checklist with findings.",
            "due_window": "week14_block2",
        }
    )

    return {
        "matrix_id": "week13-block5-live-debt-matrix-v1-2026-02-09",
        "inputs": {
            "block4_package_decision": block4_decision,
            "canonical_gate_decision": gate_decision,
        },
        "debts": debts,
        "summary": {
            "total": len(debts),
            "open": sum(1 for d in debts if d["status"] == "open"),
            "closed": sum(1 for d in debts if d["status"] == "closed"),
            "high_or_critical_open": sum(
                1 for d in debts if d["status"] == "open" and d["severity"] in {"high", "critical"}
            ),
        },
    }


def _package_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 13 Block 5 - Operational Continuity Package")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Block4 dashboard: `{payload['metadata']['block4_dashboard_path']}`")
    lines.append(f"- Block4 drift v2: `{payload['metadata']['block4_drift_path']}`")
    lines.append(f"- Canonical gate: `{payload['metadata']['canonical_gate_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, chk in payload["evaluation"]["checks"].items():
        lines.append(f"| {key} | {chk['pass']} |")
    lines.append("")
    lines.append("## Deliverables")
    lines.append("")
    lines.append(f"- Weekly cadence: `{payload['artifacts']['weekly_cadence_path']}`")
    lines.append(f"- Monthly audit window: `{payload['artifacts']['monthly_audit_path']}`")
    lines.append(f"- Live debt matrix: `{payload['artifacts']['live_debt_matrix_path']}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week13 Block5 continuity package.")
    parser.add_argument("--block4-dashboard", required=True)
    parser.add_argument("--block4-drift", required=True)
    parser.add_argument("--policy-v2-path", required=True)
    parser.add_argument("--canonical-gate", required=True)
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week13_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week13_block5_operational_continuity")
    args = parser.parse_args()

    dashboard_path = Path(args.block4_dashboard).resolve()
    drift_path = Path(args.block4_drift).resolve()
    policy_v2_path = Path(args.policy_v2_path).resolve()
    gate_path = Path(args.canonical_gate).resolve()
    preprod_dir = Path(args.preprod_signoff_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    preprod_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dashboard = _read_json(dashboard_path)
    drift = _read_json(drift_path)
    gate = _read_json(gate_path)

    gate_decision = str(gate.get("evaluation", {}).get("decision", "unknown"))
    block4_decision = str(dashboard.get("package_decision", "unknown"))
    drift_decision = str(drift.get("decision", "unknown"))

    validation_cmd = (
        "./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke "
        "--report-dir research/breakthrough_lab/week8_validation_discipline"
    )
    weekly_replay_cmd = (
        "./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/"
        "run_week12_weekly_replay_automation.py --mode local --policy-path "
        "research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json"
    )
    split_cmd = (
        "./venv/bin/python research/breakthrough_lab/platform_compatibility/"
        "run_week9_block4_stress_split.py --sizes 1400 2048 3072 --kernels "
        "auto_t3_controlled auto_t5_guarded"
    )
    continuity_cmd = (
        "./venv/bin/python research/breakthrough_lab/week13_controlled_rollout/"
        "build_week13_block5_continuity_package.py"
    )

    cadence_payload = _cadence_json(
        policy_path=str(policy_v2_path),
        validation_cmd=validation_cmd,
        weekly_replay_cmd=weekly_replay_cmd,
        split_cmd=split_cmd,
        continuity_report_cmd=continuity_cmd,
    )
    cadence_path = preprod_dir / "WEEK13_BLOCK5_WEEKLY_CADENCE.json"
    _write_json(cadence_path, cadence_payload)

    audit_path = preprod_dir / "WEEK13_BLOCK5_MONTHLY_AUDIT_WINDOW.md"
    audit_path.write_text(
        _monthly_audit_md(policy_path=str(policy_v2_path), cadence_path=str(cadence_path))
    )

    debt_payload = _live_debt_matrix(block4_decision=block4_decision, gate_decision=gate_decision)
    debt_path = preprod_dir / "WEEK13_BLOCK5_LIVE_DEBT_MATRIX.json"
    _write_json(debt_path, debt_payload)

    checks: dict[str, dict[str, Any]] = {}
    checks["block4_package_promote"] = {
        "observed": block4_decision,
        "required": "promote",
        "pass": block4_decision == "promote",
    }
    checks["drift_v2_promote"] = {
        "observed": drift_decision,
        "required": "promote",
        "pass": drift_decision == "promote",
    }
    checks["canonical_gate_promote"] = {
        "observed": gate_decision,
        "required": "promote",
        "pass": gate_decision == "promote",
    }
    checks["no_high_critical_open_debt"] = {
        "observed": int(debt_payload["summary"]["high_or_critical_open"]),
        "required_max": 0,
        "pass": int(debt_payload["summary"]["high_or_critical_open"]) <= 0,
    }

    failed_checks = [name for name, chk in checks.items() if not bool(chk.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Continuity package is operationally complete with promote status on dashboard/drift/gate."
        if decision == "promote"
        else "One or more continuity checks failed; keep iterate and resolve before promotion."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "block4_dashboard_path": str(dashboard_path),
            "block4_drift_path": str(drift_path),
            "canonical_gate_path": str(gate_path),
            "policy_v2_path": str(policy_v2_path),
        },
        "artifacts": {
            "weekly_cadence_path": str(cadence_path),
            "monthly_audit_path": str(audit_path),
            "live_debt_matrix_path": str(debt_path),
        },
        "evaluation": {
            "checks": checks,
            "decision": decision,
            "failed_checks": failed_checks,
            "rationale": rationale,
        },
    }

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, payload)
    md_path.write_text(_package_md(payload))

    print(f"Week13 block5 continuity JSON: {json_path}")
    print(f"Week13 block5 continuity MD:   {md_path}")
    print(f"Weekly cadence JSON:           {cadence_path}")
    print(f"Monthly audit MD:             {audit_path}")
    print(f"Live debt matrix JSON:        {debt_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
