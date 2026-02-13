#!/usr/bin/env python3
"""Week 14 Block 3: monthly audit simulation and live debt matrix refresh."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _extract_decision(payload: dict[str, Any]) -> str:
    for candidate in (
        payload.get("evaluation", {}).get("decision"),
        payload.get("block_decision"),
        payload.get("decision"),
        payload.get("result"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def _refresh_debts(
    *,
    base_debts: list[dict[str, Any]],
    block2_promote: bool,
    git_push_verified: bool,
    audit_dry_run_completed: bool,
    timestamp_utc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refreshed: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    for debt in base_debts:
        updated = dict(debt)
        debt_id = str(updated.get("debt_id", "unknown"))
        previous_status = str(updated.get("status", "open"))
        resolved = False
        resolution_note = ""

        if debt_id == "ops_push_authentication_pending" and git_push_verified:
            resolved = True
            resolution_note = "SSH authentication and branch push verified."
        elif debt_id == "policy_v2_extended_horizon_confirmation" and block2_promote:
            resolved = True
            resolution_note = "Week14 Block2 extended-horizon replay and split closed in promote."
        elif debt_id == "monthly_audit_first_dry_run" and audit_dry_run_completed:
            resolved = True
            resolution_note = "First monthly audit simulation executed successfully in Week14 Block3."

        if resolved:
            updated["status"] = "closed"
            updated["resolution_window"] = "week14_block3"
            updated["resolved_at_utc"] = timestamp_utc
            updated["resolution_notes"] = resolution_note
        else:
            updated["status"] = previous_status

        transitions.append(
            {
                "debt_id": debt_id,
                "from_status": previous_status,
                "to_status": updated.get("status"),
            }
        )
        refreshed.append(updated)

    return refreshed, transitions


def _summary(debts: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(debts),
        "open": sum(1 for debt in debts if debt.get("status") == "open"),
        "closed": sum(1 for debt in debts if debt.get("status") == "closed"),
        "high_or_critical_open": sum(
            1
            for debt in debts
            if debt.get("status") == "open" and debt.get("severity") in {"high", "critical"}
        ),
    }


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 14 Block 3 - Monthly Audit Simulation")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Weekly cadence: `{payload['metadata']['cadence_path']}`")
    lines.append(f"- Monthly audit template: `{payload['metadata']['monthly_audit_path']}`")
    lines.append(f"- Base debt matrix: `{payload['metadata']['base_debt_matrix_path']}`")
    lines.append(f"- Updated debt matrix: `{payload['artifacts']['updated_debt_matrix_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for check_name, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {check_name} | {check['pass']} |")
    lines.append("")
    lines.append("## Debt Transitions")
    lines.append("")
    lines.append("| Debt ID | From | To |")
    lines.append("| --- | --- | --- |")
    for transition in payload["debt_transitions"]:
        lines.append(
            f"| {transition['debt_id']} | {transition['from_status']} | {transition['to_status']} |"
        )
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
        description="Run Week14 Block3 monthly audit simulation and debt refresh."
    )
    parser.add_argument(
        "--cadence-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_WEEKLY_CADENCE.json",
    )
    parser.add_argument(
        "--monthly-audit-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_MONTHLY_AUDIT_WINDOW.md",
    )
    parser.add_argument(
        "--debt-matrix-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_LIVE_DEBT_MATRIX.json",
    )
    parser.add_argument(
        "--block2-decision-path",
        default="research/breakthrough_lab/week14_block2_extended_horizon_decision.json",
    )
    parser.add_argument("--canonical-gate-path", required=True)
    parser.add_argument(
        "--updated-debt-matrix-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK3_LIVE_DEBT_MATRIX_V2.json",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week14_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week14_block3_monthly_audit_simulation")
    parser.add_argument(
        "--git-push-verified",
        action="store_true",
        help="Mark git push/auth debt as verified and closable.",
    )
    args = parser.parse_args()

    cadence_path = Path(args.cadence_path).resolve()
    monthly_audit_path = Path(args.monthly_audit_path).resolve()
    debt_matrix_path = Path(args.debt_matrix_path).resolve()
    block2_decision_path = Path(args.block2_decision_path).resolve()
    canonical_gate_path = Path(args.canonical_gate_path).resolve()
    updated_debt_matrix_path = Path(args.updated_debt_matrix_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    timestamp_utc = timestamp.isoformat()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    cadence_exists = cadence_path.exists()
    monthly_audit_exists = monthly_audit_path.exists()
    debt_exists = debt_matrix_path.exists()
    block2_exists = block2_decision_path.exists()
    gate_exists = canonical_gate_path.exists()

    if not all((cadence_exists, monthly_audit_exists, debt_exists, block2_exists, gate_exists)):
        missing = []
        if not cadence_exists:
            missing.append(str(cadence_path))
        if not monthly_audit_exists:
            missing.append(str(monthly_audit_path))
        if not debt_exists:
            missing.append(str(debt_matrix_path))
        if not block2_exists:
            missing.append(str(block2_decision_path))
        if not gate_exists:
            missing.append(str(canonical_gate_path))
        raise SystemExit(f"Missing required inputs: {missing}")

    base_debt = _read_json(debt_matrix_path)
    block2_decision_payload = _read_json(block2_decision_path)
    gate_payload = _read_json(canonical_gate_path)

    block2_decision = _extract_decision(block2_decision_payload)
    gate_decision = _extract_decision(gate_payload)
    block2_promote = block2_decision == "promote"
    gate_promote = gate_decision == "promote"

    refreshed_debts, transitions = _refresh_debts(
        base_debts=list(base_debt.get("debts", [])),
        block2_promote=block2_promote,
        git_push_verified=bool(args.git_push_verified),
        audit_dry_run_completed=True,
        timestamp_utc=timestamp_utc,
    )
    refreshed_summary = _summary(refreshed_debts)

    refreshed_matrix = {
        "matrix_id": "week14-block3-live-debt-matrix-v2-2026-02-09",
        "inherits_from": str(debt_matrix_path),
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "block2_decision": block2_decision,
            "canonical_gate_decision": gate_decision,
            "git_push_verified": bool(args.git_push_verified),
        },
        "debts": refreshed_debts,
        "summary": refreshed_summary,
    }
    _write_json(updated_debt_matrix_path, refreshed_matrix)

    checks: dict[str, dict[str, Any]] = {}
    checks["weekly_cadence_exists"] = {
        "observed": cadence_exists,
        "required": True,
        "pass": cadence_exists,
    }
    checks["monthly_audit_template_exists"] = {
        "observed": monthly_audit_exists,
        "required": True,
        "pass": monthly_audit_exists,
    }
    checks["block2_promote"] = {
        "observed": block2_decision,
        "required": "promote",
        "pass": block2_promote,
    }
    checks["canonical_gate_promote"] = {
        "observed": gate_decision,
        "required": "promote",
        "pass": gate_promote,
    }
    checks["all_known_debts_closed"] = {
        "observed_open": refreshed_summary["open"],
        "required_open": 0,
        "pass": refreshed_summary["open"] == 0,
    }
    checks["no_high_critical_open_debt"] = {
        "observed": refreshed_summary["high_or_critical_open"],
        "required_max": 0,
        "pass": refreshed_summary["high_or_critical_open"] == 0,
    }

    failed_checks = [name for name, chk in checks.items() if not bool(chk.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Monthly audit simulation completed with canonical gate green and debt matrix fully closed."
        if decision == "promote"
        else "Monthly audit simulation found unresolved checks; keep iterate and close remaining debt."
    )

    payload = {
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "cadence_path": str(cadence_path),
            "monthly_audit_path": str(monthly_audit_path),
            "base_debt_matrix_path": str(debt_matrix_path),
            "block2_decision_path": str(block2_decision_path),
            "canonical_gate_path": str(canonical_gate_path),
        },
        "artifacts": {
            "updated_debt_matrix_path": str(updated_debt_matrix_path),
        },
        "debt_transitions": transitions,
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    json_path = output_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, payload)
    md_path.write_text(_report_md(payload))

    print(f"Week14 block3 JSON: {json_path}")
    print(f"Week14 block3 MD:   {md_path}")
    print(f"Debt matrix JSON:   {updated_debt_matrix_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
