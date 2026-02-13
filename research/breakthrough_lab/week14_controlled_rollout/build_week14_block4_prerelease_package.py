#!/usr/bin/env python3
"""Week 14 Block 4: build RX590 pre-release package."""

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
    for key in ("block_decision", "decision", "result"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    eval_decision = payload.get("evaluation", {}).get("decision")
    if isinstance(eval_decision, str) and eval_decision:
        return eval_decision
    return "unknown"


def _render_runbook(
    *,
    policy_path: str,
    t5_policy_path: str,
    rollback_sla_path: str,
    weekly_cadence_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block4 RX590 Pre-Release Runbook")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Pre-release enablement workflow for controlled RX590 real-world pilots.")
    lines.append("")
    lines.append("## Active Inputs")
    lines.append("")
    lines.append(f"- Weekly policy: `{policy_path}`")
    lines.append(f"- T5 policy: `{t5_policy_path}`")
    lines.append(f"- Rollback SLA: `{rollback_sla_path}`")
    lines.append(f"- Weekly cadence baseline: `{weekly_cadence_path}`")
    lines.append("")
    lines.append("## Preconditions")
    lines.append("")
    lines.append("1. Driver smoke is healthy (`overall_status=good`).")
    lines.append("2. Week14 Block2 and Block3 are closed in `promote`.")
    lines.append("3. Canonical validation gate is `promote` before any scope increase.")
    lines.append("")
    lines.append("## Enablement Steps")
    lines.append("")
    lines.append("1. Baseline validation gate:")
    lines.append("   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`")
    lines.append("2. Verify runtime inventory:")
    lines.append("   - `./venv/bin/python scripts/verify_drivers.py --json`")
    lines.append("3. Run controlled dry-run scope:")
    lines.append("   - `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 3 --snapshot-interval-minutes 1 --sizes 1400 2048 --sessions 1 --iterations 4`")
    lines.append("4. If any hard gate fails, apply rollback:")
    lines.append("   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`")
    lines.append("5. Re-run canonical gate post-rollback and freeze promotion if not green.")
    lines.append("")
    lines.append("## Promotion Gate")
    lines.append("")
    lines.append("Promote only if all are true:")
    lines.append("- Dry-run decision is `go`.")
    lines.append("- Canonical gate post-run is `promote`.")
    lines.append("- `t5_disable_events_total == 0` and correctness bound remains within `1e-3`.")
    lines.append("- Rollback path remains executable under SLA contract.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_plugin_checklist() -> str:
    lines: list[str] = []
    lines.append("# Week14 Block4 Plugins / Base Projects Checklist")
    lines.append("")
    lines.append("## Extension Contracts")
    lines.append("")
    lines.append("- [ ] Plugin declares deterministic seed protocol for benchmarks.")
    lines.append("- [ ] Plugin exposes explicit fallback behavior (`promote|iterate|stop` compatible).")
    lines.append("- [ ] Plugin preserves correctness contract (`max_error <= 1e-3`).")
    lines.append("- [ ] Plugin emits machine-readable artifact (JSON + summary MD).")
    lines.append("- [ ] Plugin is compatible with canonical gate (`scripts/run_validation_suite.py`).")
    lines.append("")
    lines.append("## Runtime Compatibility")
    lines.append("")
    lines.append("- [ ] Works on RX590 Clover baseline.")
    lines.append("- [ ] Handles optional rusticl split mode without hard failure.")
    lines.append("- [ ] Honors rollback environment profile (`week9_block5_rusticl_rollback.sh`).")
    lines.append("")
    lines.append("## Base Project Integration")
    lines.append("")
    lines.append("- [ ] Inference project: integrates selector/guardrails without bypassing policy.")
    lines.append("- [ ] Benchmark project: logs gflops, p95, fallback, disable_events.")
    lines.append("- [ ] Operations project: supports weekly cadence + monthly audit path.")
    lines.append("- [ ] CI project: validates schema and canonical tier before merge.")
    lines.append("")
    lines.append("## Documentation Minimum")
    lines.append("")
    lines.append("- [ ] README includes activation/deactivation instructions.")
    lines.append("- [ ] Rollback instructions tested in dry-run mode.")
    lines.append("- [ ] Known limits and risk notes documented.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_manifest_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block4 RX590 Pre-Release Package")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Block2 decision: `{payload['metadata']['block2_decision']}`")
    lines.append(f"- Block3 decision: `{payload['metadata']['block3_decision']}`")
    lines.append(f"- Canonical gate: `{payload['metadata']['canonical_gate_decision']}`")
    lines.append("")
    lines.append("## Deliverables")
    lines.append("")
    lines.append(f"- Runbook: `{payload['artifacts']['runbook_path']}`")
    lines.append(f"- Plugins checklist: `{payload['artifacts']['plugins_checklist_path']}`")
    lines.append(f"- Manifest: `{payload['artifacts']['manifest_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week14 Block4 pre-release package.")
    parser.add_argument(
        "--block2-decision-path",
        default="research/breakthrough_lab/week14_block2_extended_horizon_decision.json",
    )
    parser.add_argument(
        "--block3-decision-path",
        default="research/breakthrough_lab/week14_block3_monthly_audit_simulation_decision.json",
    )
    parser.add_argument("--canonical-gate-path", required=True)
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json",
    )
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json",
    )
    parser.add_argument(
        "--rollback-sla-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md",
    )
    parser.add_argument(
        "--weekly-cadence-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_WEEKLY_CADENCE.json",
    )
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week14_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week14_block4_prerelease_package")
    args = parser.parse_args()

    block2_path = Path(args.block2_decision_path).resolve()
    block3_path = Path(args.block3_decision_path).resolve()
    gate_path = Path(args.canonical_gate_path).resolve()
    policy_path = Path(args.policy_path).resolve()
    t5_policy_path = Path(args.t5_policy_path).resolve()
    rollback_sla_path = Path(args.rollback_sla_path).resolve()
    cadence_path = Path(args.weekly_cadence_path).resolve()
    preprod_dir = Path(args.preprod_signoff_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    preprod_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    block2_decision = _extract_decision(_read_json(block2_path))
    block3_decision = _extract_decision(_read_json(block3_path))
    canonical_decision = _extract_decision(_read_json(gate_path))

    runbook_path = preprod_dir / "WEEK14_BLOCK4_RX590_PRERELEASE_RUNBOOK.md"
    checklist_path = preprod_dir / "WEEK14_BLOCK4_PLUGIN_PROJECT_BASE_CHECKLIST.md"
    manifest_path = preprod_dir / "WEEK14_BLOCK4_RX590_PRERELEASE_MANIFEST.json"

    runbook_path.write_text(
        _render_runbook(
            policy_path=str(policy_path),
            t5_policy_path=str(t5_policy_path),
            rollback_sla_path=str(rollback_sla_path),
            weekly_cadence_path=str(cadence_path),
        )
    )
    checklist_path.write_text(_render_plugin_checklist())

    timestamp = datetime.now(timezone.utc)
    timestamp_utc = timestamp.isoformat()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    manifest_payload = {
        "manifest_id": "week14-block4-rx590-prerelease-manifest-v1-2026-02-09",
        "timestamp_utc": timestamp_utc,
        "block2_decision": block2_decision,
        "block3_decision": block3_decision,
        "canonical_gate_decision": canonical_decision,
        "runbook_path": str(runbook_path),
        "plugins_checklist_path": str(checklist_path),
        "policy_path": str(policy_path),
        "t5_policy_path": str(t5_policy_path),
        "rollback_sla_path": str(rollback_sla_path),
    }
    _write_json(manifest_path, manifest_payload)

    checks: dict[str, dict[str, Any]] = {}
    checks["block2_promote"] = {
        "observed": block2_decision,
        "required": "promote",
        "pass": block2_decision == "promote",
    }
    checks["block3_promote"] = {
        "observed": block3_decision,
        "required": "promote",
        "pass": block3_decision == "promote",
    }
    checks["canonical_gate_promote"] = {
        "observed": canonical_decision,
        "required": "promote",
        "pass": canonical_decision == "promote",
    }
    checks["runbook_written"] = {
        "observed": runbook_path.exists(),
        "required": True,
        "pass": runbook_path.exists(),
    }
    checks["checklist_written"] = {
        "observed": checklist_path.exists(),
        "required": True,
        "pass": checklist_path.exists(),
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "RX590 pre-release package is complete and all upstream gates are green."
        if decision == "promote"
        else "Pre-release package cannot be promoted until upstream gates/checks are green."
    )

    payload = {
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "block2_decision": block2_decision,
            "block3_decision": block3_decision,
            "canonical_gate_decision": canonical_decision,
            "block2_decision_path": str(block2_path),
            "block3_decision_path": str(block3_path),
            "canonical_gate_path": str(gate_path),
        },
        "artifacts": {
            "runbook_path": str(runbook_path),
            "plugins_checklist_path": str(checklist_path),
            "manifest_path": str(manifest_path),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, payload)
    md_path.write_text(_render_manifest_md(payload))

    print(f"Week14 block4 JSON: {json_path}")
    print(f"Week14 block4 MD:   {md_path}")
    print(f"Runbook:            {runbook_path}")
    print(f"Checklist:          {checklist_path}")
    print(f"Manifest:           {manifest_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
