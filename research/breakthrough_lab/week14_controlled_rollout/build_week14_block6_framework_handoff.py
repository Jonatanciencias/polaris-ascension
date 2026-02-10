#!/usr/bin/env python3
"""Week 14 Block 6: framework handoff consolidation for extensions/plugins."""

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
    for key in ("decision", "block_decision", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    eval_decision = payload.get("evaluation", {}).get("decision")
    if isinstance(eval_decision, str) and eval_decision:
        return eval_decision
    return "unknown"


def _render_handoff_md(
    *,
    policy_path: str,
    runbook_path: str,
    plugin_checklist_path: str,
    block5_checklist_path: str,
    rollback_sla_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block6 Framework Handoff")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Consolidate a stable framework baseline for extension/plugin teams and dependent projects.")
    lines.append("")
    lines.append("## Mandatory Operational Inputs")
    lines.append("")
    lines.append(f"- Weekly policy: `{policy_path}`")
    lines.append(f"- RX590 pre-release runbook: `{runbook_path}`")
    lines.append(f"- Plugin/project checklist: `{plugin_checklist_path}`")
    lines.append(f"- Block5 go/no-go checklist: `{block5_checklist_path}`")
    lines.append(f"- Rollback SLA: `{rollback_sla_path}`")
    lines.append("")
    lines.append("## Handoff Rules")
    lines.append("")
    lines.append("1. No plugin promotion without canonical gate in `promote`.")
    lines.append("2. No extension can bypass policy guardrails (`T3/T4/T5`).")
    lines.append("3. Every extension must emit JSON + MD artifacts and formal decision state.")
    lines.append("4. Rollback path must remain executable before any scope expansion.")
    lines.append("5. RX590 controlled profile remains baseline reference for compatibility.")
    lines.append("")
    lines.append("## Recommended Starter Sequence")
    lines.append("")
    lines.append("1. Implement plugin skeleton with deterministic seed protocol.")
    lines.append("2. Add plugin to benchmark matrix with `auto_t3_controlled` and `auto_t5_guarded` cross-check.")
    lines.append("3. Run canonical validation gate and dry-run profile.")
    lines.append("4. Produce acta + decision JSON with promote/iterate outcome.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_extension_contracts_md() -> str:
    lines: list[str] = []
    lines.append("# Week14 Block6 Extension Contracts")
    lines.append("")
    lines.append("## Contract A - Benchmark Interface")
    lines.append("")
    lines.append("- Entry point: `src/benchmarking/production_kernel_benchmark.py`")
    lines.append("- Required output fields:")
    lines.append("  - throughput (`avg_gflops`, `peak_gflops`), latency (`time_ms`), correctness (`max_error`).")
    lines.append("- Deterministic controls:")
    lines.append("  - fixed seed path, explicit platform selector, explicit policy paths.")
    lines.append("")
    lines.append("## Contract B - Validation Gate")
    lines.append("")
    lines.append("- Entry point: `scripts/run_validation_suite.py --tier canonical --driver-smoke`")
    lines.append("- Required status: `promote` before merge/promotion.")
    lines.append("- Required checks:")
    lines.append("  - schema validation green, pytest tier green, verify_drivers JSON parse green.")
    lines.append("")
    lines.append("## Contract C - Runtime Health")
    lines.append("")
    lines.append("- Entry point: `scripts/verify_drivers.py --json`")
    lines.append("- Required field: `overall_status=good` on target host.")
    lines.append("- Runtime split support:")
    lines.append("  - Clover baseline required; rusticl split optional but encouraged.")
    lines.append("")
    lines.append("## Contract D - Safety and Rollback")
    lines.append("")
    lines.append("- Rollback entry: `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`")
    lines.append("- SLA reference: `research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`")
    lines.append("- Any correctness breach or disable-event spike forces `iterate/no-go` state.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_plugin_template_md() -> str:
    lines: list[str] = []
    lines.append("# Plugin Starter Template (Week14 Block6)")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("- `plugin_id`:")
    lines.append("- `owner`:")
    lines.append("- `target_sizes`: [1400, 2048]")
    lines.append("- `policy_path`:")
    lines.append("")
    lines.append("## Required Commands")
    lines.append("")
    lines.append("1. `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke`")
    lines.append("2. `./venv/bin/python scripts/verify_drivers.py --json`")
    lines.append("3. Controlled dry-run command for plugin profile")
    lines.append("")
    lines.append("## Required Evidence")
    lines.append("")
    lines.append("- `results.json` (schema-compatible)")
    lines.append("- Execution summary `.md`")
    lines.append("- Formal decision JSON (`promote|iterate|refine|stop`)")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block6 Framework Handoff Package")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Block4 decision: `{payload['metadata']['block4_decision']}`")
    lines.append(f"- Block5 decision: `{payload['metadata']['block5_decision']}`")
    lines.append(f"- Canonical gate: `{payload['metadata']['canonical_gate_decision']}`")
    lines.append("")
    lines.append("## Deliverables")
    lines.append("")
    lines.append(f"- Handoff guide: `{payload['artifacts']['handoff_guide_path']}`")
    lines.append(f"- Extension contracts: `{payload['artifacts']['extension_contracts_path']}`")
    lines.append(f"- Plugin template: `{payload['artifacts']['plugin_template_path']}`")
    lines.append(f"- Compatibility matrix: `{payload['artifacts']['compatibility_matrix_path']}`")
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
    parser = argparse.ArgumentParser(description="Build Week14 Block6 framework handoff package.")
    parser.add_argument(
        "--block4-report-path",
        default="research/breakthrough_lab/week14_controlled_rollout/week14_block4_prerelease_package_20260210_002923.json",
    )
    parser.add_argument(
        "--block5-report-path",
        default="research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_hardened_v2_20260210_004609.json",
    )
    parser.add_argument("--canonical-gate-path", required=True)
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json",
    )
    parser.add_argument(
        "--runbook-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK4_RX590_PRERELEASE_RUNBOOK.md",
    )
    parser.add_argument(
        "--plugin-checklist-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK4_PLUGIN_PROJECT_BASE_CHECKLIST.md",
    )
    parser.add_argument(
        "--block5-checklist-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md",
    )
    parser.add_argument(
        "--rollback-sla-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md",
    )
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week14_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week14_block6_framework_handoff")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    block4_path = (repo_root / args.block4_report_path).resolve()
    block5_path = (repo_root / args.block5_report_path).resolve()
    gate_path = (repo_root / args.canonical_gate_path).resolve()
    policy_path = (repo_root / args.policy_path).resolve()
    runbook_path = (repo_root / args.runbook_path).resolve()
    plugin_checklist_path = (repo_root / args.plugin_checklist_path).resolve()
    block5_checklist_path = (repo_root / args.block5_checklist_path).resolve()
    rollback_sla_path = (repo_root / args.rollback_sla_path).resolve()
    preprod_dir = (repo_root / args.preprod_signoff_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve()

    preprod_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    block4_decision = _extract_decision(_read_json(block4_path))
    block5_decision = _extract_decision(_read_json(block5_path))
    gate_decision = _extract_decision(_read_json(gate_path))

    handoff_guide_path = preprod_dir / "WEEK14_BLOCK6_FRAMEWORK_HANDOFF.md"
    extension_contracts_path = preprod_dir / "WEEK14_BLOCK6_EXTENSION_CONTRACTS.md"
    plugin_template_path = preprod_dir / "WEEK14_BLOCK6_PLUGIN_TEMPLATE.md"
    compatibility_matrix_path = preprod_dir / "WEEK14_BLOCK6_COMPATIBILITY_MATRIX.json"

    handoff_guide_path.write_text(
        _render_handoff_md(
            policy_path=str(policy_path),
            runbook_path=str(runbook_path),
            plugin_checklist_path=str(plugin_checklist_path),
            block5_checklist_path=str(block5_checklist_path),
            rollback_sla_path=str(rollback_sla_path),
        )
    )
    extension_contracts_path.write_text(_render_extension_contracts_md())
    plugin_template_path.write_text(_render_plugin_template_md())

    compatibility_matrix = {
        "matrix_id": "week14-block6-framework-compatibility-v1-2026-02-10",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_hardware": {
            "gpu": "AMD Radeon RX 590 GME",
            "driver_baseline": "amdgpu + Mesa 25.0.7",
            "opencl_primary": "Clover",
            "opencl_secondary": "rusticl"
        },
        "required_profiles": {
            "selector_modes": ["auto_t3_controlled", "auto_t5_guarded"],
            "size_scope_baseline": [1400, 2048],
            "correctness_max_error": 0.001,
            "t5_disable_events_total": 0
        },
        "extension_readiness": {
            "contracts_doc": str(extension_contracts_path),
            "plugin_template": str(plugin_template_path),
            "weekly_policy": str(policy_path),
            "rollback_sla": str(rollback_sla_path)
        }
    }
    _write_json(compatibility_matrix_path, compatibility_matrix)

    checks: dict[str, dict[str, Any]] = {}
    checks["block4_promote"] = {
        "observed": block4_decision,
        "required": "promote",
        "pass": block4_decision == "promote",
    }
    checks["block5_go"] = {
        "observed": block5_decision,
        "required": "go",
        "pass": block5_decision == "go",
    }
    checks["canonical_gate_promote"] = {
        "observed": gate_decision,
        "required": "promote",
        "pass": gate_decision == "promote",
    }
    checks["handoff_docs_written"] = {
        "observed": handoff_guide_path.exists() and extension_contracts_path.exists() and plugin_template_path.exists(),
        "required": True,
        "pass": handoff_guide_path.exists() and extension_contracts_path.exists() and plugin_template_path.exists(),
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Framework handoff package is complete with green upstream pre-release gates."
        if decision == "promote"
        else "Handoff package generated but upstream release gates are not fully green."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "block4_decision": block4_decision,
            "block5_decision": block5_decision,
            "canonical_gate_decision": gate_decision,
            "block4_report_path": str(block4_path),
            "block5_report_path": str(block5_path),
            "canonical_gate_path": str(gate_path),
        },
        "artifacts": {
            "handoff_guide_path": str(handoff_guide_path),
            "extension_contracts_path": str(extension_contracts_path),
            "plugin_template_path": str(plugin_template_path),
            "compatibility_matrix_path": str(compatibility_matrix_path),
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
    md_path.write_text(_report_md(payload))

    print(f"Week14 block6 JSON: {json_path}")
    print(f"Week14 block6 MD:   {md_path}")
    print(f"Handoff guide:      {handoff_guide_path}")
    print(f"Contracts doc:      {extension_contracts_path}")
    print(f"Plugin template:    {plugin_template_path}")
    print(f"Compatibility JSON: {compatibility_matrix_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
