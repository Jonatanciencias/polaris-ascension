#!/usr/bin/env python3
"""Week 15 Block 3: framework RC package for dependent-project adoption."""

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
    for key in ("decision", "block_decision", "operational_decision", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    eval_decision = payload.get("evaluation", {}).get("decision")
    if isinstance(eval_decision, str) and eval_decision:
        return eval_decision
    return "unknown"


def _release_notes_md(*, rc_tag: str, block1_path: str, block2_path: str) -> str:
    lines: list[str] = []
    lines.append(f"# Framework Release Candidate {rc_tag}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Controlled-production baseline for RX590 with extension/plugin handoff contracts.")
    lines.append("")
    lines.append("## Evidence Baseline")
    lines.append("")
    lines.append(f"- Week15 Block1 report: `{block1_path}`")
    lines.append(f"- Week15 Block2 report: `{block2_path}`")
    lines.append("")
    lines.append("## Included Capabilities")
    lines.append("")
    lines.append("- Stable canonical validation gate (`tier=canonical`, `driver-smoke`).")
    lines.append("- Controlled pilot profile for sizes `1400/2048/3072` with rollback SLA.")
    lines.append("- Plugin onboarding template and extension contracts (Week14 Block6).")
    lines.append("- Formal decision workflow (`promote|iterate|refine|stop` or `go|no-go`).")
    lines.append("")
    lines.append("## Known Limits")
    lines.append("")
    lines.append("- Expansion beyond controlled scope requires fresh canary evidence.")
    lines.append("- Any T5 disable-event spike requires immediate rollback protocol.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _adoption_checklist_md() -> str:
    lines: list[str] = []
    lines.append("# Week15 Block3 RC Adoption Checklist")
    lines.append("")
    lines.append("## Repository Readiness")
    lines.append("")
    lines.append("- [ ] Repository can run canonical gate locally.")
    lines.append("- [ ] Repository can parse `verify_drivers.py --json` output.")
    lines.append("- [ ] Repository consumes policy paths through explicit config/env.")
    lines.append("")
    lines.append("## Runtime Readiness")
    lines.append("")
    lines.append("- [ ] Clover baseline validated on RX590 host.")
    lines.append("- [ ] rusticl split tested (optional but recommended).")
    lines.append("- [ ] Rollback script tested in dry-run mode.")
    lines.append("")
    lines.append("## Integration Readiness")
    lines.append("")
    lines.append("- [ ] Plugin/project emits JSON+MD evidence per run.")
    lines.append("- [ ] Project can store formal decision JSON per block.")
    lines.append("- [ ] Project enforces `max_error <= 1e-3` and guardrail checks.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _dependent_projects_md() -> str:
    lines: list[str] = []
    lines.append("# Week15 Block3 Dependent Projects Onboarding")
    lines.append("")
    lines.append("## Tier A - Core Runtime Consumers")
    lines.append("")
    lines.append("- Inference runtimes using GEMM selector in production profile.")
    lines.append("- Benchmark suites consuming `run_production_benchmark` contracts.")
    lines.append("")
    lines.append("## Tier B - Extension / Plugin Projects")
    lines.append("")
    lines.append("- Kernel-policy plugins adding strategy metadata and telemetry.")
    lines.append("- Reliability plugins extending T5 checks without bypassing guardrails.")
    lines.append("")
    lines.append("## Tier C - Ops / CI Projects")
    lines.append("")
    lines.append("- CI jobs enforcing canonical gate and schema checks.")
    lines.append("- Monitoring jobs running weekly replay + drift control.")
    lines.append("")
    lines.append("## Adoption Sequence")
    lines.append("")
    lines.append("1. Integrate template from `WEEK14_BLOCK6_PLUGIN_TEMPLATE.md`.")
    lines.append("2. Run canonical gate + driver smoke.")
    lines.append("3. Execute controlled dry-run profile and collect evidence.")
    lines.append("4. Close local acta + decision before scope expansion.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week15 Block3 Framework RC Package")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- RC tag: `{payload['metadata']['rc_tag']}`")
    lines.append(f"- Block1 decision: `{payload['metadata']['block1_decision']}`")
    lines.append(f"- Block2 decision: `{payload['metadata']['block2_decision']}`")
    lines.append(f"- Canonical gate: `{payload['metadata']['canonical_gate_decision']}`")
    lines.append("")
    lines.append("## Deliverables")
    lines.append("")
    lines.append(f"- Release notes: `{payload['artifacts']['release_notes_path']}`")
    lines.append(f"- Adoption checklist: `{payload['artifacts']['adoption_checklist_path']}`")
    lines.append(f"- Dependent projects guide: `{payload['artifacts']['dependent_projects_path']}`")
    lines.append(f"- RC manifest: `{payload['artifacts']['manifest_path']}`")
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
    parser = argparse.ArgumentParser(description="Build Week15 Block3 framework RC package.")
    parser.add_argument(
        "--block1-report-path",
        default="research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_20260210_011756.json",
    )
    parser.add_argument(
        "--block2-report-path",
        default="research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_rerun_20260210_012358.json",
    )
    parser.add_argument("--canonical-gate-path", required=True)
    parser.add_argument("--rc-tag", default="v0.15.0-rc1")
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week15_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week15_block3_framework_rc")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    block1_path = (repo_root / args.block1_report_path).resolve()
    block2_path = (repo_root / args.block2_report_path).resolve()
    gate_path = (repo_root / args.canonical_gate_path).resolve()
    preprod_dir = (repo_root / args.preprod_signoff_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve()

    preprod_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    block1_decision = _extract_decision(_read_json(block1_path))
    block2_decision = _extract_decision(_read_json(block2_path))
    gate_decision = _extract_decision(_read_json(gate_path))

    release_notes_path = preprod_dir / "WEEK15_BLOCK3_FRAMEWORK_RC_RELEASE_NOTES.md"
    adoption_checklist_path = preprod_dir / "WEEK15_BLOCK3_FRAMEWORK_RC_ADOPTION_CHECKLIST.md"
    dependent_projects_path = preprod_dir / "WEEK15_BLOCK3_DEPENDENT_PROJECTS_ONBOARDING.md"
    manifest_path = preprod_dir / "WEEK15_BLOCK3_FRAMEWORK_RC_MANIFEST.json"

    release_notes_path.write_text(
        _release_notes_md(
            rc_tag=args.rc_tag,
            block1_path=str(block1_path),
            block2_path=str(block2_path),
        )
    )
    adoption_checklist_path.write_text(_adoption_checklist_md())
    dependent_projects_path.write_text(_dependent_projects_md())

    manifest_payload = {
        "rc_tag": args.rc_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "block1_report_path": str(block1_path),
        "block2_report_path": str(block2_path),
        "canonical_gate_path": str(gate_path),
        "deliverables": {
            "release_notes_path": str(release_notes_path),
            "adoption_checklist_path": str(adoption_checklist_path),
            "dependent_projects_path": str(dependent_projects_path),
        },
        "status": "candidate",
    }
    _write_json(manifest_path, manifest_payload)

    checks: dict[str, dict[str, Any]] = {}
    checks["block1_go"] = {
        "observed": block1_decision,
        "required": "go",
        "pass": block1_decision == "go",
    }
    checks["block2_promote"] = {
        "observed": block2_decision,
        "required": "promote",
        "pass": block2_decision == "promote",
    }
    checks["canonical_gate_promote"] = {
        "observed": gate_decision,
        "required": "promote",
        "pass": gate_decision == "promote",
    }
    checks["rc_docs_written"] = {
        "observed": (
            release_notes_path.exists()
            and adoption_checklist_path.exists()
            and dependent_projects_path.exists()
            and manifest_path.exists()
        ),
        "required": True,
        "pass": (
            release_notes_path.exists()
            and adoption_checklist_path.exists()
            and dependent_projects_path.exists()
            and manifest_path.exists()
        ),
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Framework RC package is publishable for dependent-project adoption."
        if decision == "promote"
        else "RC package generated but not publishable until upstream gates are green."
    )

    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    payload = {
        "metadata": {
            "timestamp_utc": timestamp.isoformat(),
            "rc_tag": args.rc_tag,
            "block1_decision": block1_decision,
            "block2_decision": block2_decision,
            "canonical_gate_decision": gate_decision,
            "block1_report_path": str(block1_path),
            "block2_report_path": str(block2_path),
            "canonical_gate_path": str(gate_path),
        },
        "artifacts": {
            "release_notes_path": str(release_notes_path),
            "adoption_checklist_path": str(adoption_checklist_path),
            "dependent_projects_path": str(dependent_projects_path),
            "manifest_path": str(manifest_path),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    report_json = out_dir / f"{args.output_prefix}_{stamp}.json"
    report_md = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(report_json, payload)
    report_md.write_text(_report_md(payload))

    print(f"Week15 block3 JSON: {report_json}")
    print(f"Week15 block3 MD:   {report_md}")
    print(f"Release notes:      {release_notes_path}")
    print(f"Adoption checklist: {adoption_checklist_path}")
    print(f"Dependent guide:    {dependent_projects_path}")
    print(f"RC manifest:        {manifest_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
