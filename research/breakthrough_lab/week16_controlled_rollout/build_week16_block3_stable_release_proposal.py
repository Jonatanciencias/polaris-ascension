#!/usr/bin/env python3
"""Week 16 Block 3: stable v0.15.0 release proposal from RC evidence."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RC_MANIFEST = (
    "research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_FRAMEWORK_RC_MANIFEST.json"
)
DEFAULT_VALIDATION_DIR = "research/breakthrough_lab/week8_validation_discipline"
DEFAULT_OUTPUT_DIR = "research/breakthrough_lab/week16_controlled_rollout"
DEFAULT_SIGNOFF_DIR = "research/breakthrough_lab/preprod_signoff"


def _resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
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


def _extract_decision(payload: dict[str, Any]) -> str:
    for key in ("decision", "block_decision", "operational_decision", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    eval_decision = payload.get("evaluation", {}).get("decision")
    if isinstance(eval_decision, str) and eval_decision:
        return eval_decision
    return "unknown"


def _release_notes_md(
    *,
    stable_tag: str,
    rc_tag: str,
    block1_report: Path,
    block2_report: Path,
) -> str:
    lines: list[str] = []
    lines.append(f"# Stable Release Proposal {stable_tag}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Source RC: `{rc_tag}`")
    lines.append(f"- Proposed stable tag: `{stable_tag}`")
    lines.append("")
    lines.append("## Evidence Chain")
    lines.append("")
    lines.append(f"- Week16 Block1 integration pilot: `{block1_report}`")
    lines.append(f"- Week16 Block2 weekly replay + drift: `{block2_report}`")
    lines.append("")
    lines.append("## Stable Scope")
    lines.append("")
    lines.append("- Deterministic controlled profile for `1400/2048/3072`.")
    lines.append("- Mandatory canonical gate before promotion and scope expansion.")
    lines.append("- Guardrails: correctness, T3 fallback, T5 overhead/disable events.")
    lines.append("")
    lines.append("## Deferred")
    lines.append("")
    lines.append("- Any new platform scope or larger sizes beyond `3072` need fresh evidence.")
    lines.append("- Plugin API major changes remain out-of-scope for `v0.15.0`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _release_checklist_md(stable_tag: str) -> str:
    lines: list[str] = []
    lines.append(f"# {stable_tag} Final Release Checklist")
    lines.append("")
    lines.append("- [ ] Canonical gate green (`run_validation_suite.py --tier canonical --driver-smoke`).")
    lines.append("- [ ] Week16 Block1 decision = `promote`.")
    lines.append("- [ ] Week16 Block2 decision = `promote`.")
    lines.append("- [ ] Weekly SLO policy compliance confirmed.")
    lines.append("- [ ] Drift report archived and reviewed.")
    lines.append("- [ ] Runbook + rollback SLA available to operators.")
    lines.append("- [ ] Stable manifest published and linked in roadmap.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _release_runbook_md(stable_tag: str) -> str:
    lines: list[str] = []
    lines.append(f"# {stable_tag} Controlled Release Runbook")
    lines.append("")
    lines.append("## Pre-flight")
    lines.append("")
    lines.append("1. Run canonical gate with driver smoke.")
    lines.append("2. Verify platform diagnostics (`verify_drivers.py --json`).")
    lines.append("3. Verify latest weekly replay report is `promote`.")
    lines.append("")
    lines.append("## Release")
    lines.append("")
    lines.append("1. Tag release candidate lineage as stable (`v0.15.0`).")
    lines.append("2. Publish stable manifest and notes.")
    lines.append("3. Keep controlled rollout mode for first production window.")
    lines.append("")
    lines.append("## Rollback")
    lines.append("")
    lines.append("1. If `disable_events > 0` or overhead guardrail breach, rollback immediately.")
    lines.append("2. Revert to last RC-known-good policy and rerun canonical gate.")
    lines.append("3. Open incident note and freeze scope expansion.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 16 Block 3 - Stable Release Proposal")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- RC tag: `{report['metadata']['rc_tag']}`")
    lines.append(f"- Proposed tag: `{report['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
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
    parser = argparse.ArgumentParser(description="Build Week16 Block3 stable release proposal.")
    parser.add_argument("--rc-manifest-path", default=DEFAULT_RC_MANIFEST)
    parser.add_argument("--block1-report-path", required=True)
    parser.add_argument("--block2-report-path", required=True)
    parser.add_argument("--stable-tag", default="v0.15.0")
    parser.add_argument("--validation-report-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--preprod-signoff-dir", default=DEFAULT_SIGNOFF_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="week16_block3_stable_release_proposal")
    args = parser.parse_args()

    manifest_path = _resolve(args.rc_manifest_path)
    block1_path = _resolve(args.block1_report_path)
    block2_path = _resolve(args.block2_report_path)
    signoff_dir = _resolve(args.preprod_signoff_dir)
    out_dir = _resolve(args.output_dir)
    signoff_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_cmd = [
        sys.executable,
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.validation_report_dir),
    ]
    gate_run = _run(gate_cmd)
    gate_json = _extract_prefixed_line(gate_run["stdout"], "Wrote JSON report:")
    gate_decision = "unknown"
    if gate_json:
        gate_decision = str(
            _read_json(_resolve(gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    manifest: dict[str, Any] = _read_json(manifest_path) if manifest_path.exists() else {}
    rc_tag = str(manifest.get("rc_tag", "unknown"))
    block1_decision = _extract_decision(_read_json(block1_path)) if block1_path.exists() else "unknown"
    block2_decision = _extract_decision(_read_json(block2_path)) if block2_path.exists() else "unknown"

    notes_path = signoff_dir / "WEEK16_BLOCK3_V0_15_0_RELEASE_NOTES.md"
    checklist_path = signoff_dir / "WEEK16_BLOCK3_V0_15_0_RELEASE_CHECKLIST.md"
    runbook_path = signoff_dir / "WEEK16_BLOCK3_V0_15_0_RELEASE_RUNBOOK.md"
    stable_manifest_path = signoff_dir / "WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json"

    notes_path.write_text(
        _release_notes_md(
            stable_tag=args.stable_tag,
            rc_tag=rc_tag,
            block1_report=block1_path,
            block2_report=block2_path,
        )
    )
    checklist_path.write_text(_release_checklist_md(args.stable_tag))
    runbook_path.write_text(_release_runbook_md(args.stable_tag))

    stable_manifest = {
        "stable_tag": args.stable_tag,
        "source_rc_tag": rc_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rc_manifest_path": str(manifest_path),
        "block1_report_path": str(block1_path),
        "block2_report_path": str(block2_path),
        "canonical_gate_path": str(_resolve(gate_json)) if gate_json else None,
        "status": "proposed_stable",
    }
    _write_json(stable_manifest_path, stable_manifest)

    checks: dict[str, dict[str, Any]] = {}
    checks["rc_manifest_exists"] = {
        "observed": manifest_path.exists(),
        "required": True,
        "pass": manifest_path.exists(),
    }
    checks["rc_source_is_expected"] = {
        "observed": rc_tag,
        "required": "v0.15.0-rc1",
        "pass": rc_tag == "v0.15.0-rc1",
    }
    checks["block1_promote"] = {
        "observed": block1_decision,
        "required": "promote",
        "pass": block1_decision == "promote",
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
    checks["release_docs_written"] = {
        "observed": (
            notes_path.exists()
            and checklist_path.exists()
            and runbook_path.exists()
            and stable_manifest_path.exists()
        ),
        "required": True,
        "pass": (
            notes_path.exists()
            and checklist_path.exists()
            and runbook_path.exists()
            and stable_manifest_path.exists()
        ),
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "RC lineage satisfies all gates and is ready to be proposed as stable v0.15.0."
        if decision == "promote"
        else "Stable proposal generated but one or more release gates are not green."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": args.stable_tag,
            "rc_tag": rc_tag,
            "rc_manifest_path": str(manifest_path),
        },
        "commands": {
            "canonical_gate": gate_run,
            "canonical_gate_cmd_pretty": " ".join(shlex.quote(x) for x in gate_cmd),
        },
        "artifacts": {
            "block1_report_json": str(block1_path),
            "block2_report_json": str(block2_path),
            "release_notes_md": str(notes_path),
            "release_checklist_md": str(checklist_path),
            "release_runbook_md": str(runbook_path),
            "stable_manifest_json": str(stable_manifest_path),
            "canonical_gate_json": gate_json,
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
    _write_json(json_path, report)
    md_path.write_text(_md(report))

    print(f"Week16 block3 JSON: {json_path}")
    print(f"Week16 block3 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
