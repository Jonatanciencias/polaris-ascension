#!/usr/bin/env python3
"""Week 16 Block 1: dependent-project integration pilot from RC manifest."""

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
DEFAULT_TEMPLATE = "research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK6_PLUGIN_TEMPLATE.md"
DEFAULT_T5_POLICY = (
    "research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json"
)
DEFAULT_BASELINE = (
    "research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_canary_20260210_011736.json"
)
DEFAULT_DEPENDENT_DIR = "research/breakthrough_lab/dependent_projects/rx590_rc_integration_pilot"
DEFAULT_VALIDATION_DIR = "research/breakthrough_lab/week8_validation_discipline"
DEFAULT_OUTPUT_DIR = "research/breakthrough_lab/week16_controlled_rollout"


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


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 16 Block 1 - Dependent Project Integration Pilot")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- RC tag: `{report['metadata']['rc_tag']}`")
    lines.append(f"- Dependent project dir: `{report['metadata']['dependent_project_dir']}`")
    lines.append(f"- Plugin ID: `{report['metadata']['plugin_id']}`")
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
    parser = argparse.ArgumentParser(
        description="Run Week16 Block1 dependent-project integration pilot."
    )
    parser.add_argument("--rc-manifest-path", default=DEFAULT_RC_MANIFEST)
    parser.add_argument("--plugin-template-path", default=DEFAULT_TEMPLATE)
    parser.add_argument("--dependent-project-dir", default=DEFAULT_DEPENDENT_DIR)
    parser.add_argument("--plugin-id", default="rx590_dependent_project_week16_block1")
    parser.add_argument("--owner", default="dependent-project-team")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26011)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE)
    parser.add_argument("--validation-report-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="week16_block1_dependent_integration")
    args = parser.parse_args()

    manifest_path = _resolve(args.rc_manifest_path)
    template_path = _resolve(args.plugin_template_path)
    dependent_dir = _resolve(args.dependent_project_dir)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dependent_dir.mkdir(parents=True, exist_ok=True)

    pre_gate_cmd = [
        sys.executable,
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.validation_report_dir),
    ]
    pre_gate = _run(pre_gate_cmd)
    pre_gate_json = _extract_prefixed_line(pre_gate["stdout"], "Wrote JSON report:")
    pre_gate_decision = "unknown"
    if pre_gate_json:
        pre_gate_decision = str(
            _read_json(_resolve(pre_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    manifest_payload: dict[str, Any] = {}
    manifest_deps: list[tuple[str, Path]] = []
    if manifest_path.exists():
        manifest_payload = _read_json(manifest_path)
        deliverables = manifest_payload.get("deliverables", {})
        manifest_deps = [
            ("block1_report_path", _resolve(str(manifest_payload.get("block1_report_path", "")))),
            ("block2_report_path", _resolve(str(manifest_payload.get("block2_report_path", "")))),
            ("canonical_gate_path", _resolve(str(manifest_payload.get("canonical_gate_path", "")))),
            ("release_notes_path", _resolve(str(deliverables.get("release_notes_path", "")))),
            ("adoption_checklist_path", _resolve(str(deliverables.get("adoption_checklist_path", "")))),
            ("dependent_projects_path", _resolve(str(deliverables.get("dependent_projects_path", "")))),
        ]

    integration_profile = {
        "integration_id": "week16-block1-rx590-dependent-pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rc_manifest_path": str(manifest_path),
        "rc_tag": str(manifest_payload.get("rc_tag", "unknown")),
        "plugin_id": args.plugin_id,
        "owner": args.owner,
        "sizes": [int(s) for s in args.sizes],
        "sessions": int(args.sessions),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "policy_paths": {
            "t5": str(_resolve(args.t5_policy_path)),
            "baseline": str(_resolve(args.baseline_path)),
            "template": str(template_path),
        },
        "commands": {
            "gate": " ".join(shlex.quote(x) for x in pre_gate_cmd),
            "plugin_pilot": "week15_block2 runner with week16 scope",
        },
    }
    profile_path = dependent_dir / "week16_block1_integration_profile.json"
    _write_json(profile_path, integration_profile)

    plugin_prefix = f"{args.output_prefix}_plugin_pilot"
    plugin_cmd = [
        sys.executable,
        "research/breakthrough_lab/week15_controlled_rollout/run_week15_block2_plugin_pilot.py",
        "--plugin-id",
        str(args.plugin_id),
        "--owner",
        str(args.owner),
        "--sizes",
        *[str(s) for s in args.sizes],
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--t5-policy-path",
        str(args.t5_policy_path),
        "--baseline-path",
        str(args.baseline_path),
        "--template-path",
        str(args.plugin_template_path),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        plugin_prefix,
    ]
    plugin_run = _run(plugin_cmd)
    plugin_json = _extract_prefixed_line(plugin_run["stdout"], "Week15 block2 JSON:")
    plugin_md = _extract_prefixed_line(plugin_run["stdout"], "Week15 block2 MD:")
    plugin_results = _extract_prefixed_line(
        plugin_run["stdout"], "Week15 block2 results.json:"
    )
    plugin_decision = _extract_prefixed_line(plugin_run["stdout"], "Decision:")

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    deps_exists = all(path.exists() for _, path in manifest_deps) if manifest_deps else False
    checks: dict[str, dict[str, Any]] = {}
    checks["rc_manifest_exists"] = {
        "observed": manifest_path.exists(),
        "required": True,
        "pass": manifest_path.exists(),
    }
    checks["rc_manifest_status_candidate"] = {
        "observed": str(manifest_payload.get("status", "unknown")),
        "required": "candidate",
        "pass": str(manifest_payload.get("status", "")) == "candidate",
    }
    checks["rc_manifest_dependencies_exist"] = {
        "observed": deps_exists,
        "required": True,
        "pass": deps_exists,
    }
    checks["plugin_template_exists"] = {
        "observed": template_path.exists(),
        "required": True,
        "pass": template_path.exists(),
    }
    checks["integration_profile_written"] = {
        "observed": profile_path.exists(),
        "required": True,
        "pass": profile_path.exists(),
    }
    checks["plugin_pilot_promote"] = {
        "observed": str(plugin_decision or "unknown").lower(),
        "required": "promote",
        "pass": str(plugin_decision or "").lower() == "promote",
    }
    checks["plugin_results_exists"] = {
        "observed": bool(plugin_results) and _resolve(str(plugin_results)).exists(),
        "required": True,
        "pass": bool(plugin_results) and _resolve(str(plugin_results)).exists(),
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

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Dependent project consumed RC manifest and completed integration pilot with all mandatory gates green."
        if decision == "promote"
        else "Dependent integration pilot has unresolved mandatory checks."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rc_tag": str(manifest_payload.get("rc_tag", "unknown")),
            "rc_manifest_path": str(manifest_path),
            "dependent_project_dir": str(dependent_dir),
            "plugin_id": args.plugin_id,
            "owner": args.owner,
            "sizes": [int(s) for s in args.sizes],
        },
        "commands": {
            "pre_gate": pre_gate,
            "plugin_pilot": plugin_run,
            "post_gate": post_gate,
        },
        "artifacts": {
            "integration_profile_json": str(profile_path),
            "plugin_json": plugin_json,
            "plugin_md": plugin_md,
            "plugin_results_json": plugin_results,
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

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, report)
    md_path.write_text(_md(report))

    print(f"Week16 block1 JSON: {json_path}")
    print(f"Week16 block1 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 6


if __name__ == "__main__":
    raise SystemExit(main())
