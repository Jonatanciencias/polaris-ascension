#!/usr/bin/env python3
"""Week 17 Block 2: second dependent project/plugin pilot on stable manifest."""

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


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 17 Block 2 - Second Dependent Pilot (Stable Manifest)")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
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
    parser = argparse.ArgumentParser(description="Run Week17 Block2 second dependent pilot.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--plugin-template-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK6_PLUGIN_TEMPLATE.md",
    )
    parser.add_argument(
        "--dependent-project-dir",
        default="research/breakthrough_lab/dependent_projects/rx590_stable_integration_pilot_v2",
    )
    parser.add_argument("--plugin-id", default="rx590_dependent_project_week17_block2")
    parser.add_argument("--owner", default="dependent-project-team-v2")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=27111)
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json",
    )
    parser.add_argument(
        "--baseline-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.json",
    )
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week17_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week17_block2_second_dependent_pilot")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    plugin_template_path = _resolve(args.plugin_template_path)
    dependent_dir = _resolve(args.dependent_project_dir)
    output_dir = _resolve(args.output_dir)
    dependent_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    stable_manifest: dict[str, Any] = {}
    stable_tag = "unknown"
    referenced_paths: list[tuple[str, Path]] = []
    if stable_manifest_path.exists():
        stable_manifest = _read_json(stable_manifest_path)
        stable_tag = str(stable_manifest.get("stable_tag", "unknown"))
        referenced_paths = [
            ("block1_report_path", _resolve(str(stable_manifest.get("block1_report_path", "")))),
            ("block2_report_path", _resolve(str(stable_manifest.get("block2_report_path", "")))),
            ("canonical_gate_path", _resolve(str(stable_manifest.get("canonical_gate_path", "")))),
            ("rc_manifest_path", _resolve(str(stable_manifest.get("rc_manifest_path", "")))),
        ]

    integration_profile = {
        "integration_id": "week17-block2-rx590-stable-dependent-pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stable_manifest_path": str(stable_manifest_path),
        "stable_tag": stable_tag,
        "plugin_id": args.plugin_id,
        "owner": args.owner,
        "sizes": [int(s) for s in args.sizes],
        "sessions": int(args.sessions),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "policy_paths": {
            "t5": str(_resolve(args.t5_policy_path)),
            "baseline": str(_resolve(args.baseline_path)),
            "template": str(plugin_template_path),
        },
        "commands": {
            "canonical_gate": " ".join(shlex.quote(x) for x in pre_gate_cmd),
            "plugin_pilot": "week15 block2 runner reused with stable manifest context",
        },
    }
    profile_path = dependent_dir / "week17_block2_integration_profile.json"
    _write_json(profile_path, integration_profile)

    plugin_cmd = [
        "./venv/bin/python",
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
        f"{args.output_prefix}_plugin",
    ]
    plugin = _run(plugin_cmd)
    plugin_json = _extract_prefixed_line(plugin["stdout"], "Week15 block2 JSON:")
    plugin_md = _extract_prefixed_line(plugin["stdout"], "Week15 block2 MD:")
    plugin_results = _extract_prefixed_line(plugin["stdout"], "Week15 block2 results.json:")
    plugin_decision = _extract_prefixed_line(plugin["stdout"], "Decision:")

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    referenced_ok = all(path.exists() for _, path in referenced_paths) if referenced_paths else False
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
    checks["stable_status_proposed_or_stable"] = {
        "observed": str(stable_manifest.get("status", "unknown")),
        "required_in": ["proposed_stable", "stable"],
        "pass": str(stable_manifest.get("status", "")) in {"proposed_stable", "stable"},
    }
    checks["stable_manifest_references_exist"] = {
        "observed": referenced_ok,
        "required": True,
        "pass": referenced_ok,
    }
    checks["plugin_template_exists"] = {
        "observed": plugin_template_path.exists(),
        "required": True,
        "pass": plugin_template_path.exists(),
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
        "Second dependent pilot is stable over v0.15.0 manifest with mandatory gates green."
        if decision == "promote"
        else "Second dependent pilot has unresolved checks in stable integration flow."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "dependent_project_dir": str(dependent_dir),
            "plugin_id": args.plugin_id,
            "owner": args.owner,
            "sizes": [int(s) for s in args.sizes],
        },
        "commands": {
            "pre_gate": pre_gate,
            "plugin_pilot": plugin,
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

    json_path = output_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, report)
    md_path.write_text(_md(report))

    print(f"Week17 block2 JSON: {json_path}")
    print(f"Week17 block2 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 7


if __name__ == "__main__":
    raise SystemExit(main())
