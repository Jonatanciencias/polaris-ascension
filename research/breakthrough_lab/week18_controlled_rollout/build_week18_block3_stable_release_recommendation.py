#!/usr/bin/env python3
"""Week 18 Block 3: final stable release checklist + GO recommendation."""

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


def _checklist_md(*, stable_tag: str, checks: dict[str, dict[str, Any]], go: bool) -> str:
    lines: list[str] = []
    lines.append(f"# Week 18 Block 3 - Final Stable Release Checklist ({stable_tag})")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in checks.items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append(f"- Final recommendation: `{'GO' if go else 'NO-GO'}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _recommendation_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 3 - Stable Release Recommendation")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Block 1 decision: `{payload['inputs']['block1_decision']}`")
    lines.append(f"- Block 2 decision: `{payload['inputs']['block2_decision']}`")
    lines.append(
        f"- Block 2 rusticl/clover ratio min: `{payload['inputs']['block2_ratio_min']:.6f}`"
    )
    lines.append(
        f"- Block 2 T5 disable total: `{payload['inputs']['block2_t5_disable_total']}`"
    )
    lines.append("")
    lines.append("## Release Decision")
    lines.append("")
    lines.append(f"- Recommendation: `{payload['evaluation']['release_recommendation']}`")
    lines.append(f"- Block decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 3 - Final Stable Release Decision Report")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Block decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Release recommendation: `{payload['evaluation']['release_recommendation']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week18 Block3 stable release recommendation.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--block1-decision-path",
        default="research/breakthrough_lab/week18_block1_stable_operations_package_decision.json",
    )
    parser.add_argument(
        "--block2-report-path",
        required=True,
    )
    parser.add_argument(
        "--block2-decision-path",
        default="research/breakthrough_lab/week18_block2_maintenance_split_decision.json",
    )
    parser.add_argument("--min-rusticl-ratio", type=float, default=0.85)
    parser.add_argument("--max-t5-disable-total", type=int, default=0)
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
        default="research/breakthrough_lab/week18_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week18_block3_stable_release_recommendation")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    block1_decision_path = _resolve(args.block1_decision_path)
    block2_report_path = _resolve(args.block2_report_path)
    block2_decision_path = _resolve(args.block2_decision_path)
    preprod_dir = _resolve(args.preprod_signoff_dir)
    output_dir = _resolve(args.output_dir)
    preprod_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest = _read_json(stable_manifest_path) if stable_manifest_path.exists() else {}
    stable_tag = str(stable_manifest.get("stable_tag", "unknown"))
    block1_decision = _read_json(block1_decision_path) if block1_decision_path.exists() else {}
    block2_report = _read_json(block2_report_path) if block2_report_path.exists() else {}
    block2_decision = _read_json(block2_decision_path) if block2_decision_path.exists() else {}

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

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    block1_decision_value = str(block1_decision.get("block_decision", "unknown"))
    block2_decision_value = str(block2_decision.get("block_decision", "unknown"))
    block2_ratio_min = float(block2_report.get("split_metrics", {}).get("rusticl_ratio_min", 0.0))
    block2_t5_disable_total = int(
        block2_report.get("split_metrics", {}).get("canary_t5_disable_total", 999)
    )
    block2_t5_overhead_max = float(
        block2_report.get("split_metrics", {}).get("canary_t5_overhead_max", 999.0)
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
    checks["week18_block1_promote"] = {
        "observed": block1_decision_value,
        "required": "promote",
        "pass": block1_decision_value == "promote",
    }
    checks["week18_block2_promote"] = {
        "observed": block2_decision_value,
        "required": "promote",
        "pass": block2_decision_value == "promote",
    }
    checks["week18_block2_ratio_floor"] = {
        "observed_min": block2_ratio_min,
        "required_min": float(args.min_rusticl_ratio),
        "pass": block2_ratio_min >= float(args.min_rusticl_ratio),
    }
    checks["week18_block2_t5_disable_total"] = {
        "observed": block2_t5_disable_total,
        "required_max": int(args.max_t5_disable_total),
        "pass": block2_t5_disable_total <= int(args.max_t5_disable_total),
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
    go = not failed_checks
    decision = "promote" if go else "iterate"
    release_recommendation = "GO" if go else "NO-GO"
    rationale = (
        "Stable release checklist is fully green and v0.15.0 is recommended for controlled adoption."
        if go
        else "Stable release checklist has unresolved checks; keep NO-GO and iterate."
    )

    checklist_path = preprod_dir / "WEEK18_BLOCK3_FINAL_RELEASE_CHECKLIST.md"
    recommendation_path = preprod_dir / "WEEK18_BLOCK3_STABLE_RELEASE_RECOMMENDATION.md"
    go_no_go_path = preprod_dir / "WEEK18_BLOCK3_GO_NO_GO_DECISION.json"

    checklist_path.write_text(_checklist_md(stable_tag=stable_tag, checks=checks, go=go))

    go_no_go_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stable_tag": stable_tag,
        "recommendation": release_recommendation,
        "failed_checks": failed_checks,
        "checks": checks,
        "block2_metrics": {
            "rusticl_ratio_min": block2_ratio_min,
            "t5_disable_total": block2_t5_disable_total,
            "t5_overhead_max": block2_t5_overhead_max,
        },
    }
    _write_json(go_no_go_path, go_no_go_payload)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "block1_decision_path": str(block1_decision_path),
            "block2_report_path": str(block2_report_path),
            "block2_decision_path": str(block2_decision_path),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "inputs": {
            "block1_decision": block1_decision_value,
            "block2_decision": block2_decision_value,
            "block2_ratio_min": block2_ratio_min,
            "block2_t5_disable_total": block2_t5_disable_total,
            "block2_t5_overhead_max": block2_t5_overhead_max,
        },
        "artifacts": {
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
            "final_checklist_md": str(checklist_path),
            "release_recommendation_md": str(recommendation_path),
            "go_no_go_json": str(go_no_go_path),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "release_recommendation": release_recommendation,
            "rationale": rationale,
        },
    }
    recommendation_path.write_text(_recommendation_md(report))

    report_json = output_dir / f"{args.output_prefix}_{stamp}.json"
    report_md = output_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(report_json, report)
    report_md.write_text(_report_md(report))

    print(f"Week18 block3 JSON: {report_json}")
    print(f"Week18 block3 MD:   {report_md}")
    print(f"Recommendation: {release_recommendation}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
