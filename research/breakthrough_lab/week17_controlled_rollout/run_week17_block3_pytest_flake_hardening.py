#!/usr/bin/env python3
"""Week 17 Block 3: harden pytest tier against flaky rectangular GEMM test."""

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


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _extract_prefixed_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        row = line.strip()
        if row.startswith(prefix):
            return row[len(prefix) :].strip()
    return None


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 17 Block 3 - Pytest Flake Hardening")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Target test: `{report['metadata']['target_test']}`")
    lines.append(f"- Repeats: {report['metadata']['repeat_count']}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Repeat Summary")
    lines.append("")
    lines.append(f"- Passed runs: {report['repeat_summary']['passed_runs']}")
    lines.append(f"- Failed runs: {report['repeat_summary']['failed_runs']}")
    lines.append(f"- Failure indexes: {report['repeat_summary']['failed_indexes']}")
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
    parser = argparse.ArgumentParser(description="Run Week17 Block3 pytest flake hardening campaign.")
    parser.add_argument(
        "--baseline-failure-validation-path",
        default="research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_014604.json",
    )
    parser.add_argument(
        "--target-test",
        default="tests/test_optimized_kernel_engine.py::TestGEMMCorrectness::test_gemm_rectangular",
    )
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week17_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week17_block3_pytest_flake_hardening")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = (REPO_ROOT / args.baseline_failure_validation_path).resolve()
    baseline_has_pytest_failure = False
    baseline_failed_checks: list[str] = []
    if baseline_path.exists():
        baseline_payload = _read_json(baseline_path)
        baseline_failed_checks = list(
            baseline_payload.get("evaluation", {}).get("failed_checks", [])
        )
        baseline_has_pytest_failure = "pytest_tier_green" in baseline_failed_checks

    repeat_runs: list[dict[str, Any]] = []
    failed_indexes: list[int] = []
    for idx in range(args.repeat_count):
        run = _run(
            [
                "./venv/bin/python",
                "-m",
                "pytest",
                "-q",
                args.target_test,
            ]
        )
        repeat_runs.append(run)
        if run["returncode"] != 0:
            failed_indexes.append(idx + 1)

    post_gate = _run(
        [
            "./venv/bin/python",
            "scripts/run_validation_suite.py",
            "--tier",
            "canonical",
            "--driver-smoke",
            "--report-dir",
            args.report_dir,
        ]
    )
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json((REPO_ROOT / post_gate_json).resolve()).get("evaluation", {}).get("decision", "unknown")
        )

    passed_runs = args.repeat_count - len(failed_indexes)
    checks: dict[str, dict[str, Any]] = {}
    checks["baseline_contains_pytest_failure"] = {
        "observed": baseline_has_pytest_failure,
        "required": True,
        "pass": baseline_has_pytest_failure,
    }
    checks["repeat_campaign_all_green"] = {
        "observed_passed_runs": int(passed_runs),
        "required_passed_runs": int(args.repeat_count),
        "pass": len(failed_indexes) == 0,
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Flake hardening is effective: repeated target test and canonical gate are stable."
        if decision == "promote"
        else "Flake hardening is incomplete; repeat campaign or gate still failing."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_test": args.target_test,
            "repeat_count": int(args.repeat_count),
            "baseline_failure_validation_path": str(baseline_path),
        },
        "commands": {
            "repeat_cmd": " ".join(
                shlex.quote(x)
                for x in [
                    "./venv/bin/python",
                    "-m",
                    "pytest",
                    "-q",
                    args.target_test,
                ]
            ),
            "repeat_runs": repeat_runs,
            "post_gate": post_gate,
        },
        "repeat_summary": {
            "passed_runs": int(passed_runs),
            "failed_runs": int(len(failed_indexes)),
            "failed_indexes": failed_indexes,
        },
        "artifacts": {
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

    print(f"Week17 block3 JSON: {json_path}")
    print(f"Week17 block3 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 9


if __name__ == "__main__":
    raise SystemExit(main())
