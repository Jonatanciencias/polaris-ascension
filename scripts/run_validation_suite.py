#!/usr/bin/env python3
"""Unified validation runner for local and CI execution.

This runner standardizes validation entrypoints and artifacts across:
- local development validation
- CI fast tier validation
- optional repo-wide strict validation
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(command: list[str], timeout: int = 1200) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_pytest_counts(stdout: str) -> dict[str, int | None]:
    passed_match = re.search(r"(\d+)\s+passed", stdout)
    skipped_match = re.search(r"(\d+)\s+skipped", stdout)
    failed_match = re.search(r"(\d+)\s+failed", stdout)
    return {
        "passed": int(passed_match.group(1)) if passed_match else None,
        "skipped": int(skipped_match.group(1)) if skipped_match else None,
        "failed": int(failed_match.group(1)) if failed_match else None,
    }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    body = text.strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        # Fallback: parse from the first JSON object marker.
        start = body.find("{")
        if start < 0:
            return None
        try:
            payload = json.loads(body[start:])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def _tier_pytest_command(tier: str) -> list[str]:
    if tier == "cpu-fast":
        # Coverage threshold set to 0 for fast feedback during development.
        # This tier runs a subset of tests (excluding slow/gpu/opencl) and is
        # optimized for quick iteration. Full coverage is enforced in the
        # canonical tier which runs the complete test suite.
        return [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "not slow and not gpu and not opencl",
            "--cov-fail-under=0",
            "-q",
        ]
    if tier == "canonical":
        return [sys.executable, "-m", "pytest", "-q", "tests/"]
    if tier == "full":
        return [sys.executable, "-m", "pytest", "-q"]
    raise ValueError(f"Unsupported tier: {tier}")


def _evaluate(report: dict[str, Any], *, allow_no_tests: bool) -> dict[str, Any]:
    commands = report["commands"]

    schema_rc = int(commands["validate_results_schema"]["returncode"])
    pytest_rc = int(commands["pytest_tier"]["returncode"])

    pytest_no_tests = pytest_rc == 5 and allow_no_tests
    pytest_pass = pytest_rc == 0 or pytest_no_tests

    checks: dict[str, dict[str, Any]] = {
        "results_schema_green": {
            "observed": schema_rc,
            "required": 0,
            "pass": schema_rc == 0,
        },
        "pytest_tier_green": {
            "observed": pytest_rc,
            "required": "0 or 5(no-tests)" if allow_no_tests else 0,
            "pass": pytest_pass,
            "allow_no_tests": bool(pytest_no_tests),
        },
    }

    if "verify_drivers_smoke" in commands:
        smoke = commands["verify_drivers_smoke"]
        smoke_required_keys = {"overall_status", "opencl", "recommendations"}
        smoke_keys = set(smoke.get("parsed_keys", []))
        smoke_pass = bool(smoke.get("json_parse_ok")) and smoke_required_keys.issubset(smoke_keys)
        checks["verify_drivers_json_smoke"] = {
            "observed": {
                "returncode": int(smoke["returncode"]),
                "json_parse_ok": bool(smoke.get("json_parse_ok")),
                "has_required_keys": smoke_required_keys.issubset(smoke_keys),
                "overall_status": smoke.get("overall_status"),
            },
            "required": {
                "json_parse_ok": True,
                "has_required_keys": True,
            },
            "pass": smoke_pass,
        }

    failed_checks = [name for name, payload in checks.items() if not payload["pass"]]
    if failed_checks:
        decision = "iterate"
        rationale = "One or more validation checks failed; keep rollout blocked until fixed."
    else:
        decision = "promote"
        rationale = "Validation runner checks passed for the selected tier."

    return {
        "checks": checks,
        "failed_checks": failed_checks,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    commands = report["commands"]
    evaluation = report["evaluation"]
    lines: list[str] = []
    lines.append("# Validation Suite Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Branch: `{report['metadata']['branch']}`")
    lines.append(f"- Tier: `{report['metadata']['tier']}`")
    lines.append(f"- Driver smoke enabled: `{report['metadata']['driver_smoke']}`")
    lines.append("")
    lines.append("## Command Status")
    lines.append("")
    lines.append(
        f"- `validate_breakthrough_results.py`: rc={commands['validate_results_schema']['returncode']}"
    )
    lines.append(f"- `pytest` tier command: rc={commands['pytest_tier']['returncode']}")
    pytest_counts = commands["pytest_tier"].get("counts", {})
    if pytest_counts:
        lines.append(
            f"- `pytest` counts: passed={pytest_counts.get('passed')} failed={pytest_counts.get('failed')} skipped={pytest_counts.get('skipped')}"
        )
    if "verify_drivers_smoke" in commands:
        smoke = commands["verify_drivers_smoke"]
        lines.append(f"- `verify_drivers.py --json`: rc={smoke['returncode']}")
        lines.append(f"- Driver JSON parse: `{smoke.get('json_parse_ok')}`")
        lines.append(f"- Driver status: `{smoke.get('overall_status')}`")
    lines.append("")
    lines.append("## Evaluation")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in evaluation["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{evaluation['decision']}`")
    lines.append(f"- Rationale: {evaluation['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_suite(
    *,
    tier: str,
    driver_smoke: bool,
    allow_no_tests: bool,
    skip_pytest: bool,
) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()

    commands: dict[str, dict[str, Any]] = {}
    commands["validate_results_schema"] = _run_command(
        [sys.executable, "scripts/validate_breakthrough_results.py"]
    )

    if skip_pytest:
        commands["pytest_tier"] = {
            "command": ["<skipped>", "--skip-pytest"],
            "returncode": 0,
            "stdout": "pytest execution skipped by --skip-pytest",
            "stderr": "",
            "counts": {"passed": None, "skipped": None, "failed": None},
            "skipped_by_flag": True,
        }
    else:
        pytest_cmd = _tier_pytest_command(tier)
        commands["pytest_tier"] = _run_command(pytest_cmd)
        commands["pytest_tier"]["counts"] = _extract_pytest_counts(
            commands["pytest_tier"]["stdout"]
        )

    if driver_smoke:
        smoke = _run_command([sys.executable, "scripts/verify_drivers.py", "--json"])
        payload = _extract_json_payload(smoke["stdout"])
        smoke["json_parse_ok"] = payload is not None
        smoke["parsed_keys"] = sorted(list(payload.keys())) if payload else []
        smoke["overall_status"] = payload.get("overall_status") if payload else None
        commands["verify_drivers_smoke"] = smoke

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch,
            "tier": tier,
            "driver_smoke": bool(driver_smoke),
        },
        "commands": commands,
    }
    report["evaluation"] = _evaluate(report, allow_no_tests=allow_no_tests)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified validation suite.")
    parser.add_argument(
        "--tier",
        choices=["cpu-fast", "canonical", "full"],
        default="canonical",
        help="Validation tier to execute.",
    )
    parser.add_argument(
        "--driver-smoke",
        action="store_true",
        help="Run verify_drivers JSON smoke validation (parse + contract keys).",
    )
    parser.add_argument(
        "--allow-no-tests",
        action="store_true",
        help="Treat pytest exit code 5 (no tests collected) as pass.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest execution in this runner (useful when another CI step runs pytest).",
    )
    parser.add_argument(
        "--report-dir",
        default="",
        help="Optional report output directory relative to repo root.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON report to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = run_suite(
        tier=args.tier,
        driver_smoke=bool(args.driver_smoke),
        allow_no_tests=bool(args.allow_no_tests),
        skip_pytest=bool(args.skip_pytest),
    )

    if args.report_dir:
        output_dir = (REPO_ROOT / args.report_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"validation_suite_{args.tier}_{ts}.json"
        md_path = output_dir / f"validation_suite_{args.tier}_{ts}.md"
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        md_path.write_text(_markdown(report))
        print(f"Wrote JSON report: {json_path.relative_to(REPO_ROOT)}")
        print(f"Wrote Markdown report: {md_path.relative_to(REPO_ROOT)}")

    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Tier: {report['metadata']['tier']}")
        print(f"Decision: {report['evaluation']['decision']}")
        if report["evaluation"]["failed_checks"]:
            print("Failed checks:")
            for check in report["evaluation"]["failed_checks"]:
                print(f"- {check}")

    return 0 if report["evaluation"]["decision"] == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
