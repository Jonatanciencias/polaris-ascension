#!/usr/bin/env python3
"""Week 6 final suite runner for roadmap closure."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.production_kernel_benchmark import run_production_benchmark


def _run_command(command: list[str], timeout: int = 900) -> dict[str, Any]:
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


def _extract_pytest_passed(stdout: str) -> int | None:
    match = re.search(r"(\d+)\s+passed", stdout)
    if not match:
        return None
    return int(match.group(1))


def _extract_production_summary(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    passes = [line for line in lines if "PASS" in line.upper()]
    return {
        "pass_lines": passes,
        "pass_count": len(passes),
    }


def _run_benchmarks(
    *,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    kernels = ["auto", "auto_t3_controlled", "auto_t5_guarded"]
    rows: list[dict[str, Any]] = []
    for idx, kernel in enumerate(kernels):
        run = run_production_benchmark(
            size=size,
            sessions=sessions,
            iterations=iterations,
            kernel=kernel,
            seed=seed + idx * 10_000,
        )
        summary = run["summary"]
        row: dict[str, Any] = {
            "kernel": kernel,
            "size": int(size),
            "sessions": int(sessions),
            "iterations": int(iterations),
            "peak_mean_gflops": float(summary["peak_gflops"]["mean"]),
            "avg_mean_gflops": float(summary["avg_gflops"]["mean"]),
            "p95_time_ms": float(summary["time_ms"]["p95"]),
            "max_error_mean": float(summary["max_error"]["mean"]),
            "max_error_max": float(summary["max_error"]["max"]),
            "platform": str(run["metadata"]["platform"]),
            "device": str(run["metadata"]["device"]),
        }
        if kernel == "auto_t3_controlled":
            row["fallback_rate"] = float(summary["fallback_rate"])
            row["correctness_failure_count"] = int(
                summary.get("correctness_failure_count", summary.get("correctness_failures", 0))
            )
            row["disable_events"] = int(summary.get("disable_events", 0))
            row["policy_disabled"] = bool(summary["policy_disabled"])
            row["delta_vs_static_percent"] = float(summary["delta_vs_static_percent"])
        if kernel == "auto_t5_guarded":
            t5 = summary.get("t5_abft", {})
            row["abft_effective_overhead_percent"] = float(
                t5.get("effective_overhead_percent", 0.0)
            )
            row["abft_false_positive_rate"] = float(t5.get("false_positive_rate", 0.0))
            row["abft_disable_events"] = int(t5.get("disable_events", 0))
            row["abft_state_path"] = str(t5.get("state_path", ""))
        rows.append(row)
    return rows


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    commands = report["commands"]
    benchmarks = report["benchmarks"]

    benchmark_errors_ok = all(float(row["max_error_max"]) <= 1e-3 for row in benchmarks)
    auto_peak = next(row for row in benchmarks if row["kernel"] == "auto")["peak_mean_gflops"]
    auto_min_ok = float(auto_peak) >= 700.0

    t3_row = next(row for row in benchmarks if row["kernel"] == "auto_t3_controlled")
    t3_guard_ok = (
        float(t3_row.get("fallback_rate", 1.0)) <= 0.10
        and int(t3_row.get("correctness_failure_count", 1)) == 0
    )

    t5_row = next(row for row in benchmarks if row["kernel"] == "auto_t5_guarded")
    t5_guard_ok = (
        int(t5_row.get("abft_disable_events", 1)) == 0
        and float(t5_row.get("abft_false_positive_rate", 1.0)) <= 0.05
        and float(t5_row.get("abft_effective_overhead_percent", 999.0)) <= 3.0
    )

    checks = {
        "test_production_system_green": {
            "observed": int(commands["test_production_system"]["returncode"]),
            "required": 0,
            "pass": int(commands["test_production_system"]["returncode"]) == 0,
        },
        "pytest_suite_green": {
            "observed": int(commands["pytest_core"]["returncode"]),
            "required": 0,
            "pass": int(commands["pytest_core"]["returncode"]) == 0,
        },
        "pytest_repo_discovery_non_blocking": {
            "observed": int(commands["pytest_repo_discovery"]["returncode"]),
            "required": "informational",
            "pass": True,
        },
        "results_schema_green": {
            "observed": int(commands["validate_results_schema"]["returncode"]),
            "required": 0,
            "pass": int(commands["validate_results_schema"]["returncode"]) == 0,
        },
        "benchmark_correctness_max_error": {
            "observed": max(float(row["max_error_max"]) for row in benchmarks),
            "required_max": 1e-3,
            "pass": benchmark_errors_ok,
        },
        "auto_peak_mean_gflops": {
            "observed": float(auto_peak),
            "required_min": 700.0,
            "pass": auto_min_ok,
        },
        "t3_guardrails": {
            "observed": {
                "fallback_rate": float(t3_row.get("fallback_rate", 0.0)),
                "correctness_failure_count": int(t3_row.get("correctness_failure_count", 0)),
            },
            "required": {"fallback_rate_max": 0.10, "correctness_failure_count": 0},
            "pass": t3_guard_ok,
        },
        "t5_guardrails": {
            "observed": {
                "abft_disable_events": int(t5_row.get("abft_disable_events", 0)),
                "abft_false_positive_rate": float(t5_row.get("abft_false_positive_rate", 0.0)),
                "abft_effective_overhead_percent": float(
                    t5_row.get("abft_effective_overhead_percent", 0.0)
                ),
            },
            "required": {
                "abft_disable_events": 0,
                "abft_false_positive_rate_max": 0.05,
                "abft_effective_overhead_percent_max": 3.0,
            },
            "pass": t5_guard_ok,
        },
    }

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if failed:
        decision = "iterate"
        rationale = (
            "Final suite found one or more failing closure checks; keep rollout controlled and address failures."
        )
    else:
        decision = "promote"
        rationale = (
            "Final suite is green across functional tests, schema validation and production guardrails."
        )

    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    checks = evaluation["checks"]
    lines: list[str] = []
    lines.append("# Week 6 - Final Validation Suite Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Branch: `{report['metadata']['branch']}`")
    lines.append(f"- Size tested: {report['metadata']['size']}")
    lines.append(
        f"- Sessions/Iterations: {report['metadata']['sessions']}/{report['metadata']['iterations']}"
    )
    lines.append("")
    lines.append("## Command Status")
    lines.append("")
    lines.append(
        f"- `test_production_system.py`: rc={report['commands']['test_production_system']['returncode']}"
    )
    lines.append(f"- `pytest -q tests/`: rc={report['commands']['pytest_core']['returncode']}")
    lines.append(
        f"- `pytest -q` (repo discovery, informational): rc={report['commands']['pytest_repo_discovery']['returncode']}"
    )
    lines.append(
        f"- `validate_breakthrough_results.py`: rc={report['commands']['validate_results_schema']['returncode']}"
    )
    pytest_passed = report["commands"]["pytest_core"].get("passed_count")
    if pytest_passed is not None:
        lines.append(f"- Pytest core passed count: {pytest_passed}")
    lines.append("")
    lines.append("## Production Benchmark Matrix")
    lines.append("")
    lines.append(
        "| Kernel | Peak mean GFLOPS | Avg mean GFLOPS | P95 ms | Max error (max) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["benchmarks"]:
        lines.append(
            f"| {row['kernel']} | {row['peak_mean_gflops']:.3f} | {row['avg_mean_gflops']:.3f} | {row['p95_time_ms']:.3f} | {row['max_error_max']:.7f} |"
        )
    lines.append("")
    lines.append("## Closure Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in checks.items():
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
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()

    command_reports = {
        "test_production_system": _run_command([sys.executable, "test_production_system.py"]),
        "pytest_core": _run_command([sys.executable, "-m", "pytest", "-q", "tests/"]),
        "pytest_repo_discovery": _run_command([sys.executable, "-m", "pytest", "-q"]),
        "validate_results_schema": _run_command(
            [sys.executable, "scripts/validate_breakthrough_results.py"]
        ),
    }
    command_reports["pytest_core"]["passed_count"] = _extract_pytest_passed(
        command_reports["pytest_core"]["stdout"]
    )
    command_reports["test_production_system"]["summary"] = _extract_production_summary(
        command_reports["test_production_system"]["stdout"]
    )

    benchmarks = _run_benchmarks(
        size=size,
        sessions=sessions,
        iterations=iterations,
        seed=seed,
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch,
            "size": int(size),
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "commands": command_reports,
        "benchmarks": benchmarks,
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 6 final validation suite.")
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="research/breakthrough_lab")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_suite(
        size=int(args.size),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week6_final_suite_{ts}.json"
    md_path = output_dir / f"week6_final_suite_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"Week6 final suite JSON: {json_path}")
    print(f"Week6 final suite MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
