#!/usr/bin/env python3
"""Week 8 Block 6: integrated consolidation rerun (T3 + T4 + T5 + week6 suite)."""

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


def _run_command(command: list[str], timeout: int = 3600) -> dict[str, Any]:
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


def _extract_json_path(stdout: str, label: str) -> str | None:
    # Example line: "T5 maturation JSON: /path/file.json"
    pattern = re.compile(rf"{re.escape(label)}:\s*(.+\.json)")
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip()
    return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    steps = report["steps"]
    summary = report["summary"]

    checks = {
        "week6_suite_command_green": {
            "observed": int(steps["week6_suite"]["returncode"]),
            "required": 0,
            "pass": int(steps["week6_suite"]["returncode"]) == 0,
        },
        "t3_drift_command_green": {
            "observed": int(steps["t3_drift"]["returncode"]),
            "required": 0,
            "pass": int(steps["t3_drift"]["returncode"]) == 0,
        },
        "t4_mixed_command_green": {
            "observed": int(steps["t4_mixed"]["returncode"]),
            "required": 0,
            "pass": int(steps["t4_mixed"]["returncode"]) == 0,
        },
        "t5_maturation_command_green": {
            "observed": int(steps["t5_maturation"]["returncode"]),
            "required": 0,
            "pass": int(steps["t5_maturation"]["returncode"]) == 0,
        },
        "week6_decision_promote": {
            "observed": summary["week6_decision"],
            "required": "promote",
            "pass": summary["week6_decision"] == "promote",
        },
        "t3_decision_promote": {
            "observed": summary["t3_decision"],
            "required": "promote",
            "pass": summary["t3_decision"] == "promote",
        },
        "t4_decision_promote": {
            "observed": summary["t4_decision"],
            "required": "promote",
            "pass": summary["t4_decision"] == "promote",
        },
        "t5_decision_promote": {
            "observed": summary["t5_decision"],
            "required": "promote",
            "pass": summary["t5_decision"] == "promote",
        },
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if failed:
        decision = "iterate"
        rationale = (
            "Integrated consolidation rerun found one or more failing checks; keep controlled rollout."
        )
    else:
        decision = "promote"
        rationale = (
            "Integrated rerun remains green across week6 suite and promoted T3/T4/T5 tracks."
        )
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    evaluation = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 8 Block 6 - Integrated Consolidation Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Branch: `{report['metadata']['branch']}`")
    lines.append("")
    lines.append("## Decision Snapshot")
    lines.append("")
    lines.append(f"- Week6 suite decision: `{summary['week6_decision']}`")
    lines.append(f"- T3 decision: `{summary['t3_decision']}`")
    lines.append(f"- T4 decision: `{summary['t4_decision']}`")
    lines.append(f"- T5 decision: `{summary['t5_decision']}`")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(
        f"- Week6 auto peak mean GFLOPS: {summary['week6_auto_peak_mean_gflops']:.3f}"
    )
    lines.append(
        f"- T3 warm+pressure delta vs auto: {summary['t3_delta_vs_auto_pressure_percent']:+.3f}%"
    )
    lines.append(
        f"- T4 fallback reduction vs baseline: {summary['t4_fallback_reduction_abs']:.3f}"
    )
    lines.append(
        f"- T5 uniform recall delta vs baseline: {summary['t5_uniform_recall_delta']:+.3f}"
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in evaluation["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- Decision: `{evaluation['decision']}`")
    lines.append(f"- Rationale: {evaluation['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(*, output_dir: Path) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()

    week6_cmd = [
        sys.executable,
        "research/breakthrough_lab/run_week6_final_suite.py",
        "--size",
        "1400",
        "--sessions",
        "5",
        "--iterations",
        "10",
        "--seed",
        "42",
    ]
    t3_cmd = [
        sys.executable,
        "research/breakthrough_lab/t3_online_control/run_week8_t3_drift_campaign.py",
        "--sessions",
        "2",
        "--iterations",
        "6",
        "--sizes",
        "1400",
        "1536",
        "2048",
        "--seed",
        "42",
        "--pressure-size",
        "896",
        "--pressure-iterations",
        "3",
        "--pressure-pulses",
        "2",
        "--pressure-pause-ms",
        "20",
    ]
    t4_cmd = [
        sys.executable,
        "research/breakthrough_lab/t4_approximate_gemm/run_week8_t4_mixed_campaign.py",
        "--sessions",
        "6",
        "--seed",
        "42",
    ]
    t5_cmd = [
        sys.executable,
        "research/breakthrough_lab/t5_reliability_abft/run_week8_t5_maturation.py",
    ]

    steps = {
        "week6_suite": _run_command(week6_cmd),
        "t3_drift": _run_command(t3_cmd),
        "t4_mixed": _run_command(t4_cmd),
        "t5_maturation": _run_command(t5_cmd),
    }

    week6_json_path = _extract_json_path(
        steps["week6_suite"]["stdout"], "Week6 final suite JSON"
    )
    t3_json_path = _extract_json_path(steps["t3_drift"]["stdout"], "T3 drift JSON")
    t4_json_path = _extract_json_path(steps["t4_mixed"]["stdout"], "T4 mixed JSON")
    t5_json_path = _extract_json_path(steps["t5_maturation"]["stdout"], "T5 maturation JSON")

    week6 = _load_json(Path(week6_json_path) if week6_json_path else None)
    t3 = _load_json(Path(t3_json_path) if t3_json_path else None)
    t4 = _load_json(Path(t4_json_path) if t4_json_path else None)
    t5 = _load_json(Path(t5_json_path) if t5_json_path else None)

    summary = {
        "week6_decision": week6["evaluation"]["decision"] if week6 else "missing",
        "t3_decision": t3["decision"]["decision"] if t3 else "missing",
        "t4_decision": t4["decision"]["decision"] if t4 else "missing",
        "t5_decision": t5["decision"]["decision"] if t5 else "missing",
        "week6_auto_peak_mean_gflops": float(
            next(row for row in (week6["benchmarks"] if week6 else []) if row["kernel"] == "auto")[
                "peak_mean_gflops"
            ]
        )
        if week6
        else 0.0,
        "t3_delta_vs_auto_pressure_percent": float(
            t3["scenario_summary"]["warm_queue_pressure"]["comparison"]["t3_delta_vs_auto_percent"]
        )
        if t3
        else 0.0,
        "t4_fallback_reduction_abs": float(t4["evaluation"]["fallback_reduction_abs"]) if t4 else 0.0,
        "t5_uniform_recall_delta": float(
            t5["evaluation"]["uniform_recall_delta_vs_baseline"]
        )
        if t5
        else 0.0,
    }

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch,
            "artifacts": {
                "week6_json": week6_json_path,
                "t3_json": t3_json_path,
                "t4_json": t4_json_path,
                "t5_json": t5_json_path,
            },
        },
        "steps": steps,
        "summary": summary,
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week8 Block6 integrated consolidation.")
    parser.add_argument("--output-dir", default="research/breakthrough_lab")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_campaign(output_dir=output_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_block6_integrated_consolidation_{timestamp}.json"
    md_path = output_dir / f"week8_block6_integrated_consolidation_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week8 block6 JSON: {json_path}")
    print(f"Week8 block6 MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
