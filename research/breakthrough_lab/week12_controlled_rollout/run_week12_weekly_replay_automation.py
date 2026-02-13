#!/usr/bin/env python3
"""Week 12 Block 1: scheduled weekly replay automation (local/CI)."""

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
DEFAULT_POLICY = (
    "research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json"
)
DEFAULT_T5_POLICY = (
    "research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json"
)
DEFAULT_BASELINE = (
    "research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json"
)
DEFAULT_VALIDATION_DIR = "research/breakthrough_lab/week8_validation_discipline"
DEFAULT_OUTPUT_DIR = "research/breakthrough_lab/week12_controlled_rollout"


def _extract_prefixed_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        raw = line.strip()
        if raw.startswith(prefix):
            return raw[len(prefix) :].strip()
    return None


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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _decision_from_validation(path: str) -> str:
    payload = _load_json(path)
    return str(payload.get("evaluation", {}).get("decision", "unknown"))


def _decision_from_eval(path: str) -> str:
    payload = _load_json(path)
    return str(payload.get("evaluation", {}).get("decision", "unknown"))


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 12 Block 1 - Weekly Replay Automation")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Mode: `{report['metadata']['mode']}`")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(f"- Snapshots: {report['metadata']['snapshots']}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append("| Step | Return code | Decision |")
    lines.append("| --- | ---: | --- |")
    lines.append(
        f"| pre_validation | {report['steps']['pre_validation']['returncode']} | {report['steps']['pre_validation'].get('decision')} |"
    )
    lines.append(
        f"| canary_run | {report['steps']['canary_run']['returncode']} | {report['steps']['canary_run'].get('decision')} |"
    )
    lines.append(
        f"| policy_eval | {report['steps']['policy_eval']['returncode']} | {report['steps']['policy_eval'].get('decision')} |"
    )
    lines.append(
        f"| post_validation | {report['steps']['post_validation']['returncode']} | {report['steps']['post_validation'].get('decision')} |"
    )
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
    return "\n".join(lines)


def run() -> int:
    parser = argparse.ArgumentParser(description="Automate weekly replay for local/CI scheduling.")
    parser.add_argument("--mode", choices=["local", "ci"], default="local")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--snapshot-interval-minutes", type=float, default=60.0)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=3)
    parser.add_argument("--pressure-pulses-per-snapshot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12011)
    parser.add_argument("--validation-report-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="week12_block1_weekly_automation")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    pre_validation_cmd = [
        sys.executable,
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.validation_report_dir),
    ]
    pre_validation = _run(pre_validation_cmd)
    pre_validation_report = _extract_prefixed_line(
        pre_validation["stdout"], "Wrote JSON report:"
    )
    if pre_validation["returncode"] != 0 or pre_validation_report is None:
        print("Pre validation failed")
        return 2
    pre_validation_decision = _decision_from_validation(pre_validation_report)

    canary_prefix = f"{args.output_prefix}_canary"
    canary_cmd = [
        sys.executable,
        "research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py",
        "--snapshots",
        str(args.snapshots),
        "--snapshot-interval-minutes",
        str(args.snapshot_interval_minutes),
        "--sleep-between-snapshots-seconds",
        "0",
        "--sizes",
        *[str(s) for s in args.sizes],
        "--kernels",
        "auto_t3_controlled",
        "auto_t5_guarded",
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--pressure-size",
        str(args.pressure_size),
        "--pressure-iterations",
        str(args.pressure_iterations),
        "--pressure-pulses-per-snapshot",
        str(args.pressure_pulses_per_snapshot),
        "--seed",
        str(args.seed),
        "--t5-policy-path",
        str(args.t5_policy_path),
        "--baseline-block6-path",
        str(args.baseline_path),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        canary_prefix,
    ]
    canary_run = _run(canary_cmd)
    canary_json = _extract_prefixed_line(canary_run["stdout"], "Week10 block1 JSON:")
    canary_md = _extract_prefixed_line(canary_run["stdout"], "Week10 block1 MD:")
    canary_decision = _extract_prefixed_line(canary_run["stdout"], "Decision:")
    if canary_run["returncode"] != 0 or canary_json is None:
        print("Canary run failed")
        return 3

    eval_prefix = f"{args.output_prefix}_eval"
    eval_cmd = [
        sys.executable,
        "research/breakthrough_lab/week11_controlled_rollout/evaluate_week11_weekly_replay.py",
        "--canary-path",
        canary_json,
        "--policy-path",
        str(args.policy_path),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        eval_prefix,
    ]
    policy_eval = _run(eval_cmd)
    eval_json = _extract_prefixed_line(policy_eval["stdout"], "Weekly replay eval JSON:")
    eval_md = _extract_prefixed_line(policy_eval["stdout"], "Weekly replay eval MD:")
    eval_decision = _extract_prefixed_line(policy_eval["stdout"], "Decision:")
    if policy_eval["returncode"] != 0 or eval_json is None:
        print("Policy evaluation failed")
        return 4

    post_validation = _run(pre_validation_cmd)
    post_validation_report = _extract_prefixed_line(
        post_validation["stdout"], "Wrote JSON report:"
    )
    if post_validation["returncode"] != 0 or post_validation_report is None:
        print("Post validation failed")
        return 5
    post_validation_decision = _decision_from_validation(post_validation_report)

    failed_checks: list[str] = []
    if pre_validation_decision != "promote":
        failed_checks.append("pre_validation_not_promote")
    if post_validation_decision != "promote":
        failed_checks.append("post_validation_not_promote")
    if eval_decision != "promote":
        failed_checks.append("policy_eval_not_promote")

    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Automated weekly replay completed with promote in canary, policy evaluation, and both canonical gates."
        if decision == "promote"
        else "Automated weekly replay found one or more non-promote steps."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "policy_path": str(args.policy_path),
            "t5_policy_path": str(args.t5_policy_path),
            "baseline_path": str(args.baseline_path),
            "sizes": [int(s) for s in args.sizes],
            "snapshots": int(args.snapshots),
            "seed": int(args.seed),
            "output_dir": str(out_dir),
        },
        "steps": {
            "pre_validation": {
                "returncode": int(pre_validation["returncode"]),
                "decision": pre_validation_decision,
                "command": " ".join(shlex.quote(x) for x in pre_validation_cmd),
                "report_json": pre_validation_report,
            },
            "canary_run": {
                "returncode": int(canary_run["returncode"]),
                "decision": canary_decision,
                "command": " ".join(shlex.quote(x) for x in canary_cmd),
            },
            "policy_eval": {
                "returncode": int(policy_eval["returncode"]),
                "decision": eval_decision,
                "command": " ".join(shlex.quote(x) for x in eval_cmd),
            },
            "post_validation": {
                "returncode": int(post_validation["returncode"]),
                "decision": post_validation_decision,
                "command": " ".join(shlex.quote(x) for x in pre_validation_cmd),
                "report_json": post_validation_report,
            },
        },
        "artifacts": {
            "canary_json": canary_json,
            "canary_md": canary_md,
            "eval_json": eval_json,
            "eval_md": eval_md,
            "pre_validation_json": pre_validation_report,
            "post_validation_json": post_validation_report,
        },
        "evaluation": {
            "decision": decision,
            "failed_checks": failed_checks,
            "rationale": rationale,
        },
    }

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_to_markdown(report) + "\n")

    print(f"Week12 block1 JSON: {json_path}")
    print(f"Week12 block1 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 6


if __name__ == "__main__":
    raise SystemExit(run())
