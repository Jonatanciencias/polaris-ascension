#!/usr/bin/env python3
"""Week 17 Block 4: weekly replay post-hardening with drift confirmation."""

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
    lines.append("# Week 17 Block 4 - Post-Hardening Weekly Replay")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append(f"- T5 policy: `{report['metadata']['t5_policy_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Drift")
    lines.append("")
    drift = report["drift"]
    lines.append(
        f"- max_abs_throughput_drift_percent: {drift['max_abs_throughput_drift_percent']:.4f}"
    )
    lines.append(
        f"- max_positive_p95_drift_percent: {drift['max_positive_p95_drift_percent']:.4f}"
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
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week17 Block4 post-hardening weekly replay.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json",
    )
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json",
    )
    parser.add_argument(
        "--baseline-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.json",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=27211)
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week17_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week17_block4_posthardening_replay")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    policy_path = _resolve(args.policy_path)
    t5_policy_path = _resolve(args.t5_policy_path)
    baseline_path = _resolve(args.baseline_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest: dict[str, Any] = {}
    stable_tag = "unknown"
    if stable_manifest_path.exists():
        stable_manifest = _read_json(stable_manifest_path)
        stable_tag = str(stable_manifest.get("stable_tag", "unknown"))

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
    pre_gate_pytest_green = False
    if pre_gate_json:
        pre_payload = _read_json(_resolve(pre_gate_json))
        pre_gate_decision = str(pre_payload.get("evaluation", {}).get("decision", "unknown"))
        pre_gate_pytest_green = bool(
            pre_payload.get("evaluation", {}).get("checks", {}).get("pytest_tier_green", {}).get("pass", False)
        )

    replay_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py",
        "--mode",
        "local",
        "--policy-path",
        str(policy_path),
        "--t5-policy-path",
        str(t5_policy_path),
        "--baseline-path",
        str(baseline_path),
        "--sizes",
        *[str(s) for s in args.sizes],
        "--snapshots",
        str(args.snapshots),
        "--snapshot-interval-minutes",
        "60",
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--validation-report-dir",
        str(args.report_dir),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        f"{args.output_prefix}_automation",
    ]
    replay = _run(replay_cmd)
    replay_json = _extract_prefixed_line(replay["stdout"], "Week12 block1 JSON:")
    replay_md = _extract_prefixed_line(replay["stdout"], "Week12 block1 MD:")
    replay_decision = _extract_prefixed_line(replay["stdout"], "Decision:")

    replay_eval_json = ""
    replay_canary_json = ""
    eval_decision = "unknown"
    drift_max_abs_thr = 999.0
    drift_max_p95 = 999.0
    if replay_json:
        replay_payload = _read_json(_resolve(replay_json))
        replay_eval_json = str(replay_payload.get("artifacts", {}).get("eval_json", ""))
        replay_canary_json = str(replay_payload.get("artifacts", {}).get("canary_json", ""))
        if replay_eval_json:
            eval_payload = _read_json(_resolve(replay_eval_json))
            eval_decision = str(eval_payload.get("evaluation", {}).get("decision", "unknown"))
            checks = eval_payload.get("evaluation", {}).get("checks", {})
            drift_max_abs_thr = float(
                checks.get("throughput_drift_abs_bound", {}).get("observed_max_abs_percent", 999.0)
            )
            drift_max_p95 = float(
                checks.get("p95_drift_bound", {}).get("observed_max_percent", 999.0)
            )

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    post_gate_pytest_green = False
    if post_gate_json:
        post_payload = _read_json(_resolve(post_gate_json))
        post_gate_decision = str(post_payload.get("evaluation", {}).get("decision", "unknown"))
        post_gate_pytest_green = bool(
            post_payload.get("evaluation", {}).get("checks", {}).get("pytest_tier_green", {}).get("pass", False)
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
    checks["replay_automation_promote"] = {
        "observed": str(replay_decision or "unknown").lower(),
        "required": "promote",
        "pass": str(replay_decision or "").lower() == "promote",
    }
    checks["replay_eval_promote"] = {
        "observed": eval_decision,
        "required": "promote",
        "pass": eval_decision == "promote",
    }
    checks["throughput_drift_bound"] = {
        "observed_max_abs_percent": float(drift_max_abs_thr),
        "required_max_abs_percent": 15.0,
        "pass": drift_max_abs_thr <= 15.0,
    }
    checks["p95_drift_bound"] = {
        "observed_max_percent": float(drift_max_p95),
        "required_max_percent": 20.0,
        "pass": drift_max_p95 <= 20.0,
    }
    checks["pre_gate_promote"] = {
        "observed": pre_gate_decision,
        "required": "promote",
        "pass": pre_gate_decision == "promote",
    }
    checks["pre_gate_pytest_tier_green"] = {
        "observed": bool(pre_gate_pytest_green),
        "required": True,
        "pass": bool(pre_gate_pytest_green),
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }
    checks["post_gate_pytest_tier_green"] = {
        "observed": bool(post_gate_pytest_green),
        "required": True,
        "pass": bool(post_gate_pytest_green),
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Post-hardening weekly replay is stable with bounded drift and green pytest-tier gates."
        if decision == "promote"
        else "Post-hardening replay found unresolved issues in drift or validation gates."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "policy_path": str(policy_path),
            "t5_policy_path": str(t5_policy_path),
            "baseline_path": str(baseline_path),
            "sizes": [int(s) for s in args.sizes],
            "snapshots": int(args.snapshots),
            "seed": int(args.seed),
        },
        "commands": {
            "pre_gate": pre_gate,
            "replay_automation": replay,
            "post_gate": post_gate,
            "replay_cmd_pretty": " ".join(shlex.quote(x) for x in replay_cmd),
        },
        "artifacts": {
            "replay_json": replay_json,
            "replay_md": replay_md,
            "replay_eval_json": replay_eval_json,
            "replay_canary_json": replay_canary_json,
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "drift": {
            "max_abs_throughput_drift_percent": float(drift_max_abs_thr),
            "max_positive_p95_drift_percent": float(drift_max_p95),
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

    print(f"Week17 block4 JSON: {json_path}")
    print(f"Week17 block4 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
