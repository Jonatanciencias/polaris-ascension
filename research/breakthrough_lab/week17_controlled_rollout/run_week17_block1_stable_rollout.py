#!/usr/bin/env python3
"""Week 17 Block 1: initial controlled rollout for stable v0.15.0."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_line_value(stdout: str, prefix: str) -> str | None:
    for line in stdout.splitlines():
        row = line.strip()
        if row.startswith(prefix):
            value = row[len(prefix) :].strip()
            return value if value else None
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _checklist_md(checks: dict[str, dict[str, Any]], go: bool) -> str:
    lines: list[str] = []
    lines.append("# Week17 Block1 GO/NO-GO Checklist")
    lines.append("")
    lines.append("## Technical and Operational Gates")
    lines.append("")
    for name, check in checks.items():
        mark = "x" if bool(check.get("pass")) else " "
        lines.append(f"- [{mark}] {name}")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- [{'x' if go else ' '}] `GO`")
    lines.append(f"- [{' ' if go else 'x'}] `NO-GO`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week17 Block1 Stable Rollout")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Scope: {payload['metadata']['scope']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append(f"- Snapshots target: {payload['metadata']['params']['target_snapshots']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
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
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week17 Block1 stable controlled rollout.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--stable-runbook-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_RELEASE_RUNBOOK.md",
    )
    parser.add_argument(
        "--rollback-sla-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md",
    )
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    parser.add_argument("--snapshot-interval-minutes", type=float, default=1.0)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["auto_t3_controlled", "auto_t5_guarded"],
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=3)
    parser.add_argument("--pressure-pulses-per-snapshot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=27011)
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json",
    )
    parser.add_argument(
        "--baseline-block5-path",
        default="research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_canary_20260210_011736.json",
    )
    parser.add_argument(
        "--rollback-script",
        default="research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh",
    )
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
        default="research/breakthrough_lab/week17_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week17_block1_stable_rollout")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preprod_dir = (REPO_ROOT / args.preprod_signoff_dir).resolve()
    preprod_dir.mkdir(parents=True, exist_ok=True)
    stable_manifest_path = (REPO_ROOT / args.stable_manifest_path).resolve()
    stable_runbook_path = (REPO_ROOT / args.stable_runbook_path).resolve()
    rollback_sla_path = (REPO_ROOT / args.rollback_sla_path).resolve()

    stable_manifest = {}
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
        args.report_dir,
    ]
    pre_gate = _run(pre_gate_cmd, cwd=REPO_ROOT)
    pre_gate_json = _extract_line_value(pre_gate["stdout"], "Wrote JSON report:")
    pre_gate_decision = "unknown"
    if pre_gate_json:
        pre_gate_decision = str(
            _read_json((REPO_ROOT / pre_gate_json).resolve()).get("evaluation", {}).get("decision", "unknown")
        )

    canary_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/platform_compatibility/run_week9_block6_wallclock_canary.py",
        "--duration-minutes",
        str(args.duration_minutes),
        "--snapshot-interval-minutes",
        str(args.snapshot_interval_minutes),
        "--sizes",
        *[str(s) for s in args.sizes],
        "--kernels",
        *[str(k) for k in args.kernels],
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
        args.t5_policy_path,
        "--baseline-block5-path",
        args.baseline_block5_path,
        "--output-dir",
        args.output_dir,
        "--output-prefix",
        f"{args.output_prefix}_canary",
    ]
    canary = _run(canary_cmd, cwd=REPO_ROOT)
    canary_json = _extract_line_value(canary["stdout"], "Week9 block6 JSON:")
    canary_md = _extract_line_value(canary["stdout"], "Week9 block6 MD:")

    rollback = _run([args.rollback_script, "dry-run"], cwd=REPO_ROOT)

    post_gate = _run(pre_gate_cmd, cwd=REPO_ROOT)
    post_gate_json = _extract_line_value(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json((REPO_ROOT / post_gate_json).resolve()).get("evaluation", {}).get("decision", "unknown")
        )

    canary_decision = "unknown"
    canary_t5_disable_total = -1
    canary_correctness_max = None
    executed_snapshots = -1
    if canary_json:
        canary_payload = _read_json(Path(canary_json))
        canary_eval = canary_payload.get("evaluation", {})
        canary_decision = str(canary_eval.get("decision", "unknown"))
        t5_guard = canary_eval.get("checks", {}).get("t5_guardrails_all_runs", {})
        canary_t5_disable_total = int(t5_guard.get("observed_disable_total", -1))
        corr = canary_eval.get("checks", {}).get("correctness_bound_all_runs", {})
        canary_correctness_max = corr.get("observed_max")
        snapshots_list = canary_payload.get("metadata", {}).get("snapshots", [])
        if isinstance(snapshots_list, list):
            executed_snapshots = len(snapshots_list)
        else:
            executed_snapshots = int(canary_payload.get("metadata", {}).get("executed_snapshots", -1))

    target_snapshots = int(args.duration_minutes // args.snapshot_interval_minutes)

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
    checks["stable_runbook_exists"] = {
        "observed": stable_runbook_path.exists(),
        "required": True,
        "pass": stable_runbook_path.exists(),
    }
    checks["pre_gate_promote"] = {
        "observed": pre_gate_decision,
        "required": "promote",
        "pass": pre_gate_decision == "promote",
    }
    checks["canary_returncode_zero"] = {
        "observed": int(canary["returncode"]),
        "required": 0,
        "pass": int(canary["returncode"]) == 0,
    }
    checks["canary_promote"] = {
        "observed": canary_decision,
        "required": "promote",
        "pass": canary_decision == "promote",
    }
    checks["canary_t5_disable_zero"] = {
        "observed": int(canary_t5_disable_total),
        "required": 0,
        "pass": canary_t5_disable_total == 0,
    }
    checks["extended_snapshots_reached"] = {
        "observed": int(executed_snapshots),
        "required_min": target_snapshots,
        "pass": executed_snapshots >= target_snapshots,
    }
    checks["rollback_dry_run_ok"] = {
        "observed": int(rollback["returncode"]),
        "required": 0,
        "pass": int(rollback["returncode"]) == 0,
    }
    checks["rollback_sla_exists"] = {
        "observed": rollback_sla_path.exists(),
        "required": True,
        "pass": rollback_sla_path.exists(),
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    go = not failed_checks
    decision = "go" if go else "no-go"
    rationale = (
        "Initial stable rollout reached extended horizon with green gates and rollback readiness."
        if go
        else "One or more stable rollout gates failed; keep no-go and iterate."
    )

    checklist_path = preprod_dir / "WEEK17_BLOCK1_GO_NO_GO_CHECKLIST.md"
    checklist_path.write_text(_checklist_md(checks, go))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "week17_block1_initial_stable_rollout",
            "stable_tag": stable_tag,
            "rollback_sla_path": str(rollback_sla_path),
            "stable_manifest_path": str(stable_manifest_path),
            "params": {
                "duration_minutes": float(args.duration_minutes),
                "snapshot_interval_minutes": float(args.snapshot_interval_minutes),
                "target_snapshots": int(target_snapshots),
                "sizes": [int(s) for s in args.sizes],
                "kernels": [str(k) for k in args.kernels],
                "sessions": int(args.sessions),
                "iterations": int(args.iterations),
                "seed": int(args.seed),
            },
        },
        "commands": {
            "pre_gate": pre_gate,
            "canary": canary,
            "rollback_dry_run": rollback,
            "post_gate": post_gate,
        },
        "artifacts": {
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
            "canary_json": canary_json,
            "canary_md": canary_md,
            "checklist_path": str(checklist_path),
        },
        "summary": {
            "pre_gate_decision": pre_gate_decision,
            "canary_decision": canary_decision,
            "canary_t5_disable_total": int(canary_t5_disable_total),
            "canary_correctness_max": canary_correctness_max,
            "executed_snapshots": int(executed_snapshots),
            "post_gate_decision": post_gate_decision,
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
    _write_json(json_path, payload)
    md_path.write_text(_report_md(payload))

    print(f"Week17 block1 JSON: {json_path}")
    print(f"Week17 block1 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if go else 8


if __name__ == "__main__":
    raise SystemExit(main())
