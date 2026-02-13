#!/usr/bin/env python3
"""Week 14 Block 5: low-scope RX590 dry-run with formal go/no-go."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value if value else None
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _checklist_md(checks: dict[str, dict[str, Any]], final_go: bool) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block5 GO/NO-GO Checklist")
    lines.append("")
    lines.append("## Technical and Operational Gates")
    lines.append("")
    for key, check in checks.items():
        mark = "x" if bool(check.get("pass")) else " "
        lines.append(f"- [{mark}] {key}")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- [{'x' if final_go else ' '}] `GO`")
    lines.append(f"- [{' ' if final_go else 'x'}] `NO-GO`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week14 Block5 RX590 Dry-Run")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Scope: {payload['metadata']['scope']}")
    lines.append(f"- Rollback SLA: `{payload['metadata']['rollback_sla_path']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Canary JSON: `{payload['artifacts'].get('canary_json')}`")
    lines.append(f"- Canary MD: `{payload['artifacts'].get('canary_md')}`")
    lines.append(f"- Pre-gate JSON: `{payload['artifacts'].get('pre_gate_json')}`")
    lines.append(f"- Post-gate JSON: `{payload['artifacts'].get('post_gate_json')}`")
    lines.append(f"- Checklist: `{payload['artifacts'].get('checklist_path')}`")
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
    parser = argparse.ArgumentParser(description="Run Week14 Block5 low-scope dry-run.")
    parser.add_argument("--duration-minutes", type=float, default=3.0)
    parser.add_argument("--snapshot-interval-minutes", type=float, default=1.0)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["auto_t3_controlled", "auto_t5_guarded"],
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses-per-snapshot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17051)
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json",
    )
    parser.add_argument(
        "--baseline-block5-path",
        default="research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json",
    )
    parser.add_argument(
        "--rollback-script",
        default="research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh",
    )
    parser.add_argument(
        "--rollback-sla-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md",
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
        default="research/breakthrough_lab/week14_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week14_block5_rx590_dry_run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preprod_dir = (repo_root / args.preprod_signoff_dir).resolve()
    preprod_dir.mkdir(parents=True, exist_ok=True)
    rollback_sla_path = (repo_root / args.rollback_sla_path).resolve()

    pre_gate_cmd = [
        "./venv/bin/python",
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        args.report_dir,
    ]
    pre_gate = _run(pre_gate_cmd, cwd=repo_root)
    pre_gate_json = _extract_line_value(pre_gate["stdout"], "Wrote JSON report:")

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
    canary = _run(canary_cmd, cwd=repo_root)
    canary_json = _extract_line_value(canary["stdout"], "Week9 block6 JSON:")
    canary_md = _extract_line_value(canary["stdout"], "Week9 block6 MD:")

    rollback_cmd = [args.rollback_script, "dry-run"]
    rollback = _run(rollback_cmd, cwd=repo_root)

    post_gate = _run(pre_gate_cmd, cwd=repo_root)
    post_gate_json = _extract_line_value(post_gate["stdout"], "Wrote JSON report:")

    pre_gate_decision = "unknown"
    if pre_gate_json:
        pre_gate_payload = _read_json((repo_root / pre_gate_json).resolve())
        pre_gate_decision = str(pre_gate_payload.get("evaluation", {}).get("decision", "unknown"))

    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_payload = _read_json((repo_root / post_gate_json).resolve())
        post_gate_decision = str(post_gate_payload.get("evaluation", {}).get("decision", "unknown"))

    canary_decision = "unknown"
    canary_t5_disable_total = -1
    canary_correctness_max = None
    if canary_json:
        canary_payload = _read_json(Path(canary_json))
        canary_eval = canary_payload.get("evaluation", {})
        canary_decision = str(canary_eval.get("decision", "unknown"))
        t5_guard = canary_eval.get("checks", {}).get("t5_guardrails_all_runs", {})
        canary_t5_disable_total = int(t5_guard.get("observed_disable_total", -1))
        corr = canary_eval.get("checks", {}).get("correctness_bound_all_runs", {})
        canary_correctness_max = corr.get("observed_max")

    checks: dict[str, dict[str, Any]] = {}
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

    failed_checks = [name for name, c in checks.items() if not bool(c.get("pass"))]
    go = not failed_checks
    decision = "go" if go else "no-go"
    rationale = (
        "Low-scope RX590 dry-run passed canary, canonical gates, and rollback readiness."
        if go
        else "One or more dry-run gates failed; keep no-go and resolve before expansion."
    )

    checklist_path = preprod_dir / "WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md"
    checklist_path.write_text(_checklist_md(checks, go))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "rx590_low_scope_dry_run",
            "rollback_sla_path": str(rollback_sla_path),
            "params": {
                "duration_minutes": float(args.duration_minutes),
                "snapshot_interval_minutes": float(args.snapshot_interval_minutes),
                "sizes": [int(s) for s in args.sizes],
                "kernels": [str(k) for k in args.kernels],
                "sessions": int(args.sessions),
                "iterations": int(args.iterations),
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
            "canary_t5_disable_total": canary_t5_disable_total,
            "canary_correctness_max": canary_correctness_max,
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

    print(f"Week14 block5 JSON: {json_path}")
    print(f"Week14 block5 MD:   {md_path}")
    print(f"Checklist:          {checklist_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if go else 8


if __name__ == "__main__":
    raise SystemExit(main())
