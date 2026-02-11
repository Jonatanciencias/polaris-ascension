#!/usr/bin/env python3
"""Week 18 Block 2: maintenance canary split Clover/rusticl on stable baseline."""

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


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 2 - Stable Maintenance Split Canary")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append(f"- T5 policy: `{report['metadata']['t5_policy_path']}`")
    lines.append(f"- Baseline: `{report['metadata']['baseline_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Split Metrics")
    lines.append("")
    lines.append(
        f"- rusticl_ratio_min: {float(report['split_metrics']['rusticl_ratio_min']):.6f}"
    )
    lines.append(
        f"- canary_t5_overhead_max: {float(report['split_metrics']['canary_t5_overhead_max']):.6f}"
    )
    lines.append(
        f"- canary_t5_disable_total: {int(report['split_metrics']['canary_t5_disable_total'])}"
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
    parser = argparse.ArgumentParser(description="Run Week18 Block2 stable maintenance split canary.")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[181, 313])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["auto_t3_controlled", "auto_t5_guarded"],
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses-per-seed", type=int, default=2)
    parser.add_argument("--min-rusticl-ratio", type=float, default=0.85)
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week18_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week18_block2_maintenance_split")
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
            pre_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
        )

    canary_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py",
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--sizes",
        *[str(size) for size in args.sizes],
        "--kernels",
        *[str(kernel) for kernel in args.kernels],
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--pressure-size",
        str(args.pressure_size),
        "--pressure-iterations",
        str(args.pressure_iterations),
        "--pressure-pulses-per-seed",
        str(args.pressure_pulses_per_seed),
        "--t5-policy-path",
        str(t5_policy_path),
        "--baseline-block3-path",
        str(baseline_path),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_canary",
    ]
    canary = _run(canary_cmd)
    canary_json = _extract_prefixed_line(canary["stdout"], "Week9 block4 JSON:")
    canary_md = _extract_prefixed_line(canary["stdout"], "Week9 block4 MD:")
    canary_decision = str(_extract_prefixed_line(canary["stdout"], "Decision:") or "unknown")

    canary_split_present = False
    canary_t5_guardrails = False
    canary_no_regression = False
    canary_t5_overhead_max = 999.0
    canary_t5_disable_total = 999
    if canary_json:
        canary_payload = _read_json(_resolve(canary_json))
        canary_checks = canary_payload.get("evaluation", {}).get("checks", {})
        canary_split_present = bool(
            canary_checks.get("platform_split_clover_and_rusticl", {}).get("pass", False)
        )
        canary_t5_guardrails = bool(
            canary_checks.get("t5_guardrails_all_runs", {}).get("pass", False)
        )
        canary_no_regression = bool(
            canary_checks.get("no_regression_vs_block3_clover", {}).get("pass", False)
        )
        canary_t5_overhead_max = float(
            canary_checks.get("t5_guardrails_all_runs", {}).get("observed_overhead_max", 999.0)
        )
        canary_t5_disable_total = int(
            canary_checks.get("t5_guardrails_all_runs", {}).get("observed_disable_total", 999)
        )

    split_eval_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py",
        "--split-artifact",
        str(canary_json or ""),
        "--policy-path",
        str(policy_path),
        "--min-rusticl-ratio",
        str(args.min_rusticl_ratio),
        "--required-sizes",
        *[str(size) for size in args.sizes],
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_eval",
    ]
    split_eval = _run(split_eval_cmd) if canary_json else {
        "command": split_eval_cmd,
        "returncode": 2,
        "stdout": "",
        "stderr": "canary_json_missing",
    }
    split_eval_json = _extract_prefixed_line(split_eval["stdout"], "Week12 block2 eval JSON:")
    split_eval_md = _extract_prefixed_line(split_eval["stdout"], "Week12 block2 eval MD:")
    split_eval_decision = str(_extract_prefixed_line(split_eval["stdout"], "Decision:") or "unknown")

    split_required_sizes_ok = False
    split_ratio_ok = False
    split_ratio_min = 0.0
    if split_eval_json:
        split_eval_payload = _read_json(_resolve(split_eval_json))
        split_checks = split_eval_payload.get("evaluation", {}).get("checks", {})
        split_required_sizes_ok = bool(
            split_checks.get("required_sizes_present_on_split", {}).get("pass", False)
        )
        split_ratio_row = split_checks.get("rusticl_ratio_vs_clover", {}).get("rows", [])
        ratios = [float(row.get("ratio_rusticl_vs_clover", 0.0)) for row in split_ratio_row]
        split_ratio_min = float(min(ratios)) if ratios else 0.0
        split_ratio_ok = bool(split_checks.get("rusticl_ratio_vs_clover", {}).get("pass", False))

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    post_gate_pytest_green = False
    if post_gate_json:
        post_payload = _read_json(_resolve(post_gate_json))
        post_gate_decision = str(post_payload.get("evaluation", {}).get("decision", "unknown"))
        post_gate_pytest_green = bool(
            post_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
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
    checks["canary_promote"] = {
        "observed": canary_decision.lower(),
        "required": "promote",
        "pass": canary_decision.lower() == "promote",
    }
    checks["canary_split_present"] = {
        "observed": canary_split_present,
        "required": True,
        "pass": canary_split_present,
    }
    checks["canary_t5_guardrails"] = {
        "observed": canary_t5_guardrails,
        "required": True,
        "pass": canary_t5_guardrails,
    }
    checks["canary_no_regression_vs_baseline"] = {
        "observed": canary_no_regression,
        "required": True,
        "pass": canary_no_regression,
    }
    checks["split_eval_promote"] = {
        "observed": split_eval_decision.lower(),
        "required": "promote",
        "pass": split_eval_decision.lower() == "promote",
    }
    checks["split_required_sizes_present"] = {
        "observed": split_required_sizes_ok,
        "required": True,
        "pass": split_required_sizes_ok,
    }
    checks["split_ratio_meets_floor"] = {
        "observed_min": float(split_ratio_min),
        "required_min": float(args.min_rusticl_ratio),
        "pass": split_ratio_ok and float(split_ratio_min) >= float(args.min_rusticl_ratio),
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
        "Stable split maintenance canary passed policy, guardrails, and mandatory canonical gates."
        if decision == "promote"
        else "Stable split maintenance canary failed one or more guardrails/policy/gate checks."
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
            "seeds": [int(seed) for seed in args.seeds],
            "sizes": [int(size) for size in args.sizes],
            "kernels": [str(kernel) for kernel in args.kernels],
            "min_rusticl_ratio": float(args.min_rusticl_ratio),
        },
        "commands": {
            "pre_gate": pre_gate,
            "canary": canary,
            "split_eval": split_eval,
            "post_gate": post_gate,
            "canary_cmd_pretty": " ".join(shlex.quote(x) for x in canary_cmd),
            "split_eval_cmd_pretty": " ".join(shlex.quote(x) for x in split_eval_cmd),
        },
        "artifacts": {
            "canary_json": canary_json,
            "canary_md": canary_md,
            "split_eval_json": split_eval_json,
            "split_eval_md": split_eval_md,
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "split_metrics": {
            "rusticl_ratio_min": float(split_ratio_min),
            "canary_t5_overhead_max": float(canary_t5_overhead_max),
            "canary_t5_disable_total": int(canary_t5_disable_total),
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

    print(f"Week18 block2 JSON: {json_path}")
    print(f"Week18 block2 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
