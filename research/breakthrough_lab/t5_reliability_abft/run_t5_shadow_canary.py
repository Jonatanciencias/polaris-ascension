#!/usr/bin/env python3
"""
Week 4 T5 shadow canary integration runner.

Runs ABFT-lite in deterministic shadow mode using the block3 hardening policy,
evaluates operational guardrails, and emits a formal canary decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_t5_abft_detect_only import run_experiment


def _parse_mode_to_period(mode_label: str) -> int:
    if mode_label == "always":
        return 1
    if mode_label.startswith("periodic_"):
        return int(mode_label.split("_", maxsplit=1)[1])
    raise ValueError(f"Unsupported mode label in policy: {mode_label}")


def _load_policy(policy_path: Path) -> dict[str, Any]:
    data = json.loads(policy_path.read_text())
    required = ["policy_id", "scope", "abft_mode", "runtime_guardrails", "stress_evidence"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Policy missing required fields: {missing}")
    return data


def _guardrail_checks(
    *,
    mode: dict[str, Any],
    guardrails: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "false_positive_rate": {
            "observed": float(mode["false_positive_rate"]),
            "threshold": float(guardrails["disable_if_false_positive_rate_gt"]),
            "comparator": "<=",
        },
        "effective_overhead_percent": {
            "observed": float(mode["effective_overhead_percent"]),
            "threshold": float(guardrails["disable_if_effective_overhead_percent_gt"]),
            "comparator": "<=",
        },
        "correctness_error": {
            "observed": float(mode["max_error"]),
            "threshold": float(guardrails["disable_if_correctness_error_gt"]),
            "comparator": "<=",
        },
        "uniform_recall": {
            "observed": float(mode["uniform_recall"]),
            "threshold": float(guardrails["disable_if_uniform_recall_lt"]),
            "comparator": ">=",
        },
        "critical_recall": {
            "observed": float(mode["critical_recall"]),
            "threshold": float(guardrails["disable_if_critical_recall_lt"]),
            "comparator": ">=",
        },
    }
    for entry in checks.values():
        if entry["comparator"] == "<=":
            entry["pass"] = bool(entry["observed"] <= entry["threshold"])
        else:
            entry["pass"] = bool(entry["observed"] >= entry["threshold"])

    failures = [name for name, entry in checks.items() if not entry["pass"]]
    return {
        "checks": checks,
        "all_passed": len(failures) == 0,
        "failed_guardrails": failures,
        "disable_signal": len(failures) > 0,
        "fallback_action": (
            "auto_disable_abft_shadow_path"
            if failures
            else "continue_shadow_canary_ready_for_gate_review"
        ),
    }


def _decision_from_eval(guardrail_eval: dict[str, Any], mode: dict[str, Any]) -> tuple[str, str]:
    if guardrail_eval["all_passed"]:
        return (
            "promote",
            "Shadow canary satisfies all guardrails with deterministic evidence; track is ready for promotion gate review.",
        )
    if bool(mode.get("stop_rule_triggered", False)):
        return (
            "drop",
            "Shadow canary violates guardrails and triggers stop rule; ABFT mode should not proceed.",
        )
    return (
        "iterate",
        "Shadow canary found guardrail violations; keep fallback active and refine ABFT policy before promotion.",
    )


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T5 Week 4 Block 4 - Shadow Canary Integration Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['policy']['policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']} | Warmup={report['metadata']['warmup']}"
    )
    lines.append(
        f"- Shadow sampling period: `{report['metadata']['sampling_period']}` ({report['metadata']['sampling_mode']})"
    )
    lines.append("")
    lines.append("## Canary Metrics")
    lines.append("")
    lines.append(f"- Effective overhead: {report['metrics']['effective_overhead_percent']:.3f}%")
    lines.append(f"- False positive rate: {report['metrics']['false_positive_rate']:.3f}")
    lines.append(f"- Critical recall: {report['metrics']['critical_recall']:.3f}")
    lines.append(f"- Uniform recall: {report['metrics']['uniform_recall']:.3f}")
    lines.append(f"- Critical misses: {report['metrics']['critical_misses']}")
    lines.append(f"- Max correctness error: {report['metrics']['max_error']:.7f}")
    lines.append("")
    lines.append("## Guardrail Evaluation")
    lines.append("")
    lines.append("| Guardrail | Observed | Threshold | Comparator | Pass |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for name, entry in report["guardrails"]["checks"].items():
        lines.append(
            f"| {name} | {entry['observed']:.6f} | {entry['threshold']:.6f} | {entry['comparator']} | {entry['pass']} |"
        )
    lines.append("")
    lines.append(f"- All guardrails passed: {report['guardrails']['all_passed']}")
    lines.append(f"- Disable signal: {report['guardrails']['disable_signal']}")
    lines.append(f"- Fallback action: `{report['guardrails']['fallback_action']}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_shadow_canary(
    *,
    policy_path: Path,
    sessions: int,
    iterations: int,
    warmup: int,
    seed: int,
    sampling_period_override: int | None,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    scope_sizes = [int(x) for x in policy["scope"]["sizes"]]
    abft_mode = policy["abft_mode"]
    guardrails = policy["runtime_guardrails"]

    recommended_mode = policy["stress_evidence"]["recommended_mode"]
    sampling_period = (
        int(sampling_period_override)
        if sampling_period_override is not None
        else _parse_mode_to_period(str(recommended_mode))
    )

    base_report = run_experiment(
        sizes=scope_sizes,
        sessions=sessions,
        iterations=iterations,
        warmup=warmup,
        sampling_periods=[sampling_period],
        row_samples=int(abft_mode["row_samples"]),
        col_samples=int(abft_mode["col_samples"]),
        faults_per_matrix=int(abft_mode["faults_per_matrix_modelled"]),
        fault_scale=0.05,
        fault_abs_min=1.0,
        residual_scale=float(abft_mode["residual_scale"]),
        residual_margin=float(abft_mode["residual_margin"]),
        residual_floor=float(abft_mode["residual_floor"]),
        projection_count=int(abft_mode["projection_count"]),
        correctness_threshold=float(guardrails["disable_if_correctness_error_gt"]),
        seed=seed,
    )

    mode = base_report["summary"]["recommended_mode_details"]
    guardrail_eval = _guardrail_checks(mode=mode, guardrails=guardrails)
    decision, rationale = _decision_from_eval(guardrail_eval, mode)

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_path": str(policy_path),
            "sizes": scope_sizes,
            "sessions": sessions,
            "iterations": iterations,
            "warmup": warmup,
            "sampling_period": sampling_period,
            "sampling_mode": mode["mode"],
            "seed": seed,
            "opencl_platform": base_report["metadata"]["opencl_platform"],
            "opencl_device": base_report["metadata"]["opencl_device"],
        },
        "policy": {
            "policy_id": policy["policy_id"],
            "status": policy.get("status"),
            "next_gate": policy.get("next_gate"),
            "runtime_guardrails": guardrails,
        },
        "metrics": {
            "effective_overhead_percent": float(mode["effective_overhead_percent"]),
            "false_positive_rate": float(mode["false_positive_rate"]),
            "critical_recall": float(mode["critical_recall"]),
            "uniform_recall": float(mode["uniform_recall"]),
            "critical_misses": int(mode["critical_misses"]),
            "max_error": float(mode["max_error"]),
            "sampling_coverage": float(mode["sampling_coverage"]),
            "kernel_gflops_mean": float(mode["kernel_gflops"]["mean"]),
            "effective_gflops_mean": float(mode["effective_gflops"]["mean"]),
            "kernel_time_p95_ms": float(mode["kernel_time_ms"]["p95"]),
        },
        "guardrails": guardrail_eval,
        "decision": {
            "decision": decision,
            "rationale": rationale,
        },
        "source_report": base_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T5 shadow canary integration campaign")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json",
    )
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-period", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t5_reliability_abft",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    policy_path = (repo_root / args.policy_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_shadow_canary(
        policy_path=policy_path,
        sessions=args.sessions,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        sampling_period_override=args.sampling_period,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week4_t5_shadow_canary_{timestamp}.json"
    md_path = output_dir / f"week4_t5_shadow_canary_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T5 shadow canary JSON: {json_path}")
    print(f"T5 shadow canary MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(f"Guardrails passed: {report['guardrails']['all_passed']}")
    print(f"Disable signal: {report['guardrails']['disable_signal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
