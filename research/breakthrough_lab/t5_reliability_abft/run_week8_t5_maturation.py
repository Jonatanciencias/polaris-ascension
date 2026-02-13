#!/usr/bin/env python3
"""Week 8 Block 5 - T5 reliability maturation campaign.

Compares baseline ABFT policy vs tuned candidate policy using deterministic
fault-injection replay to optimize recall/overhead tradeoff.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_t5_abft_detect_only import run_experiment


def _load_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = [
        "policy_id",
        "scope",
        "abft_mode",
        "runtime_guardrails",
    ]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"T5 policy missing required fields: {missing}")
    return data


def _mode_summary(mode: dict[str, Any]) -> dict[str, float]:
    return {
        "sampling_coverage": float(mode["sampling_coverage"]),
        "effective_overhead_percent": float(mode["effective_overhead_percent"]),
        "critical_recall": float(mode["critical_recall"]),
        "uniform_recall": float(mode["uniform_recall"]),
        "critical_misses": float(mode["critical_misses"]),
        "false_positive_rate": float(mode["false_positive_rate"]),
        "max_error": float(mode["max_error"]),
        "kernel_gflops_mean": float(mode["kernel_gflops"]["mean"]),
        "effective_gflops_mean": float(mode["effective_gflops"]["mean"]),
        "effective_gflops_std": float(mode["effective_gflops"]["std"]),
        "effective_gflops_max": float(mode["effective_gflops"]["max"]),
        "kernel_time_p95_ms": float(mode["kernel_time_ms"]["p95"]),
    }


def _single_mode(report: dict[str, Any]) -> dict[str, Any]:
    if len(report["modes"]) != 1:
        raise ValueError("Expected a single sampling mode report for maturation comparison.")
    return report["modes"][0]


def _evaluate(
    *,
    baseline_mode: dict[str, Any],
    candidate_mode: dict[str, Any],
    guardrails: dict[str, float],
) -> dict[str, Any]:
    baseline = _mode_summary(baseline_mode)
    candidate = _mode_summary(candidate_mode)

    uniform_delta = float(candidate["uniform_recall"] - baseline["uniform_recall"])
    overhead_delta = float(
        candidate["effective_overhead_percent"] - baseline["effective_overhead_percent"]
    )

    checks = {
        "candidate_correctness": {
            "observed": candidate["max_error"],
            "threshold_max": float(guardrails["disable_if_correctness_error_gt"]),
            "comparator": "<=",
        },
        "candidate_false_positive_rate": {
            "observed": candidate["false_positive_rate"],
            "threshold_max": float(guardrails["disable_if_false_positive_rate_gt"]),
            "comparator": "<=",
        },
        "candidate_effective_overhead_percent": {
            "observed": candidate["effective_overhead_percent"],
            "threshold_max": float(guardrails["disable_if_effective_overhead_percent_gt"]),
            "comparator": "<=",
        },
        "candidate_critical_recall": {
            "observed": candidate["critical_recall"],
            "threshold_min": float(guardrails["disable_if_critical_recall_lt"]),
            "comparator": ">=",
        },
        "candidate_uniform_recall": {
            "observed": candidate["uniform_recall"],
            "threshold_min": float(guardrails["disable_if_uniform_recall_lt"]),
            "comparator": ">=",
        },
        "uniform_recall_delta_vs_baseline": {
            "observed": uniform_delta,
            "threshold_min": float(guardrails["require_uniform_recall_delta_vs_baseline_ge"]),
            "comparator": ">=",
        },
        "overhead_delta_vs_baseline": {
            "observed": overhead_delta,
            "threshold_max": float(guardrails["max_overhead_delta_vs_baseline_percent"]),
            "comparator": "<=",
        },
    }

    for payload in checks.values():
        if payload["comparator"] == "<=":
            payload["pass"] = bool(payload["observed"] <= payload["threshold_max"])
        else:
            payload["pass"] = bool(payload["observed"] >= payload["threshold_min"])

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "uniform_recall_delta_vs_baseline": uniform_delta,
        "overhead_delta_vs_baseline_percent": overhead_delta,
        "checks": checks,
        "failed_checks": failed,
        "all_passed": len(failed) == 0,
    }


def _decision(report: dict[str, Any]) -> tuple[str, str]:
    checks = report["evaluation"]["checks"]
    failed = report["evaluation"]["failed_checks"]
    if not checks["candidate_correctness"]["pass"]:
        return (
            "drop",
            "Correctness guard failed under fault-injection replay; reject candidate policy.",
        )
    if not checks["candidate_critical_recall"]["pass"]:
        return (
            "drop",
            "Critical recall guard failed; rollback to baseline policy immediately.",
        )
    if failed:
        return (
            "iterate",
            "Safety constraints hold but maturation thresholds are not fully satisfied; continue tuning.",
        )
    return (
        "promote",
        "Candidate policy improved uniform recall while maintaining low overhead and full safety guards.",
    )


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T5 Week 8 Block 5 - Reliability Maturation Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline policy: `{report['metadata']['baseline_policy_id']}`")
    lines.append(f"- Candidate policy: `{report['metadata']['candidate_policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']} | Seed={report['metadata']['seed']}"
    )
    lines.append("")
    lines.append("## Baseline vs Candidate")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    for metric in [
        "effective_overhead_percent",
        "critical_recall",
        "uniform_recall",
        "false_positive_rate",
        "max_error",
        "effective_gflops_mean",
    ]:
        lines.append(
            f"| {metric} | {report['evaluation']['baseline'][metric]:.6f} | {report['evaluation']['candidate'][metric]:.6f} |"
        )
    lines.append(
        f"| uniform_recall_delta_vs_baseline | - | {report['evaluation']['uniform_recall_delta_vs_baseline']:.6f} |"
    )
    lines.append(
        f"| overhead_delta_vs_baseline_percent | - | {report['evaluation']['overhead_delta_vs_baseline_percent']:.6f} |"
    )
    lines.append("")
    lines.append("## Guardrail Evaluation")
    lines.append("")
    lines.append("| Guardrail | Observed | Threshold | Comparator | Pass |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for name, payload in report["evaluation"]["checks"].items():
        threshold = payload.get("threshold_max", payload.get("threshold_min"))
        lines.append(
            f"| {name} | {payload['observed']:.6f} | {threshold:.6f} | {payload['comparator']} | {payload['pass']} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    baseline_policy_path: Path,
    candidate_policy_path: Path,
) -> dict[str, Any]:
    baseline_policy = _load_policy(baseline_policy_path)
    candidate_policy = _load_policy(candidate_policy_path)
    campaign = candidate_policy.get("maturation_campaign", {})

    sizes = [int(x) for x in candidate_policy["scope"]["sizes"]]
    sessions = int(campaign.get("sessions", 10))
    iterations = int(campaign.get("iterations", 24))
    warmup = int(campaign.get("warmup", 2))
    fault_scale = float(campaign.get("fault_scale", 0.05))
    fault_abs_min = float(campaign.get("fault_abs_min", 1.0))
    seed = int(campaign.get("seed", 42))

    b_mode = baseline_policy["abft_mode"]
    c_mode = candidate_policy["abft_mode"]

    baseline_report = run_experiment(
        sizes=sizes,
        sessions=sessions,
        iterations=iterations,
        warmup=warmup,
        sampling_periods=[int(b_mode["sampling_period"])],
        row_samples=int(b_mode["row_samples"]),
        col_samples=int(b_mode["col_samples"]),
        faults_per_matrix=int(b_mode["faults_per_matrix_modelled"]),
        fault_scale=fault_scale,
        fault_abs_min=fault_abs_min,
        residual_scale=float(b_mode["residual_scale"]),
        residual_margin=float(b_mode["residual_margin"]),
        residual_floor=float(b_mode["residual_floor"]),
        projection_count=int(b_mode["projection_count"]),
        correctness_threshold=float(
            candidate_policy["runtime_guardrails"]["disable_if_correctness_error_gt"]
        ),
        seed=seed,
    )
    candidate_report = run_experiment(
        sizes=sizes,
        sessions=sessions,
        iterations=iterations,
        warmup=warmup,
        sampling_periods=[int(c_mode["sampling_period"])],
        row_samples=int(c_mode["row_samples"]),
        col_samples=int(c_mode["col_samples"]),
        faults_per_matrix=int(c_mode["faults_per_matrix_modelled"]),
        fault_scale=fault_scale,
        fault_abs_min=fault_abs_min,
        residual_scale=float(c_mode["residual_scale"]),
        residual_margin=float(c_mode["residual_margin"]),
        residual_floor=float(c_mode["residual_floor"]),
        projection_count=int(c_mode["projection_count"]),
        correctness_threshold=float(
            candidate_policy["runtime_guardrails"]["disable_if_correctness_error_gt"]
        ),
        seed=seed,
    )

    baseline_mode = _single_mode(baseline_report)
    candidate_mode = _single_mode(candidate_report)
    evaluation = _evaluate(
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
        guardrails=candidate_policy["runtime_guardrails"],
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_policy_path": str(baseline_policy_path),
            "baseline_policy_id": str(baseline_policy["policy_id"]),
            "candidate_policy_path": str(candidate_policy_path),
            "candidate_policy_id": str(candidate_policy["policy_id"]),
            "sizes": sizes,
            "sessions": sessions,
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "fault_scale": fault_scale,
            "fault_abs_min": fault_abs_min,
        },
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "evaluation": evaluation,
        "baseline_source_report": baseline_report,
        "candidate_source_report": candidate_report,
    }
    decision, rationale = _decision(report)
    report["decision"] = {"decision": decision, "rationale": rationale}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 8 Block 5 T5 reliability maturation.")
    parser.add_argument(
        "--baseline-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json",
    )
    parser.add_argument(
        "--candidate-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t5_reliability_abft",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    baseline_policy_path = (repo_root / args.baseline_policy_path).resolve()
    candidate_policy_path = (repo_root / args.candidate_policy_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_campaign(
        baseline_policy_path=baseline_policy_path,
        candidate_policy_path=candidate_policy_path,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_t5_maturation_{timestamp}.json"
    md_path = output_dir / f"week8_t5_maturation_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"T5 maturation JSON: {json_path}")
    print(f"T5 maturation MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(
        "Uniform recall delta vs baseline: "
        f"{report['evaluation']['uniform_recall_delta_vs_baseline']:+.3f}"
    )
    print(
        "Overhead delta vs baseline (%): "
        f"{report['evaluation']['overhead_delta_vs_baseline_percent']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
