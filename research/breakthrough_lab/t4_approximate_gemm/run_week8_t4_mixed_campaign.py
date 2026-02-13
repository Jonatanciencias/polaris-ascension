#!/usr/bin/env python3
"""Week 8 Block 4: mixed-workload T4 activation refinement campaign.

Objective:
- compare baseline policy (block3) vs refined policy (block4 candidate)
- reduce postcheck fallback rate without breaking error-contract safety
- keep a minimum compressible workload speedup floor
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_t4_policy_gating import run_experiment


def _load_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = [
        "policy_id",
        "scope",
        "activation_policy",
        "runtime_guardrails",
    ]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Policy missing required fields: {missing}")
    return data


def _summary_view(report: dict[str, Any]) -> dict[str, float]:
    s = report["summary"]
    return {
        "contract_compliance_rate": float(s["contract_compliance_rate"]),
        "post_fallback_violation_rate": float(s["post_fallback_violation_rate"]),
        "fallback_rate": float(s["fallback_rate"]),
        "policy_exact_route_rate": float(s["policy_exact_route_rate"]),
        "approximate_attempt_rate": float(s["approximate_attempt_rate"]),
        "compressible_speedup_vs_exact_mean": float(s["compressible_speedup_vs_exact_mean"]),
        "delta_vs_exact_percent": float(s["delta_vs_exact_percent"]),
    }


def _evaluate(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    guardrails: dict[str, float],
) -> dict[str, Any]:
    b = _summary_view(baseline)
    c = _summary_view(candidate)
    fallback_reduction_abs = float(b["fallback_rate"] - c["fallback_rate"])

    checks = {
        "candidate_contract_compliance": {
            "observed": c["contract_compliance_rate"],
            "threshold_min": float(guardrails["disable_if_contract_compliance_rate_lt"]),
            "comparator": ">=",
        },
        "candidate_post_fallback_violation_rate": {
            "observed": c["post_fallback_violation_rate"],
            "threshold_max": float(guardrails["disable_if_post_fallback_violation_rate_gt"]),
            "comparator": "<=",
        },
        "candidate_fallback_rate": {
            "observed": c["fallback_rate"],
            "threshold_max": float(guardrails["disable_if_fallback_rate_gt"]),
            "comparator": "<=",
        },
        "candidate_compressible_speedup": {
            "observed": c["compressible_speedup_vs_exact_mean"],
            "threshold_min": float(guardrails["disable_if_compressible_speedup_lt"]),
            "comparator": ">=",
        },
        "fallback_reduction_vs_baseline": {
            "observed": fallback_reduction_abs,
            "threshold_min": float(guardrails["require_fallback_reduction_abs_ge"]),
            "comparator": ">=",
        },
        "policy_exact_route_cap": {
            "observed": c["policy_exact_route_rate"],
            "threshold_max": float(guardrails["max_policy_exact_route_rate"]),
            "comparator": "<=",
        },
    }

    for payload in checks.values():
        comparator = payload["comparator"]
        if comparator == "<=":
            payload["pass"] = bool(payload["observed"] <= payload["threshold_max"])
        else:
            payload["pass"] = bool(payload["observed"] >= payload["threshold_min"])

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    return {
        "baseline": b,
        "candidate": c,
        "fallback_reduction_abs": fallback_reduction_abs,
        "checks": checks,
        "failed_checks": failed,
        "all_passed": len(failed) == 0,
    }


def _decision(report: dict[str, Any]) -> tuple[str, str]:
    checks = report["evaluation"]["checks"]
    failed_checks = report["evaluation"]["failed_checks"]
    if checks["candidate_contract_compliance"]["pass"] is False:
        return (
            "drop",
            "Candidate policy violated contract compliance threshold; keep exact fallback mode and reject activation change.",
        )
    if checks["candidate_post_fallback_violation_rate"]["pass"] is False:
        return (
            "drop",
            "Post-fallback violations detected above hard safety threshold.",
        )
    if failed_checks:
        return (
            "iterate",
            "Safety preserved but one or more refinement guardrails failed; continue tuning before promotion.",
        )
    return (
        "promote",
        "Fallback reduction and safety/performance guardrails passed on mixed workload campaign.",
    )


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T4 Week 8 Block 4 - Mixed Policy Refinement Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline policy: `{report['metadata']['baseline_policy_id']}`")
    lines.append(f"- Candidate policy: `{report['metadata']['candidate_policy_id']}`")
    lines.append(f"- Families: {report['metadata']['families']} | Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Seed={report['metadata']['seed']} | Noise={report['metadata']['noise_scale']}"
    )
    lines.append("")
    lines.append("## Baseline vs Candidate")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    for metric in [
        "contract_compliance_rate",
        "post_fallback_violation_rate",
        "fallback_rate",
        "policy_exact_route_rate",
        "approximate_attempt_rate",
        "compressible_speedup_vs_exact_mean",
        "delta_vs_exact_percent",
    ]:
        lines.append(
            f"| {metric} | {report['evaluation']['baseline'][metric]:.6f} | {report['evaluation']['candidate'][metric]:.6f} |"
        )
    lines.append(
        f"| fallback_reduction_abs | - | {report['evaluation']['fallback_reduction_abs']:.6f} |"
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
    sessions: int,
    seed: int,
) -> dict[str, Any]:
    baseline_policy = _load_policy(baseline_policy_path)
    candidate_policy = _load_policy(candidate_policy_path)

    scope = candidate_policy["scope"]
    families = [str(x) for x in scope["families_tested"]]
    sizes = [int(x) for x in scope["sizes"]]
    noise_scale = float(candidate_policy["mixed_campaign"]["noise_scale"])
    guardrails = candidate_policy["runtime_guardrails"]
    sample_size = int(candidate_policy["activation_policy"]["sample_size"])
    error_budget = float(candidate_policy["activation_policy"]["error_budget"])
    energy_threshold = float(
        candidate_policy["activation_policy"]["precheck_energy_threshold_exact_route"]
    )

    baseline_report = run_experiment(
        families=families,
        sizes=sizes,
        sessions=int(sessions),
        target_rank=int(baseline_policy["activation_policy"]["target_rank"]),
        noise_scale=noise_scale,
        error_budget=error_budget,
        precheck_energy_threshold=energy_threshold,
        sample_size=sample_size,
        seed=int(seed),
    )
    candidate_report = run_experiment(
        families=families,
        sizes=sizes,
        sessions=int(sessions),
        target_rank=int(candidate_policy["activation_policy"]["target_rank"]),
        noise_scale=noise_scale,
        error_budget=error_budget,
        precheck_energy_threshold=energy_threshold,
        sample_size=sample_size,
        seed=int(seed),
    )

    evaluation = _evaluate(
        baseline=baseline_report,
        candidate=candidate_report,
        guardrails=guardrails,
    )
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_policy_path": str(baseline_policy_path),
            "baseline_policy_id": str(baseline_policy["policy_id"]),
            "candidate_policy_path": str(candidate_policy_path),
            "candidate_policy_id": str(candidate_policy["policy_id"]),
            "families": families,
            "sizes": sizes,
            "sessions": int(sessions),
            "seed": int(seed),
            "noise_scale": noise_scale,
            "sample_size": sample_size,
            "error_budget": error_budget,
        },
        "baseline_summary": _summary_view(baseline_report),
        "candidate_summary": _summary_view(candidate_report),
        "evaluation": evaluation,
        "baseline_source_report": baseline_report,
        "candidate_source_report": candidate_report,
    }
    decision, rationale = _decision(report)
    report["decision"] = {"decision": decision, "rationale": rationale}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 8 Block 4 T4 mixed refinement campaign.")
    parser.add_argument(
        "--baseline-policy-path",
        default="research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json",
    )
    parser.add_argument(
        "--candidate-policy-path",
        default="research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t4_approximate_gemm",
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
        sessions=int(args.sessions),
        seed=int(args.seed),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_t4_mixed_campaign_{timestamp}.json"
    md_path = output_dir / f"week8_t4_mixed_campaign_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"T4 mixed JSON: {json_path}")
    print(f"T4 mixed MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(
        "Fallback reduction abs: "
        f"{report['evaluation']['fallback_reduction_abs']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
