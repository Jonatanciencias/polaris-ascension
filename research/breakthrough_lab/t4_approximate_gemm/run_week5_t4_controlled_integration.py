#!/usr/bin/env python3
"""
Week 5 Block 2 - T4 controlled integration runner.

Consumes the promoted activation policy from Week 4 Block 3 and executes
a strict deterministic rerun with explicit guardrail evaluation for controlled
integration progression.
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


def _evaluate_guardrails(
    *,
    summary: dict[str, Any],
    guardrails: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "post_fallback_violation_rate": {
            "observed": float(summary["post_fallback_violation_rate"]),
            "threshold": float(guardrails["disable_if_post_fallback_violation_rate_gt"]),
            "comparator": "<=",
        },
        "contract_compliance_rate": {
            "observed": float(summary["contract_compliance_rate"]),
            "threshold": float(guardrails["disable_if_contract_compliance_rate_lt"]),
            "comparator": ">=",
        },
        "fallback_rate": {
            "observed": float(summary["fallback_rate"]),
            "threshold": float(guardrails["disable_if_fallback_rate_gt"]),
            "comparator": "<=",
        },
        "compressible_speedup_vs_exact_mean": {
            "observed": float(summary["compressible_speedup_vs_exact_mean"]),
            "threshold": float(guardrails["disable_if_compressible_speedup_lt"]),
            "comparator": ">=",
        },
    }

    for payload in checks.values():
        if payload["comparator"] == "<=":
            payload["pass"] = bool(payload["observed"] <= payload["threshold"])
        else:
            payload["pass"] = bool(payload["observed"] >= payload["threshold"])

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    all_passed = len(failed) == 0
    disable_signal = len(failed) > 0
    return {
        "checks": checks,
        "failed_guardrails": failed,
        "all_passed": all_passed,
        "disable_signal": disable_signal,
        "fallback_action": (
            "auto_disable_t4_approximate_mode"
            if disable_signal
            else "continue_controlled_integration_ready_for_next_gate"
        ),
    }


def _decision(report: dict[str, Any]) -> tuple[str, str]:
    guardrail_eval = report["guardrails"]
    summary = report["summary"]

    if float(summary["post_fallback_violation_rate"]) > 0.05:
        return (
            "drop",
            "Stop rule triggered by post-fallback contract escapes over hard limit.",
        )
    if not guardrail_eval["all_passed"]:
        return (
            "iterate",
            "Guardrail violations detected in controlled rerun; keep exact fallback active and refine policy.",
        )
    return (
        "promote",
        "Controlled rerun passed all guardrails with deterministic behavior; policy remains valid for controlled progression.",
    )


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T4 Week 5 Block 2 - Controlled Integration Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['metadata']['policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(f"- Families: {report['metadata']['families']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Seed={report['metadata']['seed']} | Error budget={report['metadata']['error_budget']}"
    )
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- Contract compliance rate: {report['summary']['contract_compliance_rate']:.3f}")
    lines.append(
        f"- Post-fallback violation rate: {report['summary']['post_fallback_violation_rate']:.3f}"
    )
    lines.append(f"- Fallback rate: {report['summary']['fallback_rate']:.3f}")
    lines.append(f"- Policy exact-route rate: {report['summary']['policy_exact_route_rate']:.3f}")
    lines.append(
        f"- Compressible speedup vs exact: {report['summary']['compressible_speedup_vs_exact_mean']:.3f}x"
    )
    lines.append("")
    lines.append("## Guardrail Evaluation")
    lines.append("")
    lines.append("| Guardrail | Observed | Threshold | Comparator | Pass |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for name, payload in report["guardrails"]["checks"].items():
        lines.append(
            f"| {name} | {payload['observed']:.6f} | {payload['threshold']:.6f} | {payload['comparator']} | {payload['pass']} |"
        )
    lines.append("")
    lines.append(f"- Disable signal: {report['guardrails']['disable_signal']}")
    lines.append(f"- Fallback action: `{report['guardrails']['fallback_action']}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_controlled_integration(
    *,
    policy_path: Path,
    sessions: int,
    seed: int,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    scope = policy["scope"]
    activation = policy["activation_policy"]
    guardrails = policy["runtime_guardrails"]

    families = [str(x) for x in scope["families_tested"]]
    sizes = [int(x) for x in scope["sizes"]]

    base_report = run_experiment(
        families=families,
        sizes=sizes,
        sessions=sessions,
        target_rank=int(activation["target_rank"]),
        noise_scale=0.01,
        error_budget=float(activation["error_budget"]),
        precheck_energy_threshold=float(activation["precheck_energy_threshold_exact_route"]),
        sample_size=int(activation["sample_size"]),
        seed=seed,
    )

    guardrail_eval = _evaluate_guardrails(
        summary=base_report["summary"],
        guardrails=guardrails,
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_path": str(policy_path),
            "policy_id": str(policy["policy_id"]),
            "sizes": sizes,
            "families": families,
            "sessions": int(sessions),
            "seed": int(seed),
            "target_rank": int(activation["target_rank"]),
            "sample_size": int(activation["sample_size"]),
            "error_budget": float(activation["error_budget"]),
            "precheck_energy_threshold": float(
                activation["precheck_energy_threshold_exact_route"]
            ),
        },
        "summary": base_report["summary"],
        "guardrails": guardrail_eval,
        "source_report": base_report,
    }
    decision, rationale = _decision(report)
    report["decision"] = {
        "decision": decision,
        "rationale": rationale,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T4 Week 5 Block 2 controlled integration.")
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json",
    )
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t4_approximate_gemm",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    policy_path = (repo_root / args.policy_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_controlled_integration(
        policy_path=policy_path,
        sessions=int(args.sessions),
        seed=int(args.seed),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week5_t4_controlled_integration_{timestamp}.json"
    md_path = output_dir / f"week5_t4_controlled_integration_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"T4 controlled JSON: {json_path}")
    print(f"T4 controlled MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(f"Guardrails passed: {report['guardrails']['all_passed']}")
    print(f"Fallback rate: {report['summary']['fallback_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
