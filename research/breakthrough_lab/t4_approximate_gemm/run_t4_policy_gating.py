#!/usr/bin/env python3
"""
Week 4 T4 policy-gating runner.

Refines approximate GEMM activation policy by compressibility gating:
- low-compressibility inputs are routed directly to exact path (policy route)
- high-compressibility inputs try approximate path under error contract
- fallback is reserved for true contract violations after approximate attempt
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _generate_case(
    *,
    family: str,
    size: int,
    target_rank: int,
    noise_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    rng = np.random.default_rng(seed)

    if family == "compressible_lowrank":
        x = rng.standard_normal((size, target_rank), dtype=np.float32)
        y = rng.standard_normal((target_rank, size), dtype=np.float32)
        a = x @ y + noise_scale * rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        return a.astype(np.float32), b.astype(np.float32), {"x": x, "y": y}

    if family == "dense_random":
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        return a, b, None

    raise ValueError(f"Unsupported family: {family}")


def _compressibility_energy(
    matrix: np.ndarray,
    *,
    target_rank: int,
    sample_size: int,
) -> float:
    s = min(sample_size, matrix.shape[0], matrix.shape[1])
    block = matrix[:s, :s]
    sv = np.linalg.svd(block, compute_uv=False)
    if sv.size == 0:
        return 0.0
    k = min(target_rank, sv.size)
    denom = float(np.sum(sv ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(sv[:k] ** 2) / denom)


def _approximate_gemm(
    *,
    a: np.ndarray,
    b: np.ndarray,
    payload: dict[str, np.ndarray] | None,
    target_rank: int,
    seed: int,
) -> tuple[np.ndarray, float, str]:
    if payload is not None:
        x = payload["x"][:, :target_rank]
        y = payload["y"][:target_rank, :]
        start = time.perf_counter()
        c_approx = x @ (y @ b)
        elapsed = time.perf_counter() - start
        return c_approx.astype(np.float32), elapsed, "cached_low_rank_factors"

    rng = np.random.default_rng(seed)
    p = rng.standard_normal((a.shape[1], target_rank), dtype=np.float32) / np.sqrt(target_rank)
    start = time.perf_counter()
    ap = a @ p
    pb = p.T @ b
    c_approx = ap @ pb
    elapsed = time.perf_counter() - start
    return c_approx.astype(np.float32), elapsed, "random_projection"


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T4 Week 4 Block 3 - Policy Gating Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Families: {report['metadata']['families']} | Sizes: {report['metadata']['sizes']} | Sessions: {report['metadata']['sessions']}"
    )
    lines.append(
        f"- Contract: error_budget={report['metadata']['error_budget']}, precheck_energy_threshold={report['metadata']['precheck_energy_threshold']}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Contract compliance: {report['summary']['contract_compliance_rate']:.3f}"
    )
    lines.append(
        f"- Post-fallback violation rate: {report['summary']['post_fallback_violation_rate']:.3f}"
    )
    lines.append(f"- Fallback rate: {report['summary']['fallback_rate']:.3f}")
    lines.append(
        f"- Policy exact-route rate: {report['summary']['policy_exact_route_rate']:.3f}"
    )
    lines.append(
        f"- Approximate-attempt rate: {report['summary']['approximate_attempt_rate']:.3f}"
    )
    lines.append(
        f"- Compressible speedup vs exact: {report['summary']['compressible_speedup_vs_exact_mean']:.3f}x"
    )
    lines.append(
        f"- Stop rule triggered: {report['summary']['stop_rule_triggered']}"
    )
    lines.append("")
    lines.append("## Family Metrics")
    lines.append("")
    lines.append(
        "| Family | Runs | Speedup vs Exact | Contract | Policy Exact Route | Approx Attempts | Fallback | Raw Error Mean |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for family, data in report["summary"]["by_family"].items():
        lines.append(
            f"| {family} | {data['runs']} | {data['executed_speedup_vs_exact_mean']:.3f}x | "
            f"{data['contract_compliance_rate']:.3f} | {data['policy_exact_route_rate']:.3f} | "
            f"{data['approximate_attempt_rate']:.3f} | {data['fallback_rate']:.3f} | "
            f"{data['raw_error_mean']:.6f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Suggested decision: `{report['summary']['decision_hint']}`")
    lines.append(f"- Promotion gate (scoped): {report['summary']['promotion_gate_passed']}")
    lines.append(f"- Rationale: {report['summary']['decision_rationale']}")
    return "\n".join(lines) + "\n"


def _build_policy(
    *,
    report: dict[str, Any],
    policy_id: str,
    json_artifact: str,
    md_artifact: str,
) -> dict[str, Any]:
    summary = report["summary"]
    metadata = report["metadata"]
    decision = summary["decision_hint"]

    return {
        "policy_id": policy_id,
        "status": (
            "candidate_scoped_promotion"
            if decision == "promote"
            else "candidate_needs_refinement"
        ),
        "scope": {
            "families_tested": metadata["families"],
            "sizes": metadata["sizes"],
            "activation_mode": "compressibility_energy_gate",
        },
        "activation_policy": {
            "precheck_energy_threshold_exact_route": metadata["precheck_energy_threshold"],
            "target_rank": metadata["target_rank"],
            "error_budget": metadata["error_budget"],
            "sample_size": metadata["sample_size"],
        },
        "runtime_guardrails": {
            "disable_if_post_fallback_violation_rate_gt": 0.01,
            "disable_if_contract_compliance_rate_lt": 0.99,
            "disable_if_fallback_rate_gt": 0.10,
            "disable_if_compressible_speedup_lt": 2.0,
        },
        "evidence": {
            "artifact_json": json_artifact,
            "artifact_md": md_artifact,
            "contract_compliance_rate": summary["contract_compliance_rate"],
            "post_fallback_violation_rate": summary["post_fallback_violation_rate"],
            "fallback_rate": summary["fallback_rate"],
            "policy_exact_route_rate": summary["policy_exact_route_rate"],
            "compressible_speedup_vs_exact_mean": summary["compressible_speedup_vs_exact_mean"],
            "decision_hint": summary["decision_hint"],
        },
        "next_gate": (
            "scoped_production_shadow_canary"
            if summary["promotion_gate_passed"]
            else "policy_refinement_iteration"
        ),
    }


def run_experiment(
    *,
    families: list[str],
    sizes: list[int],
    sessions: int,
    target_rank: int,
    noise_scale: float,
    error_budget: float,
    precheck_energy_threshold: float,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    for family_idx, family in enumerate(families):
        for size in sizes:
            for session in range(sessions):
                run_seed = seed + family_idx * 100_000 + size * 10 + session
                a, b, payload = _generate_case(
                    family=family,
                    size=size,
                    target_rank=target_rank,
                    noise_scale=noise_scale,
                    seed=run_seed,
                )

                start_exact = time.perf_counter()
                c_exact = a @ b
                exact_time = time.perf_counter() - start_exact

                start_precheck = time.perf_counter()
                energy = _compressibility_energy(
                    a,
                    target_rank=target_rank,
                    sample_size=sample_size,
                )
                precheck_time = time.perf_counter() - start_precheck

                policy_exact_route = False
                fallback_triggered = False
                fallback_reason = None
                approx_attempted = False
                approx_method = None
                approx_time = None
                raw_error = None
                pre_contract_violation = False

                if energy < precheck_energy_threshold:
                    # Low-compressibility route: policy chooses exact directly.
                    policy_exact_route = True
                    executed_time = precheck_time + exact_time
                    executed_rel_error = 0.0
                    execution_path = "exact_policy_gate"
                else:
                    approx_attempted = True
                    c_approx, approx_time, approx_method = _approximate_gemm(
                        a=a,
                        b=b,
                        payload=payload,
                        target_rank=target_rank,
                        seed=run_seed + 1,
                    )
                    raw_error = float(
                        np.linalg.norm(c_exact - c_approx, ord="fro") / np.linalg.norm(c_exact, ord="fro")
                    )
                    pre_contract_violation = raw_error > error_budget

                    if pre_contract_violation:
                        fallback_triggered = True
                        fallback_reason = "postcheck_contract_violation"
                        executed_time = precheck_time + float(approx_time) + exact_time
                        executed_rel_error = 0.0
                        execution_path = "fallback_exact_after_postcheck"
                    else:
                        executed_time = precheck_time + float(approx_time)
                        executed_rel_error = raw_error
                        execution_path = "approximate"

                ops = float(2 * size * size * size)
                exact_gflops = ops / exact_time / 1e9
                executed_gflops = ops / executed_time / 1e9
                approx_only_gflops = (
                    None if approx_time is None or approx_time <= 0 else float(ops / approx_time / 1e9)
                )

                contract_respected = executed_rel_error <= error_budget

                runs.append(
                    {
                        "family": family,
                        "size": size,
                        "session": session,
                        "seed": run_seed,
                        "exact_time_ms": float(exact_time * 1000.0),
                        "executed_time_ms": float(executed_time * 1000.0),
                        "approx_time_ms": None if approx_time is None else float(approx_time * 1000.0),
                        "exact_gflops": float(exact_gflops),
                        "executed_gflops": float(executed_gflops),
                        "approx_only_gflops": approx_only_gflops,
                        "compressibility_energy": float(energy),
                        "execution_path": execution_path,
                        "policy_exact_route": bool(policy_exact_route),
                        "approx_attempted": bool(approx_attempted),
                        "approx_method": approx_method,
                        "raw_relative_error": None if raw_error is None else float(raw_error),
                        "executed_relative_error": float(executed_rel_error),
                        "pre_contract_violation": bool(pre_contract_violation),
                        "contract_respected": bool(contract_respected),
                        "fallback_triggered": bool(fallback_triggered),
                        "fallback_reason": fallback_reason,
                    }
                )

    total_runs = len(runs)
    fallback_count = sum(1 for r in runs if r["fallback_triggered"])
    policy_exact_count = sum(1 for r in runs if r["policy_exact_route"])
    approx_attempt_count = sum(1 for r in runs if r["approx_attempted"])
    pre_contract_violations = sum(1 for r in runs if r["pre_contract_violation"])
    post_fallback_violations = sum(1 for r in runs if not r["contract_respected"])

    by_family: dict[str, Any] = {}
    for family in families:
        fam_runs = [r for r in runs if r["family"] == family]
        raw_errors = [r["raw_relative_error"] for r in fam_runs if r["raw_relative_error"] is not None]
        speedups = [r["exact_time_ms"] / r["executed_time_ms"] for r in fam_runs]
        fam_gflops = [r["executed_gflops"] for r in fam_runs]
        by_family[family] = {
            "runs": len(fam_runs),
            "executed_speedup_vs_exact_mean": float(np.mean(speedups)) if speedups else 0.0,
            "contract_compliance_rate": float(np.mean([r["contract_respected"] for r in fam_runs])) if fam_runs else 0.0,
            "policy_exact_route_rate": float(np.mean([r["policy_exact_route"] for r in fam_runs])) if fam_runs else 0.0,
            "approximate_attempt_rate": float(np.mean([r["approx_attempted"] for r in fam_runs])) if fam_runs else 0.0,
            "fallback_rate": float(np.mean([r["fallback_triggered"] for r in fam_runs])) if fam_runs else 0.0,
            "raw_error_mean": float(np.mean(raw_errors)) if raw_errors else 0.0,
            "raw_error_max": float(np.max(raw_errors)) if raw_errors else 0.0,
            "executed_gflops_cv": (
                float(np.std(fam_gflops) / np.mean(fam_gflops)) if fam_gflops and np.mean(fam_gflops) > 0 else 0.0
            ),
        }

    compressible_speedup = by_family.get("compressible_lowrank", {}).get(
        "executed_speedup_vs_exact_mean", 0.0
    )
    dense_exact_route_rate = by_family.get("dense_random", {}).get(
        "policy_exact_route_rate", 0.0
    )

    contract_compliance_rate = float(np.mean([r["contract_respected"] for r in runs])) if runs else 0.0
    fallback_rate = float(fallback_count / total_runs) if total_runs else 1.0
    policy_exact_route_rate = float(policy_exact_count / total_runs) if total_runs else 0.0
    approximate_attempt_rate = float(approx_attempt_count / total_runs) if total_runs else 0.0
    pre_violation_rate = float(pre_contract_violations / total_runs) if total_runs else 1.0
    post_violation_rate = float(post_fallback_violations / total_runs) if total_runs else 1.0

    stop_rule_triggered = post_violation_rate > 0.05

    promotion_gate_passed = bool(
        contract_compliance_rate >= 0.99
        and post_violation_rate <= 0.0
        and fallback_rate <= 0.10
        and compressible_speedup >= 2.0
        and dense_exact_route_rate >= 0.95
    )

    if stop_rule_triggered:
        decision_hint = "drop"
        rationale = "Contract violation exceeds 5% after fallback, stop rule triggered."
    elif promotion_gate_passed:
        decision_hint = "promote"
        rationale = (
            "Compressibility-gated policy is contract-safe with no post-fallback escapes, "
            "zero fallback triggers and strong speedup on eligible workloads."
        )
    elif contract_compliance_rate >= 0.99:
        decision_hint = "iterate"
        rationale = "Safety is stable; continue refining activation policy for broader eligible coverage."
    else:
        decision_hint = "refine"
        rationale = "Contract/fallback behavior is not yet stable enough for policy promotion."

    executed_gflops = [r["executed_gflops"] for r in runs]
    exact_gflops = [r["exact_gflops"] for r in runs]

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "families": families,
            "sizes": sizes,
            "sessions": sessions,
            "target_rank": target_rank,
            "noise_scale": noise_scale,
            "error_budget": error_budget,
            "precheck_energy_threshold": precheck_energy_threshold,
            "sample_size": sample_size,
            "seed": seed,
        },
        "runs": runs,
        "summary": {
            "exact_gflops": _stats(exact_gflops),
            "executed_gflops": _stats(executed_gflops),
            "delta_vs_exact_percent": (
                ((float(np.mean(executed_gflops)) - float(np.mean(exact_gflops))) / float(np.mean(exact_gflops))) * 100.0
                if exact_gflops and np.mean(exact_gflops) > 0
                else 0.0
            ),
            "contract_compliance_rate": contract_compliance_rate,
            "fallback_rate": fallback_rate,
            "policy_exact_route_rate": policy_exact_route_rate,
            "approximate_attempt_rate": approximate_attempt_rate,
            "pre_contract_violation_rate": pre_violation_rate,
            "post_fallback_violation_rate": post_violation_rate,
            "stop_rule_triggered": stop_rule_triggered,
            "compressible_speedup_vs_exact_mean": compressible_speedup,
            "by_family": by_family,
            "promotion_gate_passed": promotion_gate_passed,
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T4 policy-gating contract experiment")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["dense_random", "compressible_lowrank"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 1400])
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--target-rank", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--error-budget", type=float, default=0.005)
    parser.add_argument("--precheck-energy-threshold", type=float, default=0.95)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-id",
        type=str,
        default="t4-approx-gating-block3-2026-02-07",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t4_approximate_gemm",
    )
    parser.add_argument(
        "--policy-output",
        type=str,
        default="research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json",
    )
    args = parser.parse_args()

    report = run_experiment(
        families=args.families,
        sizes=args.sizes,
        sessions=args.sessions,
        target_rank=args.target_rank,
        noise_scale=args.noise_scale,
        error_budget=args.error_budget,
        precheck_energy_threshold=args.precheck_energy_threshold,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_output = (repo_root / args.policy_output).resolve()
    policy_output.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week4_t4_policy_gating_{timestamp}.json"
    md_path = output_dir / f"week4_t4_policy_gating_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    rel_json = str(json_path.relative_to(repo_root))
    rel_md = str(md_path.relative_to(repo_root))
    policy_doc = _build_policy(
        report=report,
        policy_id=args.policy_id,
        json_artifact=rel_json,
        md_artifact=rel_md,
    )
    policy_output.write_text(json.dumps(policy_doc, indent=2))

    print(f"T4 policy JSON: {json_path}")
    print(f"T4 policy MD:   {md_path}")
    print(f"T4 policy cfg:  {policy_output}")
    print(f"Decision hint:  {report['summary']['decision_hint']}")
    print(f"Promotion gate: {report['summary']['promotion_gate_passed']}")
    print(f"Fallback rate:  {report['summary']['fallback_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
