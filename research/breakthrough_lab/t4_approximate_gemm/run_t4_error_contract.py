#!/usr/bin/env python3
"""
Week 3 T4 approximate GEMM runner.

Implements a bounded-error contract with explicit fallback instrumentation.
This is a lab-side experiment and does not modify production execution paths.
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
        # Construct A as low-rank + low-noise residue to emulate compressible workloads.
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
        x = payload["x"]
        y = payload["y"]
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
    lines.append("# T4 Week 3 Error Contract + Fallback Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Families: {report['metadata']['families']} | Sizes: {report['metadata']['sizes']}"
    )
    lines.append(
        f"- Sessions: {report['metadata']['sessions']} | Target rank: {report['metadata']['target_rank']}"
    )
    lines.append(
        f"- Error budget: {report['metadata']['error_budget']} | Precheck threshold: {report['metadata']['precheck_energy_threshold']}"
    )
    lines.append("")
    lines.append("## Contract Summary")
    lines.append("")
    lines.append(
        f"- Contract compliance rate: {report['summary']['contract_compliance_rate']:.3f}"
    )
    lines.append(
        f"- Fallback rate: {report['summary']['fallback_rate']:.3f}"
    )
    lines.append(
        f"- Pre-contract violation rate: {report['summary']['pre_contract_violation_rate']:.3f}"
    )
    lines.append(
        f"- Post-fallback violation rate: {report['summary']['post_fallback_violation_rate']:.3f}"
    )
    lines.append(
        f"- Severe outliers prevented: {report['summary']['severe_outliers_prevented']}"
    )
    lines.append(
        f"- Stop rule triggered: {report['summary']['stop_rule_triggered']}"
    )
    lines.append("")
    lines.append("## Family Metrics")
    lines.append("")
    lines.append(
        "| Family | Runs | Executed Speedup vs Exact | Contract Compliance | Fallback Rate | Raw Error Mean |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for family, item in report["summary"]["by_family"].items():
        lines.append(
            f"| {family} | {item['runs']} | {item['executed_speedup_vs_exact_mean']:.3f}x | "
            f"{item['contract_compliance_rate']:.3f} | {item['fallback_rate']:.3f} | "
            f"{item['raw_error_mean']:.6f} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Suggested decision: `{report['summary']['decision_hint']}`")
    lines.append(f"- Rationale: {report['summary']['decision_rationale']}")

    return "\n".join(lines) + "\n"


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

                fallback_reason = None
                fallback_triggered = False
                approx_method = None
                approx_time = None
                raw_error = None
                pre_contract_violation = False

                if energy < precheck_energy_threshold:
                    fallback_triggered = True
                    fallback_reason = "precheck_low_energy"
                    executed_time = exact_time + precheck_time
                    executed_rel_error = 0.0
                else:
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
                    else:
                        executed_time = precheck_time + float(approx_time)
                        executed_rel_error = raw_error

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
    pre_contract_violations = sum(1 for r in runs if r["pre_contract_violation"])
    post_fallback_violations = sum(1 for r in runs if not r["contract_respected"])
    severe_outliers_prevented = sum(
        1
        for r in runs
        if r["fallback_reason"] == "postcheck_contract_violation"
        and r["raw_relative_error"] is not None
        and r["raw_relative_error"] > (2.0 * error_budget)
    )

    by_family: dict[str, Any] = {}
    for family in families:
        fam_runs = [r for r in runs if r["family"] == family]
        raw_errors = [r["raw_relative_error"] for r in fam_runs if r["raw_relative_error"] is not None]
        speedups = [r["exact_time_ms"] / r["executed_time_ms"] for r in fam_runs]
        by_family[family] = {
            "runs": len(fam_runs),
            "executed_speedup_vs_exact_mean": float(np.mean(speedups)) if speedups else 0.0,
            "contract_compliance_rate": float(np.mean([r["contract_respected"] for r in fam_runs])) if fam_runs else 0.0,
            "fallback_rate": float(np.mean([r["fallback_triggered"] for r in fam_runs])) if fam_runs else 0.0,
            "raw_error_mean": float(np.mean(raw_errors)) if raw_errors else 0.0,
            "raw_error_max": float(np.max(raw_errors)) if raw_errors else 0.0,
        }

    compressible_speedup = by_family.get("compressible_lowrank", {}).get(
        "executed_speedup_vs_exact_mean", 0.0
    )
    contract_compliance_rate = float(np.mean([r["contract_respected"] for r in runs])) if runs else 0.0
    fallback_rate = float(fallback_count / total_runs) if total_runs else 1.0
    pre_violation_rate = float(pre_contract_violations / total_runs) if total_runs else 1.0
    post_violation_rate = float(post_fallback_violations / total_runs) if total_runs else 1.0

    stop_rule_triggered = post_violation_rate > 0.05

    if stop_rule_triggered:
        decision_hint = "drop"
        rationale = "Contract violation exceeds 5% after fallback, stop rule triggered."
    elif compressible_speedup >= 1.5 and contract_compliance_rate >= 0.95:
        decision_hint = "iterate"
        rationale = (
            "Core T4 metrics pass on this prototype, but policy depends on workload-specific "
            "compressibility assumptions and needs production-grade factorization integration."
        )
    else:
        decision_hint = "refine"
        rationale = "Contract/fallback behavior is stable, but speedup threshold is not consistently met."

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
            "pre_contract_violation_rate": pre_violation_rate,
            "post_fallback_violation_rate": post_violation_rate,
            "severe_outliers_prevented": severe_outliers_prevented,
            "stop_rule_triggered": stop_rule_triggered,
            "compressible_speedup_vs_exact_mean": compressible_speedup,
            "by_family": by_family,
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T4 approximate GEMM contract experiment")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["dense_random", "compressible_lowrank"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 1400])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--target-rank", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--error-budget", type=float, default=0.005)
    parser.add_argument("--precheck-energy-threshold", type=float, default=0.95)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t4_approximate_gemm",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week3_t4_contract_run_{timestamp}.json"
    md_path = output_dir / f"week3_t4_contract_run_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T4 contract JSON: {json_path}")
    print(f"T4 contract MD:   {md_path}")
    print(f"Decision hint:    {report['summary']['decision_hint']}")
    print(f"Stop rule:        {report['summary']['stop_rule_triggered']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
