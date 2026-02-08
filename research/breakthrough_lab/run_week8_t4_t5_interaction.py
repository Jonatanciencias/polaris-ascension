#!/usr/bin/env python3
"""Week 8 Block 6: realistic T4+T5 interaction campaign."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research/breakthrough_lab/t4_approximate_gemm"))

from run_t4_policy_gating import run_experiment as run_t4_experiment
from src.benchmarking.production_kernel_benchmark import run_production_benchmark


def _load_t4_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = ["policy_id", "scope", "activation_policy", "mixed_campaign"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"T4 policy missing required fields: {missing}")
    return data


def _run_t5_silent(
    *,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_production_benchmark(
            size=size,
            sessions=sessions,
            iterations=iterations,
            kernel="auto_t5_guarded",
            seed=seed,
        )


def _run_t4_silent(
    *,
    size: int,
    seed: int,
    policy: dict[str, Any],
) -> dict[str, Any]:
    activation = policy["activation_policy"]
    campaign = policy["mixed_campaign"]
    scope = policy["scope"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_t4_experiment(
            families=[str(x) for x in scope["families_tested"]],
            sizes=[int(size)],
            sessions=1,
            target_rank=int(activation["target_rank"]),
            noise_scale=float(campaign["noise_scale"]),
            error_budget=float(activation["error_budget"]),
            precheck_energy_threshold=float(
                activation["precheck_energy_threshold_exact_route"]
            ),
            sample_size=int(activation["sample_size"]),
            seed=seed,
        )


def _aggregate_t5(rows: list[dict[str, Any]]) -> dict[str, float]:
    def vals(key: str) -> list[float]:
        return [float(x[key]) for x in rows]

    return {
        "peak_mean_gflops": float(statistics.mean(vals("peak_mean_gflops"))),
        "avg_mean_gflops": float(statistics.mean(vals("avg_mean_gflops"))),
        "p95_time_ms": float(statistics.mean(vals("p95_time_ms"))),
        "max_error_max": float(max(vals("max_error_max"))),
        "abft_effective_overhead_percent": float(
            statistics.mean(vals("abft_effective_overhead_percent"))
        ),
        "abft_false_positive_rate": float(statistics.mean(vals("abft_false_positive_rate"))),
        "abft_disable_events": int(sum(int(x["abft_disable_events"]) for x in rows)),
    }


def _aggregate_t4(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "contract_compliance_rate": float(
            statistics.mean(float(x["contract_compliance_rate"]) for x in rows)
        ),
        "post_fallback_violation_rate": float(
            statistics.mean(float(x["post_fallback_violation_rate"]) for x in rows)
        ),
        "fallback_rate": float(statistics.mean(float(x["fallback_rate"]) for x in rows)),
        "compressible_speedup_vs_exact_mean": float(
            statistics.mean(float(x["compressible_speedup_vs_exact_mean"]) for x in rows)
        ),
        "delta_vs_exact_percent": float(
            statistics.mean(float(x["delta_vs_exact_percent"]) for x in rows)
        ),
        "executed_time_ms_p95": float(
            np.percentile(
                [float(v) for x in rows for v in x["executed_time_ms_samples"]],
                95,
            )
        ),
    }


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "t4_contract_compliance": {
            "observed": float(report["summary"]["t4_combined"]["contract_compliance_rate"]),
            "required_min": 0.99,
            "pass": float(report["summary"]["t4_combined"]["contract_compliance_rate"]) >= 0.99,
        },
        "t4_post_fallback_violations": {
            "observed": float(report["summary"]["t4_combined"]["post_fallback_violation_rate"]),
            "required_max": 0.0,
            "pass": float(report["summary"]["t4_combined"]["post_fallback_violation_rate"]) <= 0.0,
        },
        "t5_correctness_combined": {
            "observed": float(report["summary"]["t5_combined"]["max_error_max"]),
            "required_max": 1e-3,
            "pass": float(report["summary"]["t5_combined"]["max_error_max"]) <= 1e-3,
        },
        "t5_overhead_cross_delta": {
            "observed": float(report["summary"]["cross_effect"]["t5_overhead_delta_percent"]),
            "required_max": 0.30,
            "pass": float(report["summary"]["cross_effect"]["t5_overhead_delta_percent"]) <= 0.30,
        },
        "t5_p95_cross_delta": {
            "observed": float(report["summary"]["cross_effect"]["t5_p95_delta_percent"]),
            "required_max": 5.0,
            "pass": float(report["summary"]["cross_effect"]["t5_p95_delta_percent"]) <= 5.0,
        },
        "t5_avg_gflops_cross_drop": {
            "observed": float(report["summary"]["cross_effect"]["t5_avg_gflops_delta_percent"]),
            "required_min": -5.0,
            "pass": float(report["summary"]["cross_effect"]["t5_avg_gflops_delta_percent"]) >= -5.0,
        },
    }

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if not checks["t5_correctness_combined"]["pass"]:
        decision = "drop"
        rationale = "Cross-profile combined run violated correctness bounds."
    elif failed:
        decision = "iterate"
        rationale = (
            "Combined profile is functional but one or more cross-effect guardrails need tuning."
        )
    else:
        decision = "promote"
        rationale = (
            "Combined T4+T5 profile stayed within cross-effect overhead/latency bounds and preserved correctness."
        )
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    eval_data = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 8 Block 6 - T4+T5 Interaction Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']} | Seed={report['metadata']['seed']}"
    )
    lines.append("")
    lines.append("## T5 Baseline vs Combined")
    lines.append("")
    lines.append(
        f"- Baseline avg GFLOPS: {summary['t5_baseline']['avg_mean_gflops']:.3f}"
    )
    lines.append(
        f"- Combined avg GFLOPS: {summary['t5_combined']['avg_mean_gflops']:.3f}"
    )
    lines.append(f"- Baseline p95 ms: {summary['t5_baseline']['p95_time_ms']:.3f}")
    lines.append(f"- Combined p95 ms: {summary['t5_combined']['p95_time_ms']:.3f}")
    lines.append(
        f"- Overhead delta: {summary['cross_effect']['t5_overhead_delta_percent']:+.3f}%"
    )
    lines.append(
        f"- P95 delta: {summary['cross_effect']['t5_p95_delta_percent']:+.3f}%"
    )
    lines.append(
        f"- Avg GFLOPS delta: {summary['cross_effect']['t5_avg_gflops_delta_percent']:+.3f}%"
    )
    lines.append("")
    lines.append("## T4 Combined State")
    lines.append("")
    lines.append(
        f"- Contract compliance: {summary['t4_combined']['contract_compliance_rate']:.3f}"
    )
    lines.append(
        f"- Post-fallback violation rate: {summary['t4_combined']['post_fallback_violation_rate']:.3f}"
    )
    lines.append(
        f"- Fallback rate: {summary['t4_combined']['fallback_rate']:.3f}"
    )
    lines.append("")
    lines.append("## Guardrail Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in eval_data["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{eval_data['decision']}`")
    lines.append(f"- Rationale: {eval_data['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    t4_policy_path: Path,
    sizes: list[int],
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    t4_policy = _load_t4_policy(t4_policy_path)

    t5_baseline_rows: list[dict[str, Any]] = []
    t5_combined_rows: list[dict[str, Any]] = []
    t4_combined_rows: list[dict[str, Any]] = []

    for size_idx, size in enumerate(sizes):
        baseline = _run_t5_silent(
            size=int(size),
            sessions=int(sessions),
            iterations=int(iterations),
            seed=int(seed + size_idx * 10000),
        )
        bsum = baseline["summary"]
        bt5 = bsum["t5_abft"]
        t5_baseline_rows.append(
            {
                "size": int(size),
                "peak_mean_gflops": float(bsum["peak_gflops"]["mean"]),
                "avg_mean_gflops": float(bsum["avg_gflops"]["mean"]),
                "p95_time_ms": float(bsum["time_ms"]["p95"]),
                "max_error_max": float(bsum["max_error"]["max"]),
                "abft_effective_overhead_percent": float(bt5["effective_overhead_percent"]),
                "abft_false_positive_rate": float(bt5["false_positive_rate"]),
                "abft_disable_events": int(bt5["disable_events"]),
            }
        )

        for s in range(sessions):
            run_seed = int(seed + size_idx * 100000 + s * 1000)
            t4 = _run_t4_silent(size=int(size), seed=run_seed, policy=t4_policy)
            t4_summary = t4["summary"]
            executed_time_samples = [float(x["executed_time_ms"]) for x in t4["runs"]]
            t4_combined_rows.append(
                {
                    "size": int(size),
                    "session": int(s),
                    "contract_compliance_rate": float(t4_summary["contract_compliance_rate"]),
                    "post_fallback_violation_rate": float(
                        t4_summary["post_fallback_violation_rate"]
                    ),
                    "fallback_rate": float(t4_summary["fallback_rate"]),
                    "compressible_speedup_vs_exact_mean": float(
                        t4_summary["compressible_speedup_vs_exact_mean"]
                    ),
                    "delta_vs_exact_percent": float(t4_summary["delta_vs_exact_percent"]),
                    "executed_time_ms_samples": executed_time_samples,
                }
            )

            t5 = _run_t5_silent(
                size=int(size),
                sessions=1,
                iterations=int(iterations),
                seed=int(seed + size_idx * 1000000 + s * 101),
            )
            ssum = t5["summary"]
            st5 = ssum["t5_abft"]
            t5_combined_rows.append(
                {
                    "size": int(size),
                    "session": int(s),
                    "peak_mean_gflops": float(ssum["peak_gflops"]["mean"]),
                    "avg_mean_gflops": float(ssum["avg_gflops"]["mean"]),
                    "p95_time_ms": float(ssum["time_ms"]["p95"]),
                    "max_error_max": float(ssum["max_error"]["max"]),
                    "abft_effective_overhead_percent": float(st5["effective_overhead_percent"]),
                    "abft_false_positive_rate": float(st5["false_positive_rate"]),
                    "abft_disable_events": int(st5["disable_events"]),
                }
            )

    t5_baseline = _aggregate_t5(t5_baseline_rows)
    t5_combined = _aggregate_t5(t5_combined_rows)
    t4_combined = _aggregate_t4(t4_combined_rows)

    cross = {
        "t5_overhead_delta_percent": float(
            t5_combined["abft_effective_overhead_percent"]
            - t5_baseline["abft_effective_overhead_percent"]
        ),
        "t5_p95_delta_percent": float(
            ((t5_combined["p95_time_ms"] - t5_baseline["p95_time_ms"]) / t5_baseline["p95_time_ms"])
            * 100.0
        )
        if t5_baseline["p95_time_ms"] > 0
        else 0.0,
        "t5_avg_gflops_delta_percent": float(
            ((t5_combined["avg_mean_gflops"] - t5_baseline["avg_mean_gflops"])
            / t5_baseline["avg_mean_gflops"])
            * 100.0
        )
        if t5_baseline["avg_mean_gflops"] > 0
        else 0.0,
    }

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": [int(x) for x in sizes],
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
            "t4_policy_path": str(t4_policy_path),
            "t4_policy_id": str(t4_policy["policy_id"]),
        },
        "raw": {
            "t5_baseline_rows": t5_baseline_rows,
            "t4_combined_rows": t4_combined_rows,
            "t5_combined_rows": t5_combined_rows,
        },
        "summary": {
            "t5_baseline": t5_baseline,
            "t4_combined": t4_combined,
            "t5_combined": t5_combined,
            "cross_effect": cross,
        },
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week8 T4+T5 interaction campaign.")
    parser.add_argument(
        "--t4-policy-path",
        default="research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="research/breakthrough_lab")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t4_policy_path = (REPO_ROOT / args.t4_policy_path).resolve()

    report = run_campaign(
        t4_policy_path=t4_policy_path,
        sizes=list(args.sizes),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_block6_t4_t5_interaction_{timestamp}.json"
    md_path = output_dir / f"week8_block6_t4_t5_interaction_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"T4+T5 interaction JSON: {json_path}")
    print(f"T4+T5 interaction MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(
        "Cross overhead delta (%): "
        f"{report['summary']['cross_effect']['t5_overhead_delta_percent']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
