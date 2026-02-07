#!/usr/bin/env python3
"""
Week 3 T3 shadow-policy runner.

Compares static production selector vs a contextual online policy in shadow mode.
No production behavior is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.auto_tuner.gemm_auto_tuner import GEMMAutoTuner
from src.optimization_engines.adaptive_kernel_selector import ProductionKernelSelector


@dataclass(frozen=True)
class KernelSpec:
    kernel_file: str
    kernel_name: str
    tile_size: int
    local_size: tuple[int, int]


KERNEL_SPECS: dict[str, KernelSpec] = {
    "tile20": KernelSpec(
        kernel_file="src/kernels/gemm_tile20_production.cl",
        kernel_name="gemm_tile20_optimized",
        tile_size=20,
        local_size=(10, 10),
    ),
    "tile20_v3_1400": KernelSpec(
        kernel_file="src/kernels/gemm_tile20_v3_vectorized.cl",
        kernel_name="gemm_tile20_vectorized",
        tile_size=20,
        local_size=(10, 10),
    ),
    "tile24": KernelSpec(
        kernel_file="src/kernels/gemm_tile24_production.cl",
        kernel_name="gemm_tile24_vectorized",
        tile_size=24,
        local_size=(12, 12),
    ),
}


class ContextualEpsilonGreedy:
    def __init__(self, arms: list[str], epsilon: float, seed: int):
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.stats: dict[str, dict[str, dict[str, float]]] = {}

    def _ensure_context(self, context: str) -> None:
        if context not in self.stats:
            self.stats[context] = {
                arm: {"count": 0.0, "reward_sum": 0.0}
                for arm in self.arms
            }

    def choose(self, context: str, static_arm: str) -> tuple[str, bool]:
        self._ensure_context(context)
        context_stats = self.stats[context]

        if float(self.rng.random()) < self.epsilon:
            return str(self.rng.choice(self.arms)), True

        observed = [arm for arm in self.arms if context_stats[arm]["count"] > 0]
        if not observed:
            return static_arm, False

        best_arm = max(
            self.arms,
            key=lambda arm: (
                (context_stats[arm]["reward_sum"] / context_stats[arm]["count"])
                if context_stats[arm]["count"] > 0
                else -1e12,
                -self.arms.index(arm),
            ),
        )
        return best_arm, False

    def update(self, context: str, arm: str, reward: float) -> None:
        self._ensure_context(context)
        self.stats[context][arm]["count"] += 1.0
        self.stats[context][arm]["reward_sum"] += float(reward)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for context, arm_stats in self.stats.items():
            out[context] = {}
            for arm, values in arm_stats.items():
                count = float(values["count"])
                mean_reward = float(values["reward_sum"] / count) if count > 0 else 0.0
                out[context][arm] = {
                    "count": int(count),
                    "mean_reward": mean_reward,
                }
        return out


def _context_bucket(size: int) -> str:
    if size <= 1400:
        return "small"
    if size <= 1920:
        return "mid"
    return "large"


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


def _benchmark_arm(
    *,
    tuner: GEMMAutoTuner,
    arm: str,
    size: int,
    runs: int,
    warmup: int,
    seed: int,
) -> Any | None:
    spec = KERNEL_SPECS[arm]
    return tuner.benchmark_custom_kernel(
        kernel_file=spec.kernel_file,
        kernel_name=spec.kernel_name,
        tile_size=spec.tile_size,
        local_size=spec.local_size,
        M=size,
        N=size,
        K=size,
        runs=runs,
        warmup=warmup,
        seed=seed,
        input_distribution="standard_normal",
    )


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T3 Week 3 Shadow Policy Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Protocol: epochs={report['metadata']['epochs']}, runs_per_decision={report['metadata']['runs_per_decision']}, warmup={report['metadata']['warmup']}, seed={report['metadata']['seed']}"
    )
    lines.append(
        f"- Epsilon: {report['metadata']['epsilon']}, fallback_regression_limit={report['metadata']['fallback_regression_limit']}, max_fallback_rate={report['metadata']['max_fallback_rate']}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Static mean GFLOPS: {report['summary']['static_gflops']['mean']:.3f}"
    )
    lines.append(
        f"- Shadow executed mean GFLOPS: {report['summary']['shadow_gflops']['mean']:.3f}"
    )
    lines.append(
        f"- Mean uplift vs static: {report['summary']['delta_vs_static_percent']:+.3f}%"
    )
    lines.append(
        f"- P95 latency delta vs static: {report['summary']['p95_latency_delta_percent']:+.3f}%"
    )
    lines.append(
        f"- Fallback rate: {report['summary']['fallback_rate']:.3f}"
    )
    lines.append(
        f"- Correctness failures: {report['summary']['correctness_failures']}"
    )
    lines.append(
        f"- Stop rule triggered: {report['summary']['stop_rule_triggered']}"
    )
    lines.append(
        f"- Decision hint: {report['summary']['decision_hint']}"
    )
    lines.append("")
    lines.append("## Policy Snapshot")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["policy_snapshot"], indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_shadow_experiment(
    *,
    sizes: list[int],
    epochs: int,
    runs_per_decision: int,
    warmup: int,
    epsilon: float,
    fallback_regression_limit: float,
    max_fallback_rate: float,
    correctness_threshold: float,
    seed: int,
) -> dict[str, Any]:
    selector = ProductionKernelSelector()
    tuner = GEMMAutoTuner(output_dir="results/auto_tuner", verbose=False)
    policy = ContextualEpsilonGreedy(
        arms=["tile20", "tile20_v3_1400", "tile24"],
        epsilon=epsilon,
        seed=seed,
    )

    workload: list[int] = []
    for epoch in range(epochs):
        # Rotate sizes each epoch to simulate non-stationary mix but remain deterministic.
        shift = epoch % len(sizes)
        rotated = sizes[shift:] + sizes[:shift]
        workload.extend(rotated)

    decisions: list[dict[str, Any]] = []
    static_gflops: list[float] = []
    static_time_ms: list[float] = []
    shadow_gflops: list[float] = []
    shadow_time_ms: list[float] = []

    fallback_count = 0
    correctness_failures = 0
    stop_rule_triggered = False
    stop_reason = None

    for step_idx, size in enumerate(workload):
        context = _context_bucket(size)
        static_rec = selector.select_kernel(size, size, size)
        static_arm = static_rec["kernel_key"]

        static_result = _benchmark_arm(
            tuner=tuner,
            arm=static_arm,
            size=size,
            runs=runs_per_decision,
            warmup=warmup,
            seed=seed + step_idx * 100 + 1,
        )
        if static_result is None:
            decisions.append(
                {
                    "step": step_idx,
                    "size": size,
                    "context": context,
                    "status": "failed",
                    "error": "static benchmark failed",
                }
            )
            continue

        online_arm, explored = policy.choose(context=context, static_arm=static_arm)

        if online_arm == static_arm:
            online_result = static_result
        else:
            online_result = _benchmark_arm(
                tuner=tuner,
                arm=online_arm,
                size=size,
                runs=runs_per_decision,
                warmup=warmup,
                seed=seed + step_idx * 100 + 2,
            )
            if online_result is None:
                online_result = static_result

        correctness_failed = online_result.max_error > correctness_threshold
        if correctness_failed:
            correctness_failures += 1

        regress_guard = online_result.gflops < (
            static_result.gflops * (1.0 - fallback_regression_limit)
        )
        fallback_triggered = bool(correctness_failed or regress_guard)

        if fallback_triggered:
            fallback_count += 1
            executed_gflops = float(static_result.gflops)
            executed_time_ms = float(static_result.avg_time_ms)
            fallback_reason = "correctness" if correctness_failed else "regression_guard"
        else:
            executed_gflops = float(online_result.gflops)
            executed_time_ms = float(online_result.avg_time_ms)
            fallback_reason = None

        policy.update(context=context, arm=online_arm, reward=executed_gflops)

        static_gflops.append(float(static_result.gflops))
        static_time_ms.append(float(static_result.avg_time_ms))
        shadow_gflops.append(float(executed_gflops))
        shadow_time_ms.append(float(executed_time_ms))

        decisions.append(
            {
                "step": step_idx,
                "size": size,
                "context": context,
                "status": "completed",
                "static_arm": static_arm,
                "online_arm": online_arm,
                "explored": explored,
                "static_gflops": float(static_result.gflops),
                "online_gflops": float(online_result.gflops),
                "executed_shadow_gflops": float(executed_gflops),
                "static_time_ms": float(static_result.avg_time_ms),
                "online_time_ms": float(online_result.avg_time_ms),
                "executed_shadow_time_ms": float(executed_time_ms),
                "online_max_error": float(online_result.max_error),
                "fallback_triggered": fallback_triggered,
                "fallback_reason": fallback_reason,
            }
        )

        fallback_rate = fallback_count / max(1, len(shadow_gflops))
        if correctness_failures > 0:
            stop_rule_triggered = True
            stop_reason = "correctness gate failed"
            break
        if fallback_rate > max_fallback_rate:
            stop_rule_triggered = True
            stop_reason = (
                f"fallback rate exceeded threshold ({fallback_rate:.3f} > {max_fallback_rate:.3f})"
            )
            break

    if not shadow_gflops:
        return {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sizes": sizes,
                "epochs": epochs,
                "runs_per_decision": runs_per_decision,
                "warmup": warmup,
                "epsilon": epsilon,
                "fallback_regression_limit": fallback_regression_limit,
                "max_fallback_rate": max_fallback_rate,
                "correctness_threshold": correctness_threshold,
                "seed": seed,
            },
            "decisions": decisions,
            "policy_snapshot": policy.snapshot(),
            "summary": {
                "decision_hint": "refine",
                "decision_rationale": "No completed shadow runs.",
                "stop_rule_triggered": True,
                "stop_reason": "no_completed_runs",
                "correctness_failures": correctness_failures,
                "fallback_rate": 1.0,
            },
        }

    static_mean = float(np.mean(static_gflops))
    shadow_mean = float(np.mean(shadow_gflops))
    delta_vs_static = ((shadow_mean - static_mean) / static_mean * 100.0) if static_mean > 0 else 0.0
    static_p95 = float(np.percentile(np.array(static_time_ms, dtype=float), 95))
    shadow_p95 = float(np.percentile(np.array(shadow_time_ms, dtype=float), 95))
    p95_latency_delta = ((shadow_p95 - static_p95) / static_p95 * 100.0) if static_p95 > 0 else 0.0
    fallback_rate = fallback_count / max(1, len(shadow_gflops))
    cv_shadow = float(np.std(shadow_gflops) / np.mean(shadow_gflops)) if np.mean(shadow_gflops) > 0 else 1.0

    correctness_passed = correctness_failures == 0
    stability_passed = cv_shadow <= 0.03
    promotion_gate_passed = (
        delta_vs_static >= 5.0
        and p95_latency_delta <= 3.0
        and correctness_passed
        and fallback_rate <= max_fallback_rate
    )

    if stop_rule_triggered:
        decision_hint = "drop"
        rationale = f"Stop rule triggered: {stop_reason}."
    elif promotion_gate_passed:
        decision_hint = "promote"
        rationale = "Shadow policy met uplift, latency, correctness and fallback gates."
    else:
        decision_hint = "iterate"
        rationale = "Shadow policy is stable/correct but did not meet full promotion thresholds."

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": sizes,
            "epochs": epochs,
            "runs_per_decision": runs_per_decision,
            "warmup": warmup,
            "epsilon": epsilon,
            "fallback_regression_limit": fallback_regression_limit,
            "max_fallback_rate": max_fallback_rate,
            "correctness_threshold": correctness_threshold,
            "seed": seed,
            "total_steps_planned": len(workload),
            "total_steps_executed": len(shadow_gflops),
        },
        "decisions": decisions,
        "policy_snapshot": policy.snapshot(),
        "summary": {
            "static_gflops": _stats(static_gflops),
            "shadow_gflops": _stats(shadow_gflops),
            "static_time_ms": _stats(static_time_ms),
            "shadow_time_ms": _stats(shadow_time_ms),
            "delta_vs_static_percent": float(delta_vs_static),
            "p95_latency_delta_percent": float(p95_latency_delta),
            "cv_shadow": cv_shadow,
            "correctness_failures": correctness_failures,
            "correctness_passed": correctness_passed,
            "stability_passed": stability_passed,
            "fallback_rate": float(fallback_rate),
            "stop_rule_triggered": stop_rule_triggered,
            "stop_reason": stop_reason,
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
            "promotion_gate_passed": promotion_gate_passed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T3 shadow online-policy experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1200, 1280, 1400, 1536, 1600, 1792, 1920, 2048])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--runs-per-decision", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--fallback-regression-limit", type=float, default=0.10)
    parser.add_argument("--max-fallback-rate", type=float, default=0.20)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t3_online_control",
    )
    args = parser.parse_args()

    report = run_shadow_experiment(
        sizes=args.sizes,
        epochs=args.epochs,
        runs_per_decision=args.runs_per_decision,
        warmup=args.warmup,
        epsilon=args.epsilon,
        fallback_regression_limit=args.fallback_regression_limit,
        max_fallback_rate=args.max_fallback_rate,
        correctness_threshold=args.correctness_threshold,
        seed=args.seed,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week3_t3_shadow_policy_{timestamp}.json"
    md_path = output_dir / f"week3_t3_shadow_policy_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T3 shadow JSON: {json_path}")
    print(f"T3 shadow MD:   {md_path}")
    print(f"Decision hint:  {report['summary']['decision_hint']}")
    print(f"Stop rule:      {report['summary']['stop_rule_triggered']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
