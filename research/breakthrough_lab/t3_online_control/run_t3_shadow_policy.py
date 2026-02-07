#!/usr/bin/env python3
"""
Week 3 T3 shadow-policy runner.

Block 3 redesign:
- bootstrap priors from strict T2 boundary evidence
- policy guardrails (allowed arms by size, forced-static sizes)
- strict rerun with deterministic workload

Compares static production selector vs contextual online policy in shadow mode.
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

CONFIG_TO_ARM = {
    "t20_prod_v4_u10_l10": "tile20",
    "t20_v3vec_v4_u0_l10": "tile20_v3_1400",
    "t24_prod_v4_u0_l12": "tile24",
}


class GuardedContextualBandit:
    def __init__(
        self,
        *,
        arms: list[str],
        epsilon: float,
        seed: int,
        bootstrap_weight: float,
        warmup_steps_per_context: int,
    ):
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        self.bootstrap_weight = float(bootstrap_weight)
        self.warmup_steps_per_context = int(warmup_steps_per_context)
        self.rng = np.random.default_rng(seed)

        self.stats: dict[str, dict[str, dict[str, float]]] = {}
        self.context_steps: dict[str, int] = {}

    def _ensure_context(self, context: str, arm_priors: dict[str, float] | None = None) -> None:
        if context in self.stats:
            return

        self.stats[context] = {
            arm: {"count": 0.0, "reward_sum": 0.0}
            for arm in self.arms
        }
        self.context_steps[context] = 0

        if not arm_priors:
            return

        for arm, prior_reward in arm_priors.items():
            if arm not in self.stats[context]:
                continue
            self.stats[context][arm]["count"] += self.bootstrap_weight
            self.stats[context][arm]["reward_sum"] += self.bootstrap_weight * float(prior_reward)

    def _mean_reward(self, context: str, arm: str) -> float:
        v = self.stats[context][arm]
        if v["count"] <= 0:
            return -1e12
        return float(v["reward_sum"] / v["count"])

    def choose(
        self,
        *,
        context: str,
        static_arm: str,
        allowed_arms: list[str],
        force_static: bool,
        arm_priors: dict[str, float] | None,
    ) -> tuple[str, bool, str]:
        self._ensure_context(context, arm_priors=arm_priors)

        if static_arm not in allowed_arms:
            allowed_arms = list(dict.fromkeys([static_arm] + allowed_arms))

        if force_static:
            return static_arm, False, "force_static_guardrail"

        if self.context_steps[context] < self.warmup_steps_per_context:
            best_warm = max(allowed_arms, key=lambda arm: (self._mean_reward(context, arm), -self.arms.index(arm)))
            return best_warm, False, "bootstrap_warmup"

        if len(allowed_arms) > 1 and float(self.rng.random()) < self.epsilon:
            return str(self.rng.choice(allowed_arms)), True, "epsilon_exploration"

        best = max(allowed_arms, key=lambda arm: (self._mean_reward(context, arm), -self.arms.index(arm)))
        return best, False, "policy_exploitation"

    def update(self, *, context: str, arm: str, reward: float) -> None:
        self._ensure_context(context, arm_priors=None)
        self.stats[context][arm]["count"] += 1.0
        self.stats[context][arm]["reward_sum"] += float(reward)
        self.context_steps[context] += 1

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for context, arm_stats in self.stats.items():
            out[context] = {
                "steps": self.context_steps.get(context, 0),
                "arms": {},
            }
            for arm, values in arm_stats.items():
                count = float(values["count"])
                mean_reward = float(values["reward_sum"] / count) if count > 0 else 0.0
                out[context]["arms"][arm] = {
                    "count": int(round(count)),
                    "mean_reward": mean_reward,
                }
        return out


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


def _context_from_size(size: int) -> str:
    return f"size_{size}"


def _load_boundary_priors(
    *,
    report_path: Path,
    min_delta_for_nonstatic: float,
) -> tuple[dict[int, dict[str, dict[str, float]]], str]:
    if not report_path.exists():
        return {}, f"missing:{report_path}"

    data = json.loads(report_path.read_text())
    ranked = data.get("search", {}).get("ranked_candidates", [])
    priors: dict[int, dict[str, dict[str, float]]] = {}

    for item in ranked:
        config_id = item.get("config_id")
        arm = CONFIG_TO_ARM.get(config_id)
        if arm is None:
            continue

        size = int(item["size"])
        gflops = float(item["gflops"])
        delta = float(item.get("delta_vs_baseline_percent") or 0.0)
        max_error = float(item.get("max_error") or 1.0)
        correctness = bool(item.get("correctness_passed", False))

        if size not in priors:
            priors[size] = {}

        current = priors[size].get(arm)
        if current is None or gflops > current["gflops"]:
            priors[size][arm] = {
                "gflops": gflops,
                "delta_vs_baseline_percent": delta,
                "max_error": max_error,
                "correctness_passed": float(1.0 if correctness else 0.0),
            }

    # Guardrail cleanup: if non-static arm does not clear minimum expected uplift,
    # keep it only as prior info but mark as not-eligible during allowlist building.
    for size_data in priors.values():
        for arm_data in size_data.values():
            arm_data["eligible_nonstatic"] = float(
                1.0 if arm_data["delta_vs_baseline_percent"] >= min_delta_for_nonstatic else 0.0
            )

    return priors, str(report_path)


def _allowed_arms_for_step(
    *,
    size: int,
    static_arm: str,
    priors_by_size: dict[int, dict[str, dict[str, float]]],
    min_delta_for_nonstatic: float,
    freeze_sizes: set[int],
) -> list[str]:
    allowed = {static_arm}

    if size in freeze_sizes:
        return sorted(allowed)

    size_priors = priors_by_size.get(size, {})
    for arm, arm_data in size_priors.items():
        if arm == static_arm:
            continue
        if arm_data["delta_vs_baseline_percent"] < min_delta_for_nonstatic:
            continue
        if arm_data["correctness_passed"] < 1.0:
            continue
        allowed.add(arm)

    return sorted(allowed)


def _per_size_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, list[float]]] = {}
    for d in decisions:
        if d.get("status") != "completed":
            continue
        size_key = str(d["size"])
        if size_key not in out:
            out[size_key] = {"static": [], "shadow": []}
        out[size_key]["static"].append(float(d["static_gflops"]))
        out[size_key]["shadow"].append(float(d["executed_shadow_gflops"]))

    summarized: dict[str, Any] = {}
    for size_key, vals in out.items():
        static_mean = float(np.mean(vals["static"])) if vals["static"] else 0.0
        shadow_mean = float(np.mean(vals["shadow"])) if vals["shadow"] else 0.0
        delta = ((shadow_mean - static_mean) / static_mean * 100.0) if static_mean > 0 else 0.0
        summarized[size_key] = {
            "static_mean_gflops": static_mean,
            "shadow_mean_gflops": shadow_mean,
            "delta_vs_static_percent": float(delta),
            "samples": len(vals["shadow"]),
        }
    return summarized


def _fallback_reason_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in decisions:
        if d.get("status") != "completed":
            continue
        if not d.get("fallback_triggered"):
            continue
        reason = str(d.get("fallback_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T3 Week 3 Block 3 - Strict Shadow Policy Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Protocol: epochs={report['metadata']['epochs']}, runs_per_decision={report['metadata']['runs_per_decision']}, warmup={report['metadata']['warmup']}, seed={report['metadata']['seed']}"
    )
    lines.append(
        f"- Policy: epsilon={report['metadata']['epsilon']}, bootstrap_weight={report['metadata']['bootstrap_weight']}, warmup_steps_per_context={report['metadata']['warmup_steps_per_context']}"
    )
    lines.append(
        f"- Guardrails: regression_limit={report['metadata']['fallback_regression_limit']}, max_fallback_rate={report['metadata']['max_fallback_rate']}, min_delta_for_nonstatic={report['metadata']['min_delta_for_nonstatic']}, freeze_sizes={report['metadata']['freeze_sizes']}"
    )
    lines.append(f"- Bootstrap source: {report['metadata']['bootstrap_source']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Static mean GFLOPS: {report['summary']['static_gflops']['mean']:.3f}")
    lines.append(f"- Shadow mean GFLOPS: {report['summary']['shadow_gflops']['mean']:.3f}")
    lines.append(f"- Mean uplift vs static: {report['summary']['delta_vs_static_percent']:+.3f}%")
    lines.append(f"- P95 latency delta vs static: {report['summary']['p95_latency_delta_percent']:+.3f}%")
    lines.append(f"- Fallback rate: {report['summary']['fallback_rate']:.3f}")
    lines.append(f"- Exploration rate: {report['summary']['exploration_rate']:.3f}")
    lines.append(f"- Correctness failures: {report['summary']['correctness_failures']}")
    lines.append(f"- Stop rule triggered: {report['summary']['stop_rule_triggered']}")
    lines.append(f"- Decision hint: {report['summary']['decision_hint']}")
    lines.append("")
    lines.append("## Per-Size Delta")
    lines.append("")
    lines.append("| Size | Static Mean | Shadow Mean | Delta | Samples |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for size_key, data in sorted(report["summary"]["per_size"].items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {size_key} | {data['static_mean_gflops']:.3f} | {data['shadow_mean_gflops']:.3f} | {data['delta_vs_static_percent']:+.3f}% | {data['samples']} |"
        )
    lines.append("")
    lines.append("## Fallback Reasons")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["summary"]["fallback_reason_counts"], indent=2))
    lines.append("```")
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
    bootstrap_weight: float,
    warmup_steps_per_context: int,
    min_delta_for_nonstatic: float,
    freeze_sizes: list[int],
    boundary_report_path: str,
) -> dict[str, Any]:
    selector = ProductionKernelSelector()
    tuner = GEMMAutoTuner(output_dir="results/auto_tuner", verbose=False)

    priors_by_size, bootstrap_source = _load_boundary_priors(
        report_path=Path(boundary_report_path),
        min_delta_for_nonstatic=min_delta_for_nonstatic,
    )

    policy = GuardedContextualBandit(
        arms=["tile20", "tile20_v3_1400", "tile24"],
        epsilon=epsilon,
        seed=seed,
        bootstrap_weight=bootstrap_weight,
        warmup_steps_per_context=warmup_steps_per_context,
    )

    freeze_set = set(int(x) for x in freeze_sizes)

    workload: list[int] = []
    for epoch in range(epochs):
        # Strict deterministic rotation.
        shift = epoch % len(sizes)
        rotated = sizes[shift:] + sizes[:shift]
        workload.extend(rotated)

    decisions: list[dict[str, Any]] = []
    static_gflops: list[float] = []
    static_time_ms: list[float] = []
    shadow_gflops: list[float] = []
    shadow_time_ms: list[float] = []

    fallback_count = 0
    explored_count = 0
    correctness_failures = 0
    stop_rule_triggered = False
    stop_reason = None

    for step_idx, size in enumerate(workload):
        context = _context_from_size(size)
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

        allowed_arms = _allowed_arms_for_step(
            size=size,
            static_arm=static_arm,
            priors_by_size=priors_by_size,
            min_delta_for_nonstatic=min_delta_for_nonstatic,
            freeze_sizes=freeze_set,
        )
        arm_priors = {
            arm: priors_by_size.get(size, {}).get(arm, {}).get("gflops", 0.0)
            for arm in allowed_arms
        }

        online_arm, explored, selection_reason = policy.choose(
            context=context,
            static_arm=static_arm,
            allowed_arms=allowed_arms,
            force_static=(size in freeze_set),
            arm_priors=arm_priors,
        )
        if explored:
            explored_count += 1

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
                "allowed_arms": allowed_arms,
                "online_arm": online_arm,
                "selection_reason": selection_reason,
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
                "priors_gflops": arm_priors,
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
                "bootstrap_weight": bootstrap_weight,
                "warmup_steps_per_context": warmup_steps_per_context,
                "min_delta_for_nonstatic": min_delta_for_nonstatic,
                "freeze_sizes": sorted(freeze_set),
                "bootstrap_source": bootstrap_source,
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
                "exploration_rate": 0.0,
                "fallback_reason_counts": {},
                "per_size": {},
                "promotion_gate_passed": False,
            },
        }

    static_mean = float(np.mean(static_gflops))
    shadow_mean = float(np.mean(shadow_gflops))
    delta_vs_static = ((shadow_mean - static_mean) / static_mean * 100.0) if static_mean > 0 else 0.0
    static_p95 = float(np.percentile(np.array(static_time_ms, dtype=float), 95))
    shadow_p95 = float(np.percentile(np.array(shadow_time_ms, dtype=float), 95))
    p95_latency_delta = ((shadow_p95 - static_p95) / static_p95 * 100.0) if static_p95 > 0 else 0.0
    fallback_rate = fallback_count / max(1, len(shadow_gflops))
    exploration_rate = explored_count / max(1, len(shadow_gflops))

    correctness_passed = correctness_failures == 0

    promotion_gate_passed = (
        not stop_rule_triggered
        and delta_vs_static >= 5.0
        and p95_latency_delta <= 3.0
        and correctness_passed
        and fallback_rate <= max_fallback_rate
    )

    if stop_rule_triggered:
        decision_hint = "drop"
        rationale = f"Stop rule triggered: {stop_reason}."
    elif promotion_gate_passed:
        decision_hint = "promote"
        rationale = "Strict rerun met uplift, latency, correctness and fallback gates."
    else:
        decision_hint = "iterate"
        rationale = "Strict rerun is safe but did not satisfy full promotion gate."

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
            "bootstrap_weight": bootstrap_weight,
            "warmup_steps_per_context": warmup_steps_per_context,
            "min_delta_for_nonstatic": min_delta_for_nonstatic,
            "freeze_sizes": sorted(freeze_set),
            "bootstrap_source": bootstrap_source,
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
            "correctness_failures": correctness_failures,
            "correctness_passed": correctness_passed,
            "fallback_rate": float(fallback_rate),
            "exploration_rate": float(exploration_rate),
            "stop_rule_triggered": stop_rule_triggered,
            "stop_reason": stop_reason,
            "fallback_reason_counts": _fallback_reason_counts(decisions),
            "per_size": _per_size_summary(decisions),
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
            "promotion_gate_passed": promotion_gate_passed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T3 strict shadow online-policy experiment")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1200, 1280, 1400, 1536, 1600, 1792, 1920, 2048])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--runs-per-decision", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--fallback-regression-limit", type=float, default=0.08)
    parser.add_argument("--max-fallback-rate", type=float, default=0.10)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-weight", type=float, default=2.0)
    parser.add_argument("--warmup-steps-per-context", type=int, default=2)
    parser.add_argument("--min-delta-for-nonstatic", type=float, default=8.0)
    parser.add_argument("--freeze-sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument(
        "--boundary-report-path",
        type=str,
        default="research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json",
    )
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
        bootstrap_weight=args.bootstrap_weight,
        warmup_steps_per_context=args.warmup_steps_per_context,
        min_delta_for_nonstatic=args.min_delta_for_nonstatic,
        freeze_sizes=args.freeze_sizes,
        boundary_report_path=args.boundary_report_path,
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
