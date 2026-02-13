"""Controlled rollout policy for T3 online selector integration.

This module keeps the production selector deterministic by default, and enables
an opt-in contextual policy with strict guardrails for controlled campaigns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_TO_ARM = {
    "t20_prod_v4_u10_l10": "tile20",
    "t20_v3vec_v4_u0_l10": "tile20_v3_1400",
    "t24_prod_v4_u0_l12": "tile24",
}

DEFAULT_POLICY_PATH = "research/breakthrough_lab/t3_online_control/policy_controlled_block1.json"
DEFAULT_BOUNDARY_REPORT = (
    "research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json"
)


class ControlledT3Policy:
    """Contextual policy with bootstrap priors and strict fallback guards."""

    def __init__(
        self,
        *,
        seed: int = 42,
        epsilon: float = 0.05,
        bootstrap_weight: float = 3.0,
        warmup_steps_per_context: int = 2,
        min_delta_for_nonstatic: float = 8.0,
        freeze_sizes: list[int] | None = None,
        fallback_regression_limit: float = 0.08,
        max_fallback_rate: float = 0.10,
        correctness_threshold: float = 1e-3,
        boundary_report_path: str = DEFAULT_BOUNDARY_REPORT,
        priors_by_size: dict[int, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.bootstrap_weight = float(bootstrap_weight)
        self.warmup_steps_per_context = int(warmup_steps_per_context)
        self.min_delta_for_nonstatic = float(min_delta_for_nonstatic)
        self.freeze_sizes = {int(x) for x in (freeze_sizes or [1400, 2048])}
        self.fallback_regression_limit = float(fallback_regression_limit)
        self.max_fallback_rate = float(max_fallback_rate)
        self.correctness_threshold = float(correctness_threshold)
        self.boundary_report_path = str(boundary_report_path)

        self.rng = np.random.default_rng(self.seed)
        self.stats: dict[str, dict[str, dict[str, float]]] = {}
        self.context_steps: dict[str, int] = {}

        if priors_by_size is not None:
            self.priors_by_size = priors_by_size
            self.bootstrap_source = "in_memory"
        else:
            self.priors_by_size, self.bootstrap_source = self._load_boundary_priors(
                report_path=Path(self.boundary_report_path),
                min_delta_for_nonstatic=self.min_delta_for_nonstatic,
            )

        self.total_feedback = 0
        self.fallback_count = 0
        self.correctness_failures = 0
        self.disabled = False
        self.disable_reason: str | None = None

    @classmethod
    def from_policy_file(
        cls,
        *,
        policy_path: str | Path | None = None,
        seed: int = 42,
    ) -> "ControlledT3Policy":
        policy_obj = Path(policy_path) if policy_path else Path(DEFAULT_POLICY_PATH)
        cfg = cls._parse_policy_file(policy_obj)
        cfg["seed"] = int(seed)
        return cls(**cfg)

    @staticmethod
    def _parse_policy_file(policy_path: Path) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "epsilon": 0.05,
            "bootstrap_weight": 3.0,
            "warmup_steps_per_context": 2,
            "min_delta_for_nonstatic": 8.0,
            "freeze_sizes": [1400, 2048],
            "fallback_regression_limit": 0.08,
            "max_fallback_rate": 0.10,
            "correctness_threshold": 1e-3,
            "boundary_report_path": DEFAULT_BOUNDARY_REPORT,
        }

        if not policy_path.exists():
            return defaults

        data = json.loads(policy_path.read_text())
        selector_cfg = data.get("selector_policy", {})
        guardrails = data.get("guardrails", {})
        bootstrap = data.get("bootstrap", {})

        defaults["epsilon"] = float(selector_cfg.get("epsilon", defaults["epsilon"]))
        defaults["bootstrap_weight"] = float(
            selector_cfg.get("bootstrap_weight", defaults["bootstrap_weight"])
        )
        defaults["warmup_steps_per_context"] = int(
            selector_cfg.get(
                "warmup_steps_per_context",
                defaults["warmup_steps_per_context"],
            )
        )
        defaults["min_delta_for_nonstatic"] = float(
            bootstrap.get(
                "min_delta_for_nonstatic",
                selector_cfg.get(
                    "min_delta_for_nonstatic",
                    defaults["min_delta_for_nonstatic"],
                ),
            )
        )
        defaults["freeze_sizes"] = list(bootstrap.get("freeze_sizes", defaults["freeze_sizes"]))
        defaults["fallback_regression_limit"] = float(
            guardrails.get(
                "fallback_regression_limit",
                defaults["fallback_regression_limit"],
            )
        )
        defaults["max_fallback_rate"] = float(
            guardrails.get(
                "disable_if_fallback_rate_gt",
                guardrails.get("max_fallback_rate", defaults["max_fallback_rate"]),
            )
        )
        defaults["correctness_threshold"] = float(
            guardrails.get(
                "disable_if_correctness_error_gt",
                guardrails.get("correctness_threshold", defaults["correctness_threshold"]),
            )
        )
        defaults["boundary_report_path"] = str(
            bootstrap.get("boundary_report_path", defaults["boundary_report_path"])
        )

        return defaults

    @staticmethod
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
            arm = CONFIG_TO_ARM.get(str(config_id))
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

        for size_data in priors.values():
            for arm_data in size_data.values():
                arm_data["eligible_nonstatic"] = float(
                    1.0 if arm_data["delta_vs_baseline_percent"] >= min_delta_for_nonstatic else 0.0
                )

        return priors, str(report_path)

    @staticmethod
    def _context_from_size(size: int) -> str:
        return f"size_{size}"

    def _ensure_context(
        self,
        *,
        context: str,
        arm_priors: dict[str, float],
    ) -> None:
        if context in self.stats:
            return

        self.stats[context] = {
            arm: {"count": 0.0, "reward_sum": 0.0} for arm in sorted(arm_priors.keys())
        }
        self.context_steps[context] = 0

        for arm, prior_reward in arm_priors.items():
            self.stats[context][arm]["count"] += self.bootstrap_weight
            self.stats[context][arm]["reward_sum"] += self.bootstrap_weight * float(prior_reward)

    def _mean_reward(self, context: str, arm: str) -> float:
        values = self.stats[context][arm]
        if values["count"] <= 0:
            return -1e12
        return float(values["reward_sum"] / values["count"])

    def _allowed_arms(
        self,
        *,
        size: int,
        static_arm: str,
        eligible_arms: list[str],
    ) -> list[str]:
        allowed = {static_arm}
        eligible = set(eligible_arms)

        if size in self.freeze_sizes:
            return sorted(allowed)

        size_priors = self.priors_by_size.get(size, {})
        for arm, arm_data in size_priors.items():
            if arm == static_arm or arm not in eligible:
                continue
            if arm_data["delta_vs_baseline_percent"] < self.min_delta_for_nonstatic:
                continue
            if arm_data["correctness_passed"] < 1.0:
                continue
            allowed.add(arm)

        return sorted(allowed)

    def select(
        self,
        *,
        size: int,
        static_arm: str,
        eligible_arms: list[str],
    ) -> dict[str, Any]:
        if self.disabled:
            return {
                "online_arm": static_arm,
                "allowed_arms": [static_arm],
                "selection_reason": "policy_disabled_guardrail",
                "explored": False,
                "force_static": True,
            }

        allowed_arms = self._allowed_arms(
            size=size,
            static_arm=static_arm,
            eligible_arms=eligible_arms,
        )
        if static_arm not in allowed_arms:
            allowed_arms = sorted(set([static_arm] + allowed_arms))

        context = self._context_from_size(size)
        arm_priors = {
            arm: float(self.priors_by_size.get(size, {}).get(arm, {}).get("gflops", 0.0))
            for arm in allowed_arms
        }
        self._ensure_context(context=context, arm_priors=arm_priors)

        if size in self.freeze_sizes:
            return {
                "online_arm": static_arm,
                "allowed_arms": allowed_arms,
                "selection_reason": "force_static_guardrail",
                "explored": False,
                "force_static": True,
            }

        if self.context_steps[context] < self.warmup_steps_per_context:
            best_warm = max(
                allowed_arms,
                key=lambda arm: (self._mean_reward(context, arm), -allowed_arms.index(arm)),
            )
            return {
                "online_arm": best_warm,
                "allowed_arms": allowed_arms,
                "selection_reason": "bootstrap_warmup",
                "explored": False,
                "force_static": False,
            }

        if len(allowed_arms) > 1 and float(self.rng.random()) < self.epsilon:
            return {
                "online_arm": str(self.rng.choice(allowed_arms)),
                "allowed_arms": allowed_arms,
                "selection_reason": "epsilon_exploration",
                "explored": True,
                "force_static": False,
            }

        best = max(
            allowed_arms,
            key=lambda arm: (self._mean_reward(context, arm), -allowed_arms.index(arm)),
        )
        return {
            "online_arm": best,
            "allowed_arms": allowed_arms,
            "selection_reason": "policy_exploitation",
            "explored": False,
            "force_static": False,
        }

    def record_feedback(
        self,
        *,
        size: int,
        static_arm: str,
        online_arm: str,
        online_gflops: float,
        static_gflops: float,
        online_max_error: float,
    ) -> dict[str, Any]:
        context = self._context_from_size(size)
        if context not in self.stats:
            arm_priors = {arm: 0.0 for arm in sorted(set([static_arm, online_arm]))}
            self._ensure_context(context=context, arm_priors=arm_priors)

        if online_arm not in self.stats[context]:
            self.stats[context][online_arm] = {"count": 0.0, "reward_sum": 0.0}

        correctness_failed = float(online_max_error) > self.correctness_threshold
        regress_guard = float(online_gflops) < (
            float(static_gflops) * (1.0 - self.fallback_regression_limit)
        )
        fallback_triggered = bool(correctness_failed or regress_guard)

        if fallback_triggered:
            self.fallback_count += 1
            fallback_reason = "correctness" if correctness_failed else "regression_guard"
            executed_arm = static_arm
            executed_gflops = float(static_gflops)
        else:
            fallback_reason = None
            executed_arm = online_arm
            executed_gflops = float(online_gflops)

        self.stats[context][online_arm]["count"] += 1.0
        self.stats[context][online_arm]["reward_sum"] += executed_gflops
        self.context_steps[context] = self.context_steps.get(context, 0) + 1

        self.total_feedback += 1
        if correctness_failed:
            self.correctness_failures += 1

        fallback_rate = self.fallback_count / max(1, self.total_feedback)
        if correctness_failed:
            self.disabled = True
            self.disable_reason = "correctness_guard"
        elif fallback_rate > self.max_fallback_rate:
            self.disabled = True
            self.disable_reason = (
                f"fallback_rate_guard ({fallback_rate:.3f} > {self.max_fallback_rate:.3f})"
            )

        return {
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "executed_arm": executed_arm,
            "executed_gflops": executed_gflops,
            "correctness_failed": correctness_failed,
            "disable_signal": self.disabled,
            "disable_reason": self.disable_reason,
            "fallback_rate": float(fallback_rate),
        }

    def snapshot(self) -> dict[str, Any]:
        context_snapshot: dict[str, Any] = {}
        for context, arm_stats in self.stats.items():
            context_snapshot[context] = {
                "steps": int(self.context_steps.get(context, 0)),
                "arms": {},
            }
            for arm, values in arm_stats.items():
                count = float(values["count"])
                mean_reward = float(values["reward_sum"] / count) if count > 0 else 0.0
                context_snapshot[context]["arms"][arm] = {
                    "count": int(round(count)),
                    "mean_reward": mean_reward,
                }

        return {
            "seed": self.seed,
            "epsilon": self.epsilon,
            "bootstrap_weight": self.bootstrap_weight,
            "warmup_steps_per_context": self.warmup_steps_per_context,
            "min_delta_for_nonstatic": self.min_delta_for_nonstatic,
            "freeze_sizes": sorted(self.freeze_sizes),
            "fallback_regression_limit": self.fallback_regression_limit,
            "max_fallback_rate": self.max_fallback_rate,
            "correctness_threshold": self.correctness_threshold,
            "boundary_report_path": self.boundary_report_path,
            "bootstrap_source": self.bootstrap_source,
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "total_feedback": int(self.total_feedback),
            "fallback_count": int(self.fallback_count),
            "fallback_rate": float(self.fallback_count / max(1, self.total_feedback)),
            "correctness_failures": int(self.correctness_failures),
            "contexts": context_snapshot,
        }
