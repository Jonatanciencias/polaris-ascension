from __future__ import annotations

import json
from pathlib import Path

from src.optimization_engines.t3_controlled_policy import ControlledT3Policy


def _policy(**kwargs) -> ControlledT3Policy:
    priors = {
        1536: {
            "tile20": {
                "gflops": 540.0,
                "delta_vs_baseline_percent": 10.0,
                "max_error": 5e-4,
                "correctness_passed": 1.0,
                "eligible_nonstatic": 1.0,
            },
            "tile20_v3_1400": {
                "gflops": 780.0,
                "delta_vs_baseline_percent": 28.0,
                "max_error": 5e-4,
                "correctness_passed": 1.0,
                "eligible_nonstatic": 1.0,
            },
        }
    }
    config = {
        "seed": 7,
        "epsilon": 0.0,
        "bootstrap_weight": 2.0,
        "warmup_steps_per_context": 0,
        "freeze_sizes": [1400, 2048],
        "boundary_report_path": "missing.json",
        "priors_by_size": priors,
    }
    config.update(kwargs)
    return ControlledT3Policy(**config)


def test_select_respects_freeze_and_priors() -> None:
    policy = _policy()

    active = policy.select(
        size=1536,
        static_arm="tile20",
        eligible_arms=["tile20", "tile20_v3_1400", "tile24"],
    )
    assert active["online_arm"] == "tile20_v3_1400"
    assert active["selection_reason"] == "policy_exploitation"

    frozen = policy.select(
        size=1400,
        static_arm="tile20_v3_1400",
        eligible_arms=["tile20", "tile20_v3_1400", "tile24"],
    )
    assert frozen["online_arm"] == "tile20_v3_1400"
    assert frozen["selection_reason"] == "force_static_guardrail"
    assert frozen["force_static"] is True


def test_correctness_guard_disables_policy() -> None:
    policy = _policy()
    feedback = policy.record_feedback(
        size=1536,
        static_arm="tile20",
        online_arm="tile20_v3_1400",
        online_gflops=800.0,
        static_gflops=600.0,
        online_max_error=2e-3,
    )
    assert feedback["fallback_triggered"] is True
    assert feedback["fallback_reason"] == "correctness"
    assert feedback["disable_signal"] is True

    next_sel = policy.select(
        size=1536,
        static_arm="tile20",
        eligible_arms=["tile20", "tile20_v3_1400", "tile24"],
    )
    assert next_sel["online_arm"] == "tile20"
    assert next_sel["selection_reason"] == "policy_disabled_guardrail"


def test_fallback_rate_guard_disables_policy() -> None:
    policy = _policy(max_fallback_rate=0.1)
    feedback = policy.record_feedback(
        size=1536,
        static_arm="tile20",
        online_arm="tile20_v3_1400",
        online_gflops=500.0,
        static_gflops=700.0,
        online_max_error=1e-4,
    )
    assert feedback["fallback_triggered"] is True
    assert feedback["fallback_reason"] == "regression_guard"
    assert feedback["disable_signal"] is True
    assert "fallback_rate_guard" in str(feedback["disable_reason"])


def test_parse_policy_file_defaults_when_missing(tmp_path) -> None:
    missing = tmp_path / "missing_policy.json"
    parsed = ControlledT3Policy._parse_policy_file(missing)

    assert parsed["epsilon"] == 0.05
    assert parsed["warmup_steps_per_context"] == 2
    assert parsed["freeze_sizes"] == [1400, 2048]


def test_parse_policy_file_reads_overrides(tmp_path) -> None:
    policy_path = tmp_path / "policy.json"
    payload = {
        "selector_policy": {
            "epsilon": 0.15,
            "bootstrap_weight": 4.0,
            "warmup_steps_per_context": 3,
            "min_delta_for_nonstatic": 11.0,
        },
        "guardrails": {
            "fallback_regression_limit": 0.07,
            "disable_if_fallback_rate_gt": 0.2,
            "disable_if_correctness_error_gt": 2e-3,
        },
        "bootstrap": {
            "freeze_sizes": [1536],
            "boundary_report_path": "custom_boundary.json",
            "min_delta_for_nonstatic": 12.0,
        },
    }
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = ControlledT3Policy._parse_policy_file(policy_path)
    assert parsed["epsilon"] == 0.15
    assert parsed["bootstrap_weight"] == 4.0
    assert parsed["warmup_steps_per_context"] == 3
    assert parsed["min_delta_for_nonstatic"] == 12.0
    assert parsed["fallback_regression_limit"] == 0.07
    assert parsed["max_fallback_rate"] == 0.2
    assert parsed["correctness_threshold"] == 2e-3
    assert parsed["freeze_sizes"] == [1536]
    assert parsed["boundary_report_path"] == "custom_boundary.json"


def test_load_boundary_priors_filters_unknown_and_marks_eligibility(tmp_path) -> None:
    boundary = tmp_path / "boundary.json"
    report = {
        "search": {
            "ranked_candidates": [
                {
                    "config_id": "t20_prod_v4_u10_l10",
                    "size": 1536,
                    "gflops": 600.0,
                    "delta_vs_baseline_percent": 9.0,
                    "max_error": 5e-4,
                    "correctness_passed": True,
                },
                {
                    "config_id": "t24_prod_v4_u0_l12",
                    "size": 1536,
                    "gflops": 580.0,
                    "delta_vs_baseline_percent": 6.0,
                    "max_error": 5e-4,
                    "correctness_passed": True,
                },
                {
                    "config_id": "unknown_arm",
                    "size": 1536,
                    "gflops": 999.0,
                },
            ]
        }
    }
    boundary.write_text(json.dumps(report), encoding="utf-8")

    priors, source = ControlledT3Policy._load_boundary_priors(
        report_path=Path(boundary),
        min_delta_for_nonstatic=8.0,
    )

    assert source == str(boundary)
    assert "tile20" in priors[1536]
    assert "tile24" in priors[1536]
    assert priors[1536]["tile20"]["eligible_nonstatic"] == 1.0
    assert priors[1536]["tile24"]["eligible_nonstatic"] == 0.0


def test_select_bootstrap_warmup_then_exploration_and_snapshot() -> None:
    policy = _policy(epsilon=1.0, warmup_steps_per_context=1)

    warmup = policy.select(
        size=1536,
        static_arm="tile20",
        eligible_arms=["tile20", "tile20_v3_1400", "tile24"],
    )
    assert warmup["selection_reason"] == "bootstrap_warmup"

    _ = policy.record_feedback(
        size=1536,
        static_arm="tile20",
        online_arm="tile20_v3_1400",
        online_gflops=800.0,
        static_gflops=700.0,
        online_max_error=1e-4,
    )

    explored = policy.select(
        size=1536,
        static_arm="tile20",
        eligible_arms=["tile20", "tile20_v3_1400", "tile24"],
    )
    assert explored["selection_reason"] == "epsilon_exploration"
    assert explored["explored"] is True

    snap = policy.snapshot()
    assert snap["contexts"]["size_1536"]["steps"] == 1
    assert "tile20_v3_1400" in snap["contexts"]["size_1536"]["arms"]
