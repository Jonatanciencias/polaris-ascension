from __future__ import annotations

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
    return ControlledT3Policy(
        seed=7,
        epsilon=0.0,
        bootstrap_weight=2.0,
        warmup_steps_per_context=0,
        freeze_sizes=[1400, 2048],
        boundary_report_path="missing.json",
        priors_by_size=priors,
        **kwargs,
    )


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

