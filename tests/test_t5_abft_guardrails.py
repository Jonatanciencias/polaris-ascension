from __future__ import annotations

import json
from pathlib import Path

from src.optimization_engines.t5_abft_guardrails import T5ABFTAutoDisableGuard


def _write_policy(path: Path) -> None:
    payload = {
        "policy_id": "t5-test-policy",
        "abft_mode": {
            "sampling_period": 8,
            "row_samples": 16,
            "col_samples": 16,
            "projection_count": 4,
            "faults_per_matrix_modelled": 2,
            "residual_scale": 5.0,
            "residual_margin": 0.01,
            "residual_floor": 0.05,
        },
        "runtime_guardrails": {
            "disable_if_false_positive_rate_gt": 0.05,
            "disable_if_effective_overhead_percent_gt": 3.0,
            "disable_if_correctness_error_gt": 0.001,
            "disable_if_uniform_recall_lt": 0.95,
            "disable_if_critical_recall_lt": 0.99,
        },
        "stress_evidence": {
            "recommended_mode": "periodic_8",
            "critical_recall": 1.0,
            "uniform_recall": 0.97,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def test_guard_allows_session_when_metrics_within_limits(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    state_path = tmp_path / "state.json"
    _write_policy(policy_path)

    guard = T5ABFTAutoDisableGuard(
        policy_path=str(policy_path),
        state_path=str(state_path),
    )
    out = guard.evaluate_session(
        session_id=0,
        metrics={
            "false_positive_rate": 0.0,
            "effective_overhead_percent": 1.2,
            "max_error": 5e-4,
        },
    )
    assert out["all_passed"] is True
    assert out["disable_signal"] is False
    assert guard.enabled is True
    assert state_path.exists()


def test_guard_auto_disables_on_violation(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    state_path = tmp_path / "state.json"
    _write_policy(policy_path)

    guard = T5ABFTAutoDisableGuard(
        policy_path=str(policy_path),
        state_path=str(state_path),
    )
    out = guard.evaluate_session(
        session_id=3,
        metrics={
            "false_positive_rate": 0.2,
            "effective_overhead_percent": 1.0,
            "max_error": 5e-4,
        },
    )
    assert out["all_passed"] is False
    assert out["disable_signal"] is True
    assert "false_positive_rate" in out["failed_guardrails"]
    assert guard.enabled is False
    assert guard.disable_events == 1
    assert guard.disable_reason is not None
