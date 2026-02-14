"""T5 ABFT-lite runtime guardrails with auto-disable behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

DEFAULT_POLICY_PATH = "research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json"
DEFAULT_STATE_PATH = "results/runtime_states/t5_abft_guard_state.json"


@dataclass(frozen=True)
class T5ABFTSamplingConfig:
    sampling_period: int
    row_samples: int
    col_samples: int
    projection_count: int
    residual_scale: float
    residual_margin: float
    residual_floor: float


class T5ABFTAutoDisableGuard:
    """Applies runtime guardrails and disables ABFT when thresholds are violated."""

    def __init__(
        self,
        *,
        policy_path: str = DEFAULT_POLICY_PATH,
        state_path: str = DEFAULT_STATE_PATH,
    ) -> None:
        self.policy_path = str(policy_path)
        self.state_path = str(state_path)
        self.policy = self._load_policy(Path(self.policy_path))
        self.guardrails = dict(self.policy["runtime_guardrails"])

        mode = self.policy["abft_mode"]
        self.sampling = T5ABFTSamplingConfig(
            sampling_period=int(mode["sampling_period"]),
            row_samples=int(mode["row_samples"]),
            col_samples=int(mode["col_samples"]),
            projection_count=int(mode["projection_count"]),
            residual_scale=float(mode["residual_scale"]),
            residual_margin=float(mode["residual_margin"]),
            residual_floor=float(mode["residual_floor"]),
        )

        self.enabled = True
        self.disable_reason: str | None = None
        self.disable_events = 0
        self.evaluations: list[dict[str, Any]] = []
        self.violation_streaks: dict[str, int] = {
            "false_positive_rate": 0,
            "effective_overhead_percent": 0,
        }
        self.disable_after_consecutive_false_positive_violations = max(
            1,
            int(self.guardrails.get("disable_after_consecutive_false_positive_violations", 1)),
        )
        self.disable_after_consecutive_overhead_violations = max(
            1,
            int(self.guardrails.get("disable_after_consecutive_overhead_violations", 1)),
        )
        hard = self.guardrails.get("disable_if_effective_overhead_percent_gt_hard")
        self.disable_if_effective_overhead_percent_gt_hard = (
            float(hard) if hard is not None else None
        )
        self.stateful_hysteresis = bool(self.guardrails.get("stateful_hysteresis", False))
        self._restore_hysteresis_state()
        self._persist_state()

    @staticmethod
    def _load_policy(path: Path) -> dict[str, Any]:
        data = cast(dict[str, Any], json.loads(path.read_text()))
        required = ["policy_id", "abft_mode", "runtime_guardrails", "stress_evidence"]
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"T5 policy missing required fields: {missing}")
        return data

    def evaluate_session(
        self,
        *,
        session_id: int,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        checks = {
            "false_positive_rate": {
                "observed": float(metrics["false_positive_rate"]),
                "threshold": float(self.guardrails["disable_if_false_positive_rate_gt"]),
                "comparator": "<=",
            },
            "effective_overhead_percent": {
                "observed": float(metrics["effective_overhead_percent"]),
                "threshold": float(self.guardrails["disable_if_effective_overhead_percent_gt"]),
                "comparator": "<=",
            },
            "correctness_error": {
                "observed": float(metrics["max_error"]),
                "threshold": float(self.guardrails["disable_if_correctness_error_gt"]),
                "comparator": "<=",
            },
        }

        for payload in checks.values():
            observed = float(cast(Any, payload["observed"]))
            threshold = float(cast(Any, payload["threshold"]))
            payload["pass"] = bool(observed <= threshold)

        failed = [name for name, payload in checks.items() if not payload["pass"]]

        if checks["false_positive_rate"]["pass"]:
            self.violation_streaks["false_positive_rate"] = 0
        else:
            self.violation_streaks["false_positive_rate"] += 1

        if checks["effective_overhead_percent"]["pass"]:
            self.violation_streaks["effective_overhead_percent"] = 0
        else:
            self.violation_streaks["effective_overhead_percent"] += 1

        disable_reasons: list[str] = []
        if not checks["correctness_error"]["pass"]:
            disable_reasons.append("correctness_error")
        if (
            self.disable_if_effective_overhead_percent_gt_hard is not None
            and float(cast(Any, checks["effective_overhead_percent"]["observed"]))
            > self.disable_if_effective_overhead_percent_gt_hard
        ):
            disable_reasons.append("effective_overhead_percent_hard")
        if (
            not checks["false_positive_rate"]["pass"]
            and self.violation_streaks["false_positive_rate"]
            >= self.disable_after_consecutive_false_positive_violations
        ):
            disable_reasons.append("false_positive_rate")
        if (
            not checks["effective_overhead_percent"]["pass"]
            and self.violation_streaks["effective_overhead_percent"]
            >= self.disable_after_consecutive_overhead_violations
        ):
            disable_reasons.append("effective_overhead_percent")

        disable_signal = len(disable_reasons) > 0
        if disable_signal and self.enabled:
            self.enabled = False
            self.disable_events += 1
            self.disable_reason = f"guardrail_violation:{','.join(disable_reasons)}"

        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": int(session_id),
            "checks": checks,
            "failed_guardrails": failed,
            "disable_reasons": disable_reasons,
            "all_passed": len(failed) == 0,
            "disable_signal": disable_signal,
            "enabled_after_eval": self.enabled,
            "disable_reason": self.disable_reason,
            "violation_streaks": dict(self.violation_streaks),
            "disable_thresholds": {
                "false_positive_rate": int(
                    self.disable_after_consecutive_false_positive_violations
                ),
                "effective_overhead_percent": int(
                    self.disable_after_consecutive_overhead_violations
                ),
                "effective_overhead_percent_hard": (
                    None
                    if self.disable_if_effective_overhead_percent_gt_hard is None
                    else float(self.disable_if_effective_overhead_percent_gt_hard)
                ),
            },
            "fallback_action": (
                "auto_disable_abft_runtime" if disable_signal else "continue_abft_runtime"
            ),
        }
        self.evaluations.append(entry)
        self._persist_state()
        return entry

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy["policy_id"],
            "policy_path": self.policy_path,
            "enabled": self.enabled,
            "disable_reason": self.disable_reason,
            "disable_events": int(self.disable_events),
            "violation_streaks": dict(self.violation_streaks),
            "stateful_hysteresis": bool(self.stateful_hysteresis),
            "disable_after_consecutive_false_positive_violations": int(
                self.disable_after_consecutive_false_positive_violations
            ),
            "disable_after_consecutive_overhead_violations": int(
                self.disable_after_consecutive_overhead_violations
            ),
            "disable_if_effective_overhead_percent_gt_hard": (
                None
                if self.disable_if_effective_overhead_percent_gt_hard is None
                else float(self.disable_if_effective_overhead_percent_gt_hard)
            ),
            "sampling": {
                "sampling_period": self.sampling.sampling_period,
                "row_samples": self.sampling.row_samples,
                "col_samples": self.sampling.col_samples,
                "projection_count": self.sampling.projection_count,
                "residual_scale": self.sampling.residual_scale,
                "residual_margin": self.sampling.residual_margin,
                "residual_floor": self.sampling.residual_floor,
            },
            "guardrails": self.guardrails,
            "evaluations": self.evaluations,
        }

    def _persist_state(self) -> None:
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_snapshot()
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(payload, indent=2))

    def _restore_hysteresis_state(self) -> None:
        if not self.stateful_hysteresis:
            return
        path = Path(self.state_path)
        if not path.exists():
            return
        try:
            previous = json.loads(path.read_text())
        except Exception:
            return
        if previous.get("policy_id") != self.policy.get("policy_id"):
            return
        streaks = previous.get("violation_streaks")
        if not isinstance(streaks, dict):
            return
        for key in self.violation_streaks:
            value = streaks.get(key)
            if isinstance(value, int) and value >= 0:
                self.violation_streaks[key] = value
