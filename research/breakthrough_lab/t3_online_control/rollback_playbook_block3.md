# T3 Rollback Playbook - Week 8 Block 3

## Trigger Conditions

Rollback to static selector (`auto`) is mandatory if any condition is met:

1. Correctness error exceeds policy threshold (`disable_if_correctness_error_gt`).
2. Fallback rate exceeds policy threshold (`disable_if_fallback_rate_gt`).
3. Drift guardrails breach in `warm_queue_pressure` scenario.

## Immediate Actions

1. Force runtime mode to static:
   - CLI/API kernel mode: `auto`
2. Disable T3 controlled rollout:
   - stop using `auto_t3_controlled` in production runs
3. Preserve incident evidence:
   - keep campaign JSON/MD artifacts
   - capture policy snapshot and disable reason

## Verification After Rollback

1. Run canonical validation:
   - `python scripts/run_validation_suite.py --tier canonical --driver-smoke`
2. Confirm guardrails:
   - fallback-related counters not increasing (T3 disabled)
   - correctness remains within contract

## Re-enable Preconditions

1. Root cause identified and fixed.
2. New policy version prepared with explicit threshold deltas.
3. Drift campaign rerun passes all checks before re-enabling T3 controlled mode.
