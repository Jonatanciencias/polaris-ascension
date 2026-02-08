# Week10 Block2.3 Operations Runbook (Controlled Production Recommendation)

## Scope

Operational runbook for controlled production recommendation after Week10 Blocks `1.6` and `2.1/2.2` close in `promote`.

## Preconditions

1. Canonical gate status is `promote`:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
2. Active T5 policy:
   - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json`
3. Rollback script available:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`
4. Hot rollback thresholds loaded from:
   - `research/breakthrough_lab/preprod_signoff/WEEK10_BLOCK2_3_ROLLBACK_HOT_THRESHOLDS.json`

## Recommended Rollout Phases

1. Phase A - Controlled low traffic (Clover primary):
   - Sizes: `1400`, `2048`
   - Kernel modes: `auto_t3_controlled`, `auto_t5_guarded`
   - Monitoring interval: every 10 logical minutes
2. Phase B - Controlled split canary:
   - Enable mirrored rusticl observation (`RUSTICL_ENABLE=radeonsi`)
   - Keep Clover as production decision baseline
3. Phase C - Escaled preproduction load:
   - Raise `sessions/iterations` only if Phase A/B remain `promote` with no rollback triggers.

## Standard Runtime Cycle

1. Run scaled monitor cycle:
   - `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 4 --snapshot-interval-minutes 10 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 2 --iterations 10 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --baseline-block6-path research/breakthrough_lab/platform_compatibility/week10_block2_1_preprod_scaled_20260208_171024.json --output-prefix week10_ops_cycle`
2. Evaluate thresholds from the latest JSON artifact against hot threshold contract.
3. If any hard threshold triggers, apply rollback immediately:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
4. Run canonical gate after rollback or before promotion decision:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

## Promotion Rule

Promote only if all are true in the same operational window:

1. No rollback trigger.
2. T5 disable events remain zero.
3. Correctness bound and drift guardrails pass.
4. Canonical gate is `promote`.

## Failure Handling

1. Trigger rollback script immediately.
2. Freeze promotion state (`iterate`) and record incident in `research/breakthrough_lab/platform_compatibility/`.
3. Re-run canonical gate and attach artifact to the incident record.

