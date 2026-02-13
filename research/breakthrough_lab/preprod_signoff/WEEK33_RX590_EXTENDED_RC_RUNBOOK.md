# Week33 RX590 Extended RC Runbook

## Scope

Release candidate workflow for extended RX590 tests after full Week33 promote closure.

## Active Inputs

- Stable baseline manifest: `research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json`
- Week33 Block1 decision: `research/breakthrough_lab/week33_block1_monthly_continuity_decision.json`
- Week33 Block2 decision: `research/breakthrough_lab/week33_block2_alert_bridge_observability_decision.json`
- Week33 Block3 decision: `research/breakthrough_lab/week33_block3_biweekly_comparative_decision.json`
- Platform policy: `research/breakthrough_lab/week33_controlled_rollout/WEEK30_BLOCK3_PLATFORM_POLICY_DECISION.json`
- Rollback SLA: `research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Preconditions

1. Canonical gate is green before expanding scope.
2. Driver diagnostics are healthy (`verify_drivers.py --json`).
3. Week33 Block1/2/3 decisions are `promote`.

## Extended RC Flow

1. Canonical pre-gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
2. Driver inventory:
   - `./venv/bin/python scripts/verify_drivers.py --json`
3. Extended canary run (controlled):
   - `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week33_controlled_rollout/week33_block1_monthly_continuity_recovery_20260213_040810.json --output-dir research/breakthrough_lab/week34_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week34_controlled_rollout --output-prefix week34_block1_monthly_continuity_rc_canary --report-dir research/breakthrough_lab/week8_validation_discipline --snapshots 8 --iterations 4 --pressure-size 512 --pressure-iterations 0 --pressure-pulses 0`
4. Alert bridge verification (live path):
   - `./venv/bin/python research/breakthrough_lab/week30_controlled_rollout/run_week30_block2_alert_bridge_observability.py --mode local --bridge-endpoint http://127.0.0.1:8855/webhook --cycles 3 --retry-attempts 3 --source-cycle-report-path research/breakthrough_lab/week34_controlled_rollout/week34_block1_monthly_continuity_rc_canary_*.json --source-alerts-path research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_20260211_140738.json --output-dir research/breakthrough_lab/week34_controlled_rollout --output-prefix week34_block2_alert_bridge_observability_rc_canary`
5. Canonical post-gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

## Promotion Rules

Promote RC only if all are true:
- Block1 canary decision is `promote`.
- Block2 observability decision is `promote`.
- Canonical pre/post gates are `promote`.
- `t5_disable_events_total == 0` and weekly `t5_overhead_max <= 3.0`.
- Rollback path remains executable under SLA.

## Rollback

1. Apply rollback script:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
2. Re-run canonical gate and freeze promotion if not green.
3. Record incident and keep production policy in `clover_primary_rusticl_canary`.
