# Week9 Block6 Pre-Production Runbook (RX590)

## Scope

Controlled pre-production sign-off workflow before final production recommendation.

## Preconditions

1. `./venv/bin/python scripts/verify_drivers.py --json` returns `overall_status=good`.
2. T5 active policy is `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json`.
3. Week9 Block5 evidence is present and `promote`.

## Execution Steps

1. Launch long wall-clock canary:
   - `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block6_wallclock_canary.py --duration-minutes 30 --snapshot-interval-minutes 5 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 6 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2`
2. Run canonical validation gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
3. Rebuild comparative dashboard including Block6:
   - `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json`
   - Registrar deltas Block6 en acta/readiness aunque el dashboard conserve corte comparativo Block1..5.
4. Validate rollback drill:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
5. Confirm rollback SLA contract:
   - `research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Pass Criteria

1. Wall-clock canary decision: `promote`
2. Canonical gate decision: `promote`
3. No T5 disable events in canary (`observed_disable_total=0`)
4. Correctness `max_error <= 1e-3`
5. rusticl/clover min ratio >= `0.80`

## Failure Handling

1. Execute rollback script:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
2. Freeze canary progression and open corrective block with `iterate`.
3. Re-run canonical gate and log post-rollback evidence.
