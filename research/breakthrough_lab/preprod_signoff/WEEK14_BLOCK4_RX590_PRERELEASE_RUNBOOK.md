# Week14 Block4 RX590 Pre-Release Runbook

## Scope

Pre-release enablement workflow for controlled RX590 real-world pilots.

## Active Inputs

- Weekly policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- T5 policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json`
- Rollback SLA: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`
- Weekly cadence baseline: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_WEEKLY_CADENCE.json`

## Preconditions

1. Driver smoke is healthy (`overall_status=good`).
2. Week14 Block2 and Block3 are closed in `promote`.
3. Canonical validation gate is `promote` before any scope increase.

## Enablement Steps

1. Baseline validation gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
2. Verify runtime inventory:
   - `./venv/bin/python scripts/verify_drivers.py --json`
3. Run controlled dry-run scope:
   - `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 3 --snapshot-interval-minutes 1 --sizes 1400 2048 --sessions 1 --iterations 4`
4. If any hard gate fails, apply rollback:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
5. Re-run canonical gate post-rollback and freeze promotion if not green.

## Promotion Gate

Promote only if all are true:
- Dry-run decision is `go`.
- Canonical gate post-run is `promote`.
- `t5_disable_events_total == 0` and correctness bound remains within `1e-3`.
- Rollback path remains executable under SLA contract.

