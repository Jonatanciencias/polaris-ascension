# Acta Week 10 - Block 1.1 (T5 Hardening + Controlled Rollout Rerun)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: hardening T5 para eliminar `disable_events` en rollout de bajo alcance, rerun con `>=4` snapshots y gate canonico obligatorio antes de promocion.

## Objetivo

1. Ajustar politica T5 de rollout de bajo alcance para evitar auto-disable espurio.
2. Re-ejecutar rollout controlado con al menos 4 snapshots y buscar `promote`.
3. Verificar gate obligatorio (`run_validation_suite.py --tier canonical --driver-smoke`) antes de promocion.

## Implementacion

Assets nuevos/actualizados:

- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_1.json`
  - sampling period `10`
  - row/col samples `12/12`
  - projection count `4`
  - guardrail overhead suave `3.5`, hard `5.0`
  - disable por overhead consecutivo `3`
- Reuso del runner:
  - `research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py`

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 4 --snapshot-interval-minutes 60 --sleep-between-snapshots-seconds 0 --sizes 1400 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 6 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_1.json --output-prefix week10_block1_1_controlled_rollout`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block1_1_controlled_rollout_20260208_161153.json`

Artifacts:

- `research/breakthrough_lab/platform_compatibility/week10_block1_1_controlled_rollout_20260208_161153.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_1_controlled_rollout_20260208_161153.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_161219.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_161219.md`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_161230.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_161230.md`

## Resultados

Rollout rerun (4 snapshots):

- Decision: `promote`
- Snapshots ejecutados: `4/4`
- Rollback triggered: `False`
- Failed checks: `[]`
- T5:
  - overhead max: `1.9597%`
  - overhead mean: `1.9386%`
  - false positive max: `0.0`
  - disable events total: `0`
- T3:
  - fallback max: `0.0`
  - policy disabled total: `0`
- Correctness max global: `0.000335693359375` (`<= 1e-3`)
- Drift:
  - T3 (1400): `+0.884%`
  - T5 (1400): `+0.054%`

Gate canonico obligatorio:

- Decision: `promote`
- `pytest tests`: `85 passed`
- `verify_drivers --json`: `overall_status=good`

Dashboard extendido (Block6 + drift semanal):

- Decision dashboard: `promote`
- Cadena activa: `block2 -> block3 -> block4 -> block5 -> block6 -> block10`
- Cumulativo (`block2 -> block10`):
  - T3 avg GFLOPS: `+6.042%`
  - T5 avg GFLOPS: `+8.203%`
  - T5 disable events delta: `0`

## Decision Formal

Tracks:

- `week10_block1_1_t5_policy_hardening`: **promote**
- `week10_block1_1_controlled_rollout_rerun`: **promote**
- `week10_block1_1_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El hardening T5 elimina disable events en scope bajo con 4 snapshots completos y sin rollback.
- El gate obligatorio permanece `promote`, habilitando continuar con expansion controlada del rollout.

## Estado del Bloque

`Week 10 - Block 1.1` cerrado con `promote`, evidencia reproducible y perfil T5 estabilizado para pruebas controladas.

