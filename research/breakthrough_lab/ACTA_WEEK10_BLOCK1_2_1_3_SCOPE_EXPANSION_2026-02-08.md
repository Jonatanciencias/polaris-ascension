# Acta Week 10 - Block 1.2 + Block 1.3 (Scope Expansion)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - Block 1.2: ampliar horizonte a `>=6` snapshots en `1400`.
  - Block 1.3: incluir `2048` en el mismo perfil endurecido.
  - Mantener gate obligatorio antes de cada promocion.

## Objetivo

1. Confirmar estabilidad temporal (horizonte extendido) del perfil endurecido T5 en `1400`.
2. Confirmar estabilidad de la misma politica al ampliar cobertura a `1400+2048`.
3. Validar gate canonico obligatorio (`--tier canonical --driver-smoke`) previo a cada promocion.

## Implementacion

Assets nuevos/actualizados del bloque:

- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_2_controlled_rollout_20260208_163545.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_3_controlled_rollout_20260208_163829.json`

## Ejecucion Formal

Block 1.2 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 6 --snapshot-interval-minutes 60 --sleep-between-snapshots-seconds 0 --sizes 1400 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_1.json --output-prefix week10_block1_2_controlled_rollout`

Block 1.3 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 6 --snapshot-interval-minutes 60 --sleep-between-snapshots-seconds 0 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --output-prefix week10_block1_3_controlled_rollout`

Gates obligatorios:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 1.2 gate: `validation_suite_canonical_20260208_163611.json`
  - Block 1.3 gate: `validation_suite_canonical_20260208_163857.json`

Dashboard refresh:

- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block1_3_controlled_rollout_20260208_163829.json`

## Resultados

### Block 1.2 (`1400`, 6 snapshots)

- Decision: `promote`
- Snapshots: `6/6`
- Rollback: `False`
- Failed checks: `[]`
- Correctness max: `0.000335693359375`
- T3 fallback max: `0.0`
- T5:
  - disable events total: `0`
  - overhead max: `1.5562%`
  - overhead mean: `1.4867%`
- Drift:
  - T3 1400: `+0.427%`
  - T5 1400: `-0.314%`

### Block 1.3 (`1400+2048`, 6 snapshots)

- Decision: `promote`
- Snapshots: `6/6`
- Rollback: `False`
- Failed checks: `[]`
- Correctness max: `0.000579833984375`
- T3 fallback max: `0.0`
- T5:
  - disable events total: `0`
  - overhead max: `1.5350%`
  - overhead mean: `1.0678%`
- Drift:
  - T3 1400: `+1.179%`
  - T3 2048: `-0.006%`
  - T5 1400: `-0.286%`
  - T5 2048: `+0.454%`

### Gates obligatorios (antes de promocion)

- Block 1.2 gate: `promote`
  - `pytest`: `85 passed`
  - drivers smoke: `good`
- Block 1.3 gate: `promote`
  - `pytest`: `85 passed`
  - drivers smoke: `good`

### Dashboard actualizado

- Artifact: `week9_comparative_dashboard_20260208_163903.json`
- Decision: `promote`
- Active chain: `block2 -> block3 -> block4 -> block5 -> block6 -> block10`
- Cumulative deltas (`block2 -> block10`):
  - T3 avg GFLOPS: `+0.087%`
  - T5 avg GFLOPS: `+0.165%`
  - T5 disable events delta: `0`

## Decision Formal

Tracks:

- `week10_block1_2_horizon_expansion_1400`: **promote**
- `week10_block1_3_scope_expansion_1400_2048`: **promote**
- `week10_block1_2_and_1_3_mandatory_canonical_gates`: **promote**
- `week10_block2_dashboard_refresh_post_scope_expansion`: **promote**

Block decision:

- **promote**

Razonamiento:

- La expansion temporal (6 snapshots) y de cobertura (`2048`) se valida sin rollback, sin disable events T5 y con gates canonicos en `promote`.
- El comportamiento se mantiene estable y determinista bajo el perfil endurecido.

## Estado del Bloque

`Week 10 - Block 1.2/1.3` cerrado con `promote` y evidencia formal completa.

