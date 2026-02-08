# Acta Week 10 - Block 1.6 + Block 2.1 (Split Extendido + Preproduccion Escalada)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - Block 1.6: split extendido `Clover/rusticl` con mayor horizonte y control de drift por plataforma.
  - Block 2.1: preproduccion escalada (mayor `sessions/iterations`) manteniendo rollback SLA.
  - Gate obligatorio antes de cada promocion.

## Objetivo

1. Confirmar estabilidad split por plataforma en horizonte extendido.
2. Confirmar estabilidad en perfil preproductivo escalado con mayor carga por snapshot.
3. Mantener disciplina de gate canonico antes de cada promocion.

## Ejecucion Formal

Block 1.6 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block5_preprod_pilot.py --snapshots 8 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --baseline-block4-path research/breakthrough_lab/platform_compatibility/week10_block1_4_long_window_20260208_165345.json --rollback-script-path research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh --output-prefix week10_block1_6_platform_split_extended`

Block 2.1 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 4 --snapshot-interval-minutes 10 --sleep-between-snapshots-seconds 0 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 2 --iterations 10 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --baseline-block6-path research/breakthrough_lab/platform_compatibility/week10_block1_4_long_window_20260208_165345.json --output-prefix week10_block2_1_preprod_scaled`

Gates obligatorios:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 1.6 gate: `validation_suite_canonical_20260208_171618.json`
  - Block 2.1 gate: `validation_suite_canonical_20260208_171051.json`

Dashboard refresh:

- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block2_1_preprod_scaled_20260208_171024.json`

## Resultados

### Block 1.6 (split extendido Clover/rusticl)

- Decision: `promote`
- Snapshots: `[1,2,3,4,5,6,7,8]`
- Rollback: `false`
- Failed checks: `[]`
- Correctness max: `0.000701904296875`
- T3 fallback max: `0.0`
- T5 disable events total: `0`
- T5 overhead:
  - max: `1.7558%`
  - mean: `1.1130%`
- Ratio minimo rusticl/clover: `0.9200` (`>= 0.80`)

### Block 2.1 (preproduccion escalada)

- Decision: `promote`
- Snapshots: `4/4`
- Rollback: `false`
- Failed checks: `[]`
- Correctness max: `0.000579833984375`
- T3 fallback max: `0.0`
- T5 disable events total: `0`
- T5 overhead:
  - max: `1.2053%`
  - mean: `0.8472%`

### Gates obligatorios

- Block 1.6 gate: **promote**
  - `pytest`: `85 passed`
  - drivers smoke: `good`
- Block 2.1 gate: **promote**
  - `pytest`: `85 passed`
  - drivers smoke: `good`

### Dashboard post-bloques

- Artifact: `week9_comparative_dashboard_20260208_171111.json`
- Decision: `promote`
- Active chain: `block2 -> block3 -> block4 -> block5 -> block6 -> block10`

## Decision Formal

Tracks:

- `week10_block1_6_platform_split_extended`: **promote**
- `week10_block2_1_preproduction_scaled`: **promote**
- `week10_block1_6_and_2_1_mandatory_canonical_gates`: **promote**

Block decision:

- **promote**

Razonamiento:

- La extension split y la preproduccion escalada pasan sin rollback ni disable events.
- Los gates obligatorios previos a promocion se mantienen `promote`.

## Estado del Bloque

`Week 10 - Block 1.6/2.1` cerrado con `promote`.

