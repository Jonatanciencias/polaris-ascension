# Acta Week 10 - Block 1.4 + 1.5 (Long Window + Platform Split)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - Block 1.4: ventana mas larga (`>=45 min equivalente`) con `1400+2048`.
  - Block 1.5: canary split `Clover/rusticl` con rollback SLA activo.
  - Gate obligatorio antes de cada promocion.

## Objetivo

1. Validar estabilidad en ventana extendida con la politica endurecida actual.
2. Validar compatibilidad split de plataforma bajo los mismos guardrails.
3. Mantener disciplina de promocion con gate canonico obligatorio.

## Ejecucion Formal

Block 1.4 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week10_block1_controlled_rollout.py --snapshots 6 --snapshot-interval-minutes 10 --sleep-between-snapshots-seconds 0 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --rollback-after-consecutive-soft-overhead-violations 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --output-prefix week10_block1_4_long_window`

Block 1.5 command:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block5_preprod_pilot.py --snapshots 4 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 2 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --baseline-block4-path research/breakthrough_lab/platform_compatibility/week10_block1_4_long_window_20260208_165345.json --rollback-script-path research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh --output-prefix week10_block1_5_platform_split`

Gates obligatorios:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 1.4 gate: `validation_suite_canonical_20260208_165410.json`
  - Block 1.5 gate: `validation_suite_canonical_20260208_165700.json`

Dashboard refresh:

- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block1_5_platform_split_20260208_165631.json`

## Resultados

### Block 1.4 (long window equivalente: 6 snapshots x 10 min)

- Decision: `promote`
- Snapshots: `6/6`
- Rollback: `false`
- Failed checks: `[]`
- Correctness max: `0.000579833984375`
- T3 fallback max: `0.0`
- T5 disable events total: `0`
- T5 overhead:
  - max: `1.5122%`
  - mean: `1.1054%`

### Block 1.5 (split Clover/rusticl)

- Decision: `promote`
- Snapshots: `[1,2,3,4]`
- Rollback: `false` (SLA/rollback script activo en configuracion del runner)
- Failed checks: `[]`
- Correctness max: `0.000701904296875`
- T3 fallback max: `0.0`
- T5 disable events total: `0`
- T5 overhead:
  - max: `1.7128%`
  - mean: `1.1071%`
- Ratio minimo rusticl/clover: `0.9206` (`>=0.80`)

### Gates obligatorios

- Block 1.4 gate: **promote**
  - `pytest`: `85 passed`
  - drivers smoke: `good`
- Block 1.5 gate: **promote**
  - `pytest`: `85 passed`
  - drivers smoke: `good`

### Dashboard post-bloques

- Artifact: `week9_comparative_dashboard_20260208_165707.json`
- Decision: `promote`
- Cadena activa: `block2 -> block3 -> block4 -> block5 -> block6 -> block10`

## Decision Formal

Tracks:

- `week10_block1_4_long_window_1400_2048`: **promote**
- `week10_block1_5_platform_split_clover_rusticl`: **promote**
- `week10_block1_4_and_1_5_mandatory_canonical_gates`: **promote**

Block decision:

- **promote**

Razonamiento:

- La ventana extendida y el split de plataforma se mantienen dentro de guardrails sin rollback ni disable events.
- La disciplina de gate obligatorio se mantiene y valida cada promocion.

## Estado del Bloque

`Week 10 - Block 1.4/1.5` cerrado con `promote` y evidencia reproducible.

