# Acta Week 10 - Block 2.4.1 (T5 Hardening Long Horizon + Rerun 40m)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - hardening T5 para horizonte largo sin relajar umbral hard de overhead,
  - rerun de canary de pared 40 minutos,
  - cierre final de decision `GO/NO-GO` con gate canonico obligatorio.

## Objetivo

1. Eliminar `disable_events` en horizonte largo.
2. Mantener `t5_overhead_max <= 5%` en la corrida completa.
3. Confirmar estabilidad operativa con gates canonicos en `promote`.

## Implementacion de Hardening

Policy nueva:

- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json`

Ajustes principales:

- `projection_count`: `4 -> 2`
- `row_samples`: `12 -> 8`
- `col_samples`: `12 -> 8`
- guardrails de disable/overhead: **sin relajar** (`hard max = 5.0`)

## Ejecucion Formal

Smoke corto pre-rerun:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block6_wallclock_canary.py --duration-minutes 10 --snapshot-interval-minutes 5 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 2 --iterations 10 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-block5-path research/breakthrough_lab/platform_compatibility/week10_block2_2_preprod_scaled_long_20260208_173314.json --output-prefix week10_block2_4_wallclock_hardening_smoke`
  - Artifact: `research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_hardening_smoke_20260208_193407.json`
  - Decision: `promote`

Gate canonico pre-rerun:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_193438.json`
  - Decision: `promote`

Rerun largo (40m):

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block6_wallclock_canary.py --duration-minutes 40 --snapshot-interval-minutes 5 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 2 --iterations 10 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-block5-path research/breakthrough_lab/platform_compatibility/week10_block2_2_preprod_scaled_long_20260208_173314.json --output-prefix week10_block2_4_wallclock_hardening_rerun`
  - Artifact JSON: `research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_hardening_rerun_20260208_201448.json`
  - Artifact MD: `research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_hardening_rerun_20260208_201448.md`
  - Decision: `promote`

Gate canonico post-rerun:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_201518.json`
  - Decision: `promote`

Dashboard refresh:

- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_hardening_rerun_20260208_201448.json`
  - Artifact: `research/breakthrough_lab/week9_comparative_dashboard_20260208_201655.json`
  - Decision: `promote`

## Resultados

### Rerun 40m (objetivo principal)

- Decision: `promote`
- Failed checks: `[]`
- Wall-clock target/actual: `40.0 / 40.0` minutos
- Snapshots: `8`
- Correctness max: `0.000579833984375`
- T3 fallback max: `0.0`
- T5:
  - disable events total: `0`
  - overhead max: `1.23892304192736%`
  - overhead mean: `0.8073009274442211%`
- Ratio minimo rusticl/clover: `0.920553841356942`

### Cumplimiento de objetivos

- `disable_events == 0`: **pass**
- `t5_overhead_max <= 5%`: **pass**

## Decision Formal

Tracks:

- `week10_block2_4_t5_policy_hardening_long_horizon`: **promote**
- `week10_block2_4_t5_hardening_smoke`: **promote**
- `week10_block2_4_t5_hardening_rerun_40m`: **promote**
- `week10_block2_4_t5_hardening_mandatory_gates`: **promote**
- `week10_block2_4_go_no_go_final`: **go**

Block decision:

- **promote / go**

Razonamiento:

- El hardening elimina los disable events y mantiene overhead max muy por debajo del hard limit sin degradar correctness.
- Se cumple la condicion de cambio de checklist de `NO-GO` a `GO`.

## Estado del Bloque

`Week 10 - Block 2.4.1` cerrado en `promote` con `GO` operativo controlado.
