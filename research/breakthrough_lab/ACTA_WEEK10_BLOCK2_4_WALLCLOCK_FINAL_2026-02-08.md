# Acta Week 10 - Block 2.4 (Canary de Pared Final + Cierre Go/No-Go)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar canary de pared final en horizonte extendido,
  - aplicar runbook de rollback ante falla de guardrail,
  - cerrar decision `GO/NO-GO` con gate canonico obligatorio.

## Objetivo

1. Validar estabilidad final en ventana real de `40` minutos bajo perfil endurecido.
2. Confirmar disciplina operacional: rollback + gate canonico + decision formal.

## Ejecucion Formal

Gate canonico (precondicion):

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_175520.json`
  - Decision: `promote`

Canary de pared final:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block6_wallclock_canary.py --duration-minutes 40 --snapshot-interval-minutes 5 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 2 --iterations 10 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block1_3.json --baseline-block5-path research/breakthrough_lab/platform_compatibility/week10_block2_2_preprod_scaled_long_20260208_173314.json --output-prefix week10_block2_4_wallclock_final`
  - Artifact JSON: `research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_final_20260208_183529.json`
  - Artifact MD: `research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_final_20260208_183529.md`

Rollback runbook (por falla de guardrail):

- `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
  - Note: `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_183535.md`
  - Gate post-rollback: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_183555.json` (`promote`)

Gate canonico antes de decision final:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_183620.json`
  - Decision: `promote`

Dashboard refresh:

- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json --block6-path research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json --block10-path research/breakthrough_lab/platform_compatibility/week10_block2_4_wallclock_final_20260208_183529.json`
  - Artifact: `research/breakthrough_lab/week9_comparative_dashboard_20260208_183651.json`
  - Decision: `iterate`

## Resultados

### Canary final (40 min)

- Decision: `iterate`
- Failed checks: `["t5_guardrails_all_runs"]`
- Wall-clock target/actual: `40.0/40.0` minutos
- Snapshots: `8`
- Max error: `0.000579833984375`
- T3 fallback max: `0.0`
- T5:
  - disable events total: `2`
  - overhead max: `8.8524%`
  - overhead mean: `1.4417%`
- Ratio minimo rusticl/clover: `0.9183` (pasa umbral de ratio)

### Operacion y seguridad

- Rollback aplicado inmediatamente segun runbook.
- Gate canonico post-rollback: `promote`.
- Gate canonico pre-decision final: `promote`.

## Decision Formal

Tracks:

- `week10_block2_4_wallclock_final`: **iterate**
- `week10_block2_4_rollback_and_recovery_gate`: **promote**
- `week10_block2_4_final_go_no_go`: **no-go**

Block decision:

- **iterate / no-go**

Razonamiento:

- El guardrail duro de T5 no se sostuvo en horizonte largo (`disable_events > 0` y pico de overhead > hard limit).
- La disciplina operacional si fue correcta (rollback + gates), por lo que la recomendacion queda en `NO-GO` hasta hardening T5 adicional.

## Estado del Bloque

`Week 10 - Block 2.4` cerrado en `iterate` con `NO-GO` formal para promocion.
