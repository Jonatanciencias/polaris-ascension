# Acta Week 9 - Block 5 (Pre-Production Pilot RX590 + Rusticl Mirrored)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: piloto controlado pre-produccion en RX 590 con burn-in extendido y snapshots horarios, incluyendo piloto rusticl espejado con rollback explicito.

## Objetivo

1. Validar estabilidad extendida de T3/T5 en un horizonte de burn-in con snapshots horarios logicos.
2. Correr piloto rusticl espejado bajo los mismos criterios de guardrail.
3. Verificar plan de contingencia (rollback) con script explicito y prueba operativa.

## Implementacion

Nuevos assets del bloque:

- `research/breakthrough_lab/platform_compatibility/run_week9_block5_preprod_pilot.py`
- `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`
- `research/breakthrough_lab/build_week9_comparative_dashboard.py` (extendido para incluir `block5`)

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block5_preprod_pilot.py --snapshots 6 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 6 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 3 --seed 2026`
- `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
- `./venv/bin/python research/breakthrough_lab/build_week9_comparative_dashboard.py --block4-path research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json --block5-path research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json`

Artifacts:

- `research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json`
- `research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.md`
- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_035258.md`
- `results/runtime_states/week9_block5_runtime_env.sh`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_035317.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_035352.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_035352.md`

## Resultados

Burn-in extendido (6 snapshots horarios logicos):
- Decision: `promote`
- Runs OK: `48/48`
- Queue pulses: `36/36` completados, `0` fallos
- Correctness max: `0.000701904296875` (`<= 1e-3`)
- T3: fallback max `0.0`, policy disabled total `0`
- T5: disable total `0`, FP max `0.0`, overhead max `2.604%` (`<= 3.0`)
- Rusticl/Clover ratio minimo (peak): `0.9179` (`>= 0.80`)
- Drift burn-in (abs): dentro de `12%` en todas las combinaciones

Piloto rusticl espejado:
- Ejecutado en el mismo runner y passed en todos los snapshots.
- Sin disable events T5 ni violaciones de correctness.

Rollback explicito:
- Script ejecutado en `apply`:
  - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
- Resultado:
  - runtime profile generado (`results/runtime_states/week9_block5_runtime_env.sh`)
  - nota de rollback generada (`week9_block5_rollback_20260208_035258.md`)
  - gate canonico posterior: **promote**

Dashboard comparativo actualizado:
- `week9_comparative_dashboard_20260208_035352.json` => `promote`
- Cadena activa Week9 (Block2/3/4/5) permanece `promote`.

## Decision Formal

Tracks:
- `week9_block5_preprod_burnin`: **promote**
- `week9_block5_rusticl_mirror`: **promote**
- `week9_block5_rollback_operability`: **promote**

Block decision:
- **promote**

Razonamiento:
- El piloto extendido confirma estabilidad operativa en RX590 con split de plataforma.
- El plan de rollback esta implementado y validado operativamente.

## Estado del Bloque

`Week 9 - Block 5` cerrado con `promote`, evidencia reproducible y plan de rollback validado.
