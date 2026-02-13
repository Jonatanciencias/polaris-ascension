# Acta Week 9 - Block 4 (Stress Replay Corto + Pulsos de Cola + Split Plataforma)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: validar robustez post-hardening con replay corto bajo presion de cola y split `Clover`/`rusticl`.

## Objetivo

1. Ejecutar replay corto con pulsos de cola para tensionar el selector en condiciones realistas.
2. Verificar guardrails T3/T5 por plataforma sin regresiones vs baseline de Week 9 Block 3.
3. Cerrar bloque con gate canonico obligatorio.

## Implementacion

Nuevo runner del bloque:

- `research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py`

Capacidades:
- split dual plataforma (`Clover`/`rusticl`),
- pulsos de cola pre-run por plataforma/semilla,
- policy T5 endurecida de Week 9 Block 2,
- chequeos de no-regresion vs baseline Block 3 (Clover).

## Ejecucion Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 11 77 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 6 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-seed 2`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py`
- `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.json`
- `research/breakthrough_lab/platform_compatibility/week9_block4_stress_split_20260208_033946.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_034047.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_034047.md`

## Resultados

Stress replay:
- Decision: `promote`
- Runs OK: `16/16`
- Pressure pulses: `8/8` completados, `0` fallos
- Correctness max global: `0.0005645751953125` (`<= 1e-3`)

Guardrails:
- T3: fallback max `0.0`, policy disabled total `0`
- T5: disable total `0`, fp max `0.0`, overhead max `2.583%` (`<= 3.0`)

Split plataforma:
- Ratio minimo rusticl/clover (peak): `0.9238` (`>= 0.80`)

No-regresion vs Block 3 (Clover):
- `auto_t3_controlled` delta throughput: `-0.021%`, delta p95: `+0.899%`
- `auto_t5_guarded` delta throughput: `-0.781%`, delta p95: `+0.215%`
- dentro de umbrales de no-regresion configurados.

Gate canonico obligatorio:
- `validation_suite canonical + driver_smoke`: **promote**

## Decision Formal

Tracks:
- `week9_block4_stress_replay`: **promote**
- `week9_block4_platform_split`: **promote**

Block decision:
- **promote**

Razonamiento:
- El sistema mantiene estabilidad y guardrails bajo presion de cola.
- No se detectan regresiones materiales respecto al baseline Week 9 Block 3.

## Estado del Bloque

`Week 9 - Block 4` cerrado con `promote`, evidencia reproducible y gate canonico en verde.
