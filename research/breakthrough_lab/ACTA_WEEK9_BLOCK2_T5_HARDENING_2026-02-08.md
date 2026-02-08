# Acta Week 9 - Block 2 (T5 Hardening + Strict Rerun)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: hardening de T5 para eliminar auto-disable espurio en horizonte largo y rerun estricto del canary mixto.

## Objetivo

1. Mitigar auto-disable por pico aislado de overhead en T5 observado en Week 9 Block 1.
2. Mantener guardrails estrictos de correctness/fp/overhead sin degradar estabilidad.
3. Reejecutar el mismo canary estricto (mismos parametros) para buscar `promote`.

## Cambios Tecnicos

- Hardening runtime T5 en `src/optimization_engines/t5_abft_guardrails.py`:
  - histéresis configurable por guardrail (`disable_after_consecutive_*`),
  - umbral hard opcional para overhead (`disable_if_effective_overhead_percent_gt_hard`),
  - soporte de histéresis stateful entre invocaciones (`stateful_hysteresis` + `violation_streaks`).
- Cobertura unitaria extendida:
  - `tests/test_t5_abft_guardrails.py` ahora valida histéresis consecutiva y restauracion stateful.
- Policy candidata del bloque:
  - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json`
  - clave: `disable_after_consecutive_overhead_violations = 2`, `stateful_hysteresis = true`, `disable_if_effective_overhead_percent_gt_hard = 5.0`.
- Runner de canary actualizado para recibir policy/state path explicitos:
  - `research/breakthrough_lab/run_week9_block1_long_canary.py`

## Ejecucion Formal

Commands:

- `./venv/bin/python -m pytest -q tests/test_t5_abft_guardrails.py`
- `./venv/bin/python research/breakthrough_lab/run_week9_block1_long_canary.py --batches 24 --sessions-per-batch 1 --iterations-per-session 8 --sizes 1400 2048 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-batch 2 --seed 42 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json --t5-state-path results/runtime_states/t5_abft_guard_state_week9_block2_20260208.json --output-prefix week9_block2_long_canary_rerun`
- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

Artifacts:

- `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json`
- `research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_032043.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_032043.md`

## Resultados

Rerun estricto:
- Decision: `promote`
- Queue pressure: `48/48/0` (requested/completed/failures)
- Guardrails: **todos pass**
- T5 disable events totales: `0` (previo Block 1: `1`)

Comparativa relevante vs Block 1:
- `t5_disable_events_zero`: `False -> True`
- `t5_overhead_mean_percent`: `1.339% -> 1.313%`
- `t5_false_positive_rate_mean`: `0.0 -> 0.0`
- correctness T5 max: estable (`<= 0.0005799`)

Gate canonico obligatorio:
- `validation_suite canonical + driver_smoke`: **promote**

## Decision Formal

Tracks:
- `t5_hardening_block2`: **promote**
- `week9_long_mixed_canary_strict_rerun`: **promote**

Block decision:
- **promote**

Razonamiento:
- La mitigacion elimina el auto-disable espurio sin perder seguridad.
- El rerun estricto replica carga/escenario y cierra la deuda abierta en Block 1.

## Estado del Bloque

`Week 9 - Block 2` cerrado con `promote` y evidencia reproducible.
