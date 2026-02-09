# Acta Week 13 - Block 1 (Producción controlada ampliada + comparativo quincenal)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar fase de producción controlada ampliada con snapshots extendidos,
  - mantener tamaño ampliado (`1400`, `2048`, `3072`) y guardrails activos,
  - generar reporte comparativo quincenal formal.

## Objetivo

1. Confirmar estabilidad en horizonte extendido (`8 snapshots`).
2. Mantener cumplimiento policy formal en guardrails críticos.
3. Entregar comparativo quincenal machine-readable para seguimiento de tendencia.

## Ejecución Formal

Fase controlada ampliada:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-path research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json --sizes 1400 2048 3072 --snapshots 8 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --seed 13011 --output-dir research/breakthrough_lab/week13_controlled_rollout --output-prefix week13_block1_extended_controlled`

Artifacts principales:

- `research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_20260209_014026.json`
- `research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json`
- `research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_eval_20260209_014522.json`

Gates canónicos:

- pre: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_014046.json` (`promote`)
- post: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_014541.json` (`promote`)
- cierre: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_014803.json` (`promote`)

Comparativo quincenal:

- `./venv/bin/python research/breakthrough_lab/week13_controlled_rollout/build_week13_biweekly_comparative_report.py --baseline-canary research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json --current-canary research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json --output-dir research/breakthrough_lab/week13_controlled_rollout --output-prefix week13_block1_biweekly_comparative`
  - Artifact JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block1_biweekly_comparative_20260209_014733.json`
  - Artifact MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block1_biweekly_comparative_20260209_014733.md`
  - Decision: `promote`

## Resultados

Fase ampliada:

- Decision: `promote`
- `snapshots = 8`
- `rollback = false`
- `max_error = 0.000885009765625`
- `t3_fallback_max = 0.0`
- `t5_disable_events_total = 0`
- `t5_overhead_max = 1.4781648825890927%`

Drift extendido:

- `auto_t3_controlled:3072` -> `-1.6483%`
- `auto_t5_guarded:3072` -> `+0.0119%`
- Drift máximo absoluto global: `1.6483%` (`<= 3.0%`)

Comparativo quincenal:

- Decision: `promote`
- `failed_checks = []`
- Deltas contenidos (sin regresión material):
  - `auto_t3_controlled:1400` -> `-0.1179%`
  - `auto_t3_controlled:2048` -> `-0.0416%`
  - `auto_t3_controlled:3072` -> `+0.1092%`
  - `auto_t5_guarded:1400` -> `+0.4156%`
  - `auto_t5_guarded:2048` -> `-0.2959%`
  - `auto_t5_guarded:3072` -> `+0.1702%`

## Decisión Formal

Tracks:

- `week13_block1_extended_controlled_execution`: **promote**
- `week13_block1_policy_eval_extended_horizon`: **promote**
- `week13_block1_biweekly_comparative_report`: **promote**
- `week13_block1_mandatory_canonical_gates`: **promote**

Block decision:

- **promote**

Razonamiento:

- La fase ampliada mantiene estabilidad en `8 snapshots` y set de tamaños extendido.
- El comparativo quincenal confirma continuidad operativa sin drift crítico.

## Estado del Bloque

`Week 13 - Block 1` cerrado en `promote`.
