# Acta Week 12 - Block 2 (Split semanal Clover/rusticl contra policy formal)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar split semanal Clover/rusticl en mismo perfil operativo,
  - evaluar split contra policy formal Week 11,
  - cerrar con gate canónico obligatorio.

## Objetivo

1. Validar portabilidad operativa sin degradación relevante frente a Clover.
2. Mantener guardrails T3/T5 y correctness en ambos entornos.
3. Formalizar decisión semanal de split con evidencia trazable.

## Ejecución Formal

Split semanal:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 212 312 --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-seed 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-block3-path research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json --output-dir research/breakthrough_lab/week12_controlled_rollout --output-prefix week12_block2_platform_split`
  - Artifact JSON: `research/breakthrough_lab/week12_controlled_rollout/week12_block2_platform_split_20260209_012324.json`
  - Artifact MD: `research/breakthrough_lab/week12_controlled_rollout/week12_block2_platform_split_20260209_012324.md`
  - Decision: `promote`

Evaluación contra policy formal:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py --split-artifact research/breakthrough_lab/week12_controlled_rollout/week12_block2_platform_split_20260209_012324.json --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --min-rusticl-ratio 0.85 --output-dir research/breakthrough_lab/week12_controlled_rollout --output-prefix week12_block2_platform_split_eval`
  - Artifact JSON: `research/breakthrough_lab/week12_controlled_rollout/week12_block2_platform_split_eval_20260209_012330.json`
  - Artifact MD: `research/breakthrough_lab/week12_controlled_rollout/week12_block2_platform_split_eval_20260209_012330.md`
  - Decision: `promote`

Gate canónico obligatorio:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_012354.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_012354.md`
  - Decision: `promote`

## Resultados

- Split decision: `promote`
- Policy split eval: `promote`
- `failed_checks = []`
- `max_error = 0.0005035400390625`
- `t3_fallback_max = 0.0`
- `t5_disable_events_total = 0`
- `t5_overhead_max = 1.4662548939442208%`

Ratios rusticl/clover (avg GFLOPS):

- `auto_t3_controlled:1400` -> `0.9964`
- `auto_t3_controlled:2048` -> `0.9231`
- `auto_t5_guarded:1400` -> `1.0120`
- `auto_t5_guarded:2048` -> `0.9238`

Ratio mínimo observado: `0.9231` (`>= 0.85` requerido).

## Decisión Formal

Tracks:

- `week12_block2_platform_split_execution`: **promote**
- `week12_block2_platform_split_policy_eval`: **promote**
- `week12_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- Clover/rusticl cumplen guardrails y policy formal sin fallas.
- La portabilidad semanal queda validada con ratio mínimo por encima del umbral.

## Estado del Bloque

`Week 12 - Block 2` cerrado en `promote`.
