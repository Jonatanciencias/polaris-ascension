# Acta Week 12 - Block 4 (Canary combinado split Clover/rusticl + 3072)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar canary combinado con split Clover/rusticl y tamaño `3072`,
  - validar con el mismo policy formal (`policy_week11_block3_weekly_slo_v1.json`),
  - cerrar decisión formal con gate canónico obligatorio.

## Objetivo

1. Confirmar compatibilidad operativa ampliada (`1400/2048/3072`) en Clover/rusticl.
2. Mantener guardrails globales y ratio rusticl/clover dentro de umbral.
3. Cerrar bloque con evidencia y decisión `promote|iterate`.

## Ejecución Formal

Canary combinado split + 3072:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 412 512 --sizes 1400 2048 3072 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-seed 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-block3-path research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json --output-dir research/breakthrough_lab/week12_controlled_rollout --output-prefix week12_block4_combined_split_3072`
  - Artifact JSON: `research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_20260209_013814.json`
  - Artifact MD: `research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_20260209_013814.md`
  - Decision: `promote`

Evaluación formal contra policy:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py --split-artifact research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_20260209_013814.json --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --min-rusticl-ratio 0.85 --required-sizes 1400 2048 3072 --output-dir research/breakthrough_lab/week12_controlled_rollout --output-prefix week12_block4_combined_split_3072_eval`
  - Artifact JSON: `research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_eval_20260209_013824.json`
  - Artifact MD: `research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_eval_20260209_013824.md`
  - Decision: `promote`

Gate canónico obligatorio:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_013850.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_013850.md`
  - Decision: `promote`

## Resultados

- Split decision: `promote`
- Policy eval decision: `promote`
- `failed_checks = []`
- `max_error = 0.0008392333984375`
- `t3_fallback_max = 0.0`
- `t5_disable_events_total = 0`
- `t5_overhead_max = 1.4369515868357043%`

Ratios rusticl/clover (promedios):

- `auto_t3_controlled:3072` -> `0.9314`
- `auto_t5_guarded:3072` -> `0.9298`
- Ratio mínimo global observado: `0.9242` (`>= 0.85` requerido)

## Decisión Formal

Tracks:

- `week12_block4_combined_split_execution`: **promote**
- `week12_block4_combined_split_policy_eval`: **promote**
- `week12_block4_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El canary combinado pasa con guardrails intactos y ratio rusticl/clover por encima del mínimo.
- La evaluación formal y el gate canónico quedan en `promote`.

## Estado del Bloque

`Week 12 - Block 4` cerrado en `promote`.
