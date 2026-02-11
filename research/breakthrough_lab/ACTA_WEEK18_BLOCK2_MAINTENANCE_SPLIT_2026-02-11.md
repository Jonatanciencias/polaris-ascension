# Acta Week 18 - Block 2 (Canary de mantenimiento split Clover/rusticl)

- Date: 2026-02-11
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar canary de mantenimiento con split Clover/rusticl,
  - validar contra baseline estable `v0.15.0`,
  - mantener gate canonico obligatorio antes y despues del bloque.

## Objetivo

1. Verificar continuidad operacional estable en split por plataforma.
2. Confirmar que guardrails T3/T5 y policy semanal formal se mantienen en `promote`.
3. Cerrar bloque con evidencia reproducible y decision formal.

## Ejecucion Formal

Canary split de mantenimiento:

- `./venv/bin/python research/breakthrough_lab/week18_controlled_rollout/run_week18_block2_maintenance_split.py`
  - Artifact JSON: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_20260211_014255.json`
  - Artifact MD: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_20260211_014255.md`
  - Canary split JSON: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_canary_20260211_014235.json`
  - Canary split MD: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_canary_20260211_014235.md`
  - Split eval JSON: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_eval_20260211_014235.json`
  - Split eval MD: `research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_eval_20260211_014235.md`
  - Canonical gate pre JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_014059.json`
  - Canonical gate post JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_014255.json`
  - Decision: `promote`

## Resultados

- `stable_tag = v0.15.0`
- `canary_decision = promote`
- `split_eval_decision = promote`
- `rusticl_ratio_min = 0.9181`
- `canary_t5_overhead_max = 1.4023`
- `canary_t5_disable_total = 0`
- `pre_gate_decision = promote`
- `post_gate_decision = promote`
- `failed_checks = []`

## Decision Formal

Tracks:

- `week18_block2_stable_split_canary_execution`: **promote**
- `week18_block2_policy_eval_and_ratio_floor`: **promote**
- `week18_block2_t5_guardrails_on_split`: **promote**
- `week18_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El canary split de mantenimiento pasa sin regressiones ni disable events, con ratio rusticl/clover saludable y gates canonicos en verde.

## Estado del Bloque

`Week 18 - Block 2` cerrado en `promote`.
