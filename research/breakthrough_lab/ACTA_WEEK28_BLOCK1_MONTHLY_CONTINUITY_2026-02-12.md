# Acta Week 28 - Block 1 (Continuidad operativa mensual)

- Date: 2026-02-12
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar ciclo mensual recurrente contra baseline Week 27,
  - validar replay semanal + split Clover/rusticl,
  - mantener gate canonico obligatorio pre/post.

## Objetivo

1. Confirmar estabilidad recurrente contra baseline Week 27.
2. Verificar guardrails T5 y ratio de plataforma sin regresiones.
3. Dejar base formal para Week 28 - Block 2.

## Ejecucion

Comando ejecutado:

- `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block1_monthly_cycle.py --baseline-path research/breakthrough_lab/week27_controlled_rollout/week27_block1_monthly_continuity_20260212_145205.json --output-dir research/breakthrough_lab/week28_controlled_rollout --preprod-signoff-dir research/breakthrough_lab/week28_controlled_rollout --output-prefix week28_block1_monthly_continuity --report-dir research/breakthrough_lab/week8_validation_discipline`

## Artefactos

- Report JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_20260212_152412.json`
- Report MD: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_20260212_152412.md`
- Weekly replay JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_weekly_replay_20260212_151910.json`
- Weekly replay eval JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_weekly_replay_eval_20260212_152217.json`
- Split canary JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_split_canary_20260212_152412.json`
- Split eval JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_split_eval_20260212_152412.json`
- Dashboard JSON: `research/breakthrough_lab/week28_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260212_152412.json`
- Dashboard MD: `research/breakthrough_lab/week28_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260212_152412.md`
- Manifest: `research/breakthrough_lab/week28_controlled_rollout/WEEK20_BLOCK1_MONTHLY_CYCLE_MANIFEST.json`
- Canonical gates internos (pre/post):
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_151910.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_152432.json`

## Resultados

- `decision = promote`
- `failed_checks = []`
- `weekly_replay_decision = promote`
- `split_canary_decision = promote`
- `split_eval_decision = promote`
- `split_ratio_min = 0.922897`
- `split_t5_overhead_max = 1.309204`
- `split_t5_disable_total = 0`

## Decision Formal

Tracks:

- `week28_block1_monthly_cycle_execution`: **promote**
- `week28_block1_platform_split_guardrails`: **promote**
- `week28_block1_operational_package`: **promote**
- `week28_block1_canonical_gate_internal`: **promote**

Block decision:

- **promote**

Razonamiento:

- Week 28 Block 1 confirma continuidad estable contra Week 27, sin disable events y con ratio de plataforma saludable.

## Estado del Bloque

`Week 28 - Block 1` cerrado en `promote`.
