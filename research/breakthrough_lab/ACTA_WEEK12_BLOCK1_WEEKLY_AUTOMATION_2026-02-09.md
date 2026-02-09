# Acta Week 12 - Block 1 (Automatización Weekly Replay local/CI)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - automatizar replay semanal con ejecución reproducible local/CI,
  - enlazar ejecución a policy formal `policy_week11_block3_weekly_slo_v1.json`,
  - dejar job programado en GitHub Actions y evidencia ejecutada local.

## Objetivo

1. Estandarizar ejecución semanal (`canary -> eval policy -> gate`) con un solo entrypoint.
2. Habilitar modo programado para CI/self-hosted GPU.
3. Mantener gate canónico obligatorio antes y después del replay.

## Implementación

Automatización creada:

- Script: `research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py`
- Workflow programado: `.github/workflows/week12-weekly-replay.yml`

Características:

- Pre-gate canónico.
- Canary operativo controlado.
- Evaluación automática contra policy formal.
- Post-gate canónico.
- Reporte consolidado JSON/MD del bloque.

## Ejecución Formal

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-path research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json --sizes 1400 2048 --snapshots 6 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --seed 12011 --output-dir research/breakthrough_lab/week12_controlled_rollout --output-prefix week12_block1_weekly_automation`

Artifacts:

- `research/breakthrough_lab/week12_controlled_rollout/week12_block1_weekly_automation_20260209_011907.json`
- `research/breakthrough_lab/week12_controlled_rollout/week12_block1_weekly_automation_20260209_011907.md`
- `research/breakthrough_lab/week12_controlled_rollout/week12_block1_weekly_automation_canary_20260209_012142.json`
- `research/breakthrough_lab/week12_controlled_rollout/week12_block1_weekly_automation_eval_20260209_012142.json`
- Gate pre: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_011926.json` (`promote`)
- Gate post: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_012201.json` (`promote`)

## Resultados

- Block decision: `promote`
- `snapshots = 6/6`
- `rollback = false`
- `max_error = 0.0005950927734375`
- `t3_fallback_max = 0.0`
- `t5_disable_events_total = 0`
- `t5_overhead_max = 1.3319379671490623%`
- Policy eval: `promote` con `failed_checks = []`

## Decisión Formal

Tracks:

- `week12_block1_automation_stack_local_ci`: **promote**
- `week12_block1_automated_weekly_replay_run`: **promote**
- `week12_block1_mandatory_canonical_gates`: **promote**

Block decision:

- **promote**

Razonamiento:

- El replay semanal queda automatizado y reproducible en local/CI.
- La ejecución real y los dos gates canónicos cierran en `promote`.

## Estado del Bloque

`Week 12 - Block 1` cerrado en `promote`.
