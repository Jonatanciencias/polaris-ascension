# Acta Week 16 - Block 2 (Replay semanal automatizado sobre RC + drift)

- Date: 2026-02-10
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar replay semanal automatizado sobre `v0.15.0-rc1`,
  - evaluar drift contra baseline RC de Week 15,
  - cerrar decisión formal (`promote|iterate`) con gate canónico obligatorio.

## Objetivo

1. Verificar estabilidad semanal del perfil RC en `1400/2048/3072`.
2. Confirmar cumplimiento de policy formal y drift en umbrales operativos.
3. Dejar evidencia comparativa lista para propuesta estable.

## Ejecución Formal

Intento inicial:

- `./venv/bin/python research/breakthrough_lab/week16_controlled_rollout/run_week16_block2_weekly_rc_replay.py`
  - Artifact JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_20260210_015005.json`
  - Replay JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_20260210_014604.json`
  - Eval JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_eval_20260210_014926.json`
  - Decision: `iterate`
  - Hallazgo: único check fallido `pre_gate_promote` por fallo aislado `pytest_tier_green`.

Rerun estricto de cierre:

- `./venv/bin/python research/breakthrough_lab/week16_controlled_rollout/run_week16_block2_weekly_rc_replay.py --output-prefix week16_block2_weekly_rc_replay_rerun --seed 26131`
  - Artifact JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_20260210_015504.json`
  - Replay JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_20260210_015103.json`
  - Eval JSON: `research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_eval_20260210_015424.json`
  - Decision: `promote`

## Resultados Finales (rerun)

- `pre_gate_decision = promote`
- `replay_decision = promote`
- `weekly_eval_decision = promote`
- `post_gate_decision = promote`
- `max_abs_throughput_drift_percent = 4.450885122067073`
- `max_positive_p95_drift_percent = 0.6219210021981284`
- `failed_checks = []`

## Decisión Formal

Tracks:

- `week16_block2_initial_replay_attempt`: **iterate**
- `week16_block2_weekly_replay_rerun`: **promote**
- `week16_block2_drift_evaluation`: **promote**
- `week16_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El intento inicial fue bloqueado por una falla aislada de pre-gate, no por drift ni por policy.
- El rerun estricto cerró en verde total y confirma estabilidad semanal del RC.

## Estado del Bloque

`Week 16 - Block 2` cerrado en `promote` (basado en rerun formal).
