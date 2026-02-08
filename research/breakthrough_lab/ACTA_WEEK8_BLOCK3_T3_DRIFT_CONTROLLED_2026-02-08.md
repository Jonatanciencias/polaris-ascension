# Acta Week 8 - Block 3 (T3 Drift Controlado + Guardrails Versionados)

- Date: 2026-02-08
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: campañas de drift controlado (`warm/cold/queue pressure`) en `auto` vs `auto_t3_controlled`, con versionado de thresholds rollback-safe.

## Objetivo

1. Ejecutar campaña formal T3 con escenarios `cold`, `warm` y `warm_queue_pressure`.
2. Definir/versionar policy de guardrails T3 con defaults seguros de rollback.
3. Cerrar Week 8 Block 3 con evidencia reproducible y decisión formal por track.

## Implementación

Cambios aplicados:

- `research/breakthrough_lab/t3_online_control/policy_hardening_block3.json`
  - policy versionada `t3-controlled-week8-block3-2026-02-08` con thresholds de drift y gate de promoción.
- `research/breakthrough_lab/t3_online_control/rollback_playbook_block3.md`
  - playbook operativo de rollback/disable para breaches de guardrails.
- `research/breakthrough_lab/t3_online_control/run_week8_t3_drift_campaign.py`
  - runner estricto con escenarios `cold/warm/warm_queue_pressure` y presión de cola por pulsos deterministas.
- `research/breakthrough_lab/t3_online_control/results.json`
  - normalizado a `schema_version=1.0.0` con cierre del bloque Week 8 Block 3.
- `research/breakthrough_lab/t3_online_control/report.md`
  - actualizado al estado de promoción del bloque.

## Ejecución Formal

Commands:

- `./venv/bin/python research/breakthrough_lab/t3_online_control/run_week8_t3_drift_campaign.py --sessions 2 --iterations 6 --sizes 1400 1536 2048 --seed 42 --pressure-size 896 --pressure-iterations 3 --pressure-pulses 2 --pressure-pause-ms 20`
- `./venv/bin/python scripts/validate_breakthrough_results.py`

Artifacts:

- `research/breakthrough_lab/t3_online_control/week8_t3_drift_campaign_20260208_020148.json`
- `research/breakthrough_lab/t3_online_control/week8_t3_drift_campaign_20260208_020148.md`
- `research/breakthrough_lab/t3_online_control/policy_hardening_block3.json`
- `research/breakthrough_lab/t3_online_control/rollback_playbook_block3.md`
- `research/breakthrough_lab/t3_online_control/results.json`
- `research/breakthrough_lab/t3_online_control/report.md`

## Resultados

- Decision del campaign: `promote` (checks de guardrails sin fallos).
- En `warm_queue_pressure`: T3 vs auto `+19.487%` avg GFLOPS y `-20.313%` en delta p95.
- `fallback_rate=0.000`, `correctness_failures=0`, `policy_disabled=false`.
- `max_error` observado global: `0.000640869140625` (< `0.001`).

## Decisión Formal

Track `t3_online_control`: **promote**.

Razonamiento:

- El bloque pasa correctness + estabilidad + guardrails de drift bajo presión controlada.
- Los defaults de rollback quedan versionados y auditables para operación segura.
- Se mantiene modo determinista/reproducible para comparación `auto` vs `auto_t3_controlled`.

## Estado del Bloque

`Week 8 - Block 3` ejecutado con evidencia reproducible y decisión formal registrada.
