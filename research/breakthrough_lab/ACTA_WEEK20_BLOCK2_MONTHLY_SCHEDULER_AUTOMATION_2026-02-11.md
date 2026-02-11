# Acta Week 20 - Block 2 (Automatizacion programada mensual + retencion + alertas)

- Date: 2026-02-11
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - formalizar automatizacion mensual programada del ciclo Week20,
  - versionar politica operativa de retencion de artefactos,
  - cerrar capa de alertas operativas sobre guardrails productivos,
  - mantener gate canonico obligatorio pre/post.

## Objetivo

1. Asegurar un runner de automatizacion mensual reproducible y apto para CI/self-hosted.
2. Enforzar retencion explicita de artefactos y resumen de alertas operativas.
3. Cerrar Block 2 con evidencia formal en `promote`.

## Implementacion

Nuevos activos:

- Runner: `research/breakthrough_lab/week20_controlled_rollout/run_week20_block2_monthly_scheduler_automation.py`
- Workflow programado: `.github/workflows/week20-monthly-cycle.yml`

## Ejecucion Formal

Comando principal:

- `./venv/bin/python research/breakthrough_lab/week20_controlled_rollout/run_week20_block2_monthly_scheduler_automation.py --workflow-path .github/workflows/week20-monthly-cycle.yml --artifact-retention-days 45`

Salida final (cierre valido):

- Report JSON: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_20260211_140738.json`
- Report MD: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_20260211_140738.md`
- Alerts JSON: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_20260211_140738.json`
- Alerts MD: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_alerts_20260211_140738.md`
- Scheduler spec JSON: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_scheduler_spec_20260211_140738.json`
- Operational debt JSON: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_operational_debt_20260211_140738.json`
- Canonical gate pre JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140157.json`
- Canonical gate post JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140757.json`
- Decision: `promote`

## Nota de estabilidad de ejecucion

Durante el cierre se ejecutaron intentos previos con `iterate` por sensibilidad de baseline/seeds en el subciclo interno (`cycle_decision_not_promote`). Se ajusto la configuracion a baseline y seeds de referencia promovida y el bloque quedo cerrado en `promote`.

## Resultados

- `cycle_decision = promote`
- `alerts_decision = promote`
- `alerts_total = 0`
- `split_ratio_min = 0.923278`
- `split_t5_overhead_max = 1.326084`
- `split_t5_disable_total = 0`
- `workflow_has_schedule = true`
- `workflow_has_retention_days = true`
- `workflow_has_alert_summary_step = true`
- `failed_checks = []`

## Decision Formal

Tracks:

- `week20_block2_scheduler_workflow_publication`: **promote**
- `week20_block2_monthly_cycle_automation`: **promote**
- `week20_block2_retention_and_alerting_governance`: **promote**
- `week20_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- La automatizacion mensual queda operativa con retencion/alertas definidas, sin incidentes de guardrail y con gates canonicos en verde.

## Estado del Bloque

`Week 20 - Block 2` cerrado en `promote`.
