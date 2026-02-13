# Week 21 Block 1 - Monthly Continuity Runbook

- Cadence: mensual recurrente.
- Scope: replay semanal + split Clover/rusticl + consolidacion de continuidad contra baseline Week 20.
- Active policy: `research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Baseline reference: `research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_split_canary_20260211_140718.json`

## Mandatory Flow

1. Canonical gate pre: `scripts/run_validation_suite.py --tier canonical --driver-smoke`.
2. Ejecutar ciclo mensual recurrente con runner de continuidad.
3. Verificar split ratio y guardrails T5 en `promote`.
4. Publicar dashboard + acta + decision formal.
5. Canonical gate post y cierre de bloque.

## Rollback SLA

- `t5_disable_total > 0`: rollback inmediato al baseline Week 20.
- `rusticl_ratio_min < 0.85`: congelar expansión cross-platform.
- Gate canónico pre/post no promote: bloquear promoción.
