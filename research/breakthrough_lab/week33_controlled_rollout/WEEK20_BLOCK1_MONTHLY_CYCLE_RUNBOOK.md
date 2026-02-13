# Week 20 Block 1 - Monthly Full Cycle Runbook

- Cadence: mensual, ventana operativa de continuidad.
- Scope: replay semanal automatizado + split Clover/rusticl + consolidacion formal.
- Active policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Dashboard reference: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week33_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260213_040810.json`

## Mandatory Flow

1. Canonical gate pre: `run_validation_suite.py --tier canonical --driver-smoke`.
2. Ejecutar replay semanal automatizado sobre baseline estable.
3. Ejecutar split Clover/rusticl y evaluacion contra policy activa.
4. Consolidar dashboard + manifest + matriz de deuda operativa.
5. Canonical gate post y cierre formal (acta + decision).

## Rollback SLA

- Si `split_t5_disable_total > 0`: rollback inmediato a policy previa.
- Si ratio rusticl/clover cae por debajo del piso: congelar expansion cross-platform.
- Si gate canonico pre/post no es promote: bloquear cierre del bloque.

