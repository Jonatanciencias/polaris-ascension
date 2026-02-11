# Week 19 Block 3 - Monthly Continuity Runbook

- Cadence: mensual (primer dia habil de cada mes).
- Scope: replay semanal + split Clover/rusticl + consolidacion de drift.
- Active policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Dashboard reference: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block3_monthly_continuity_dashboard_20260211_020812.json`

## Mandatory Flow

1. Run canonical gate pre (`run_validation_suite.py --tier canonical --driver-smoke`).
2. Run weekly replay automation on stable baseline.
3. Run Clover/rusticl split canary and policy evaluation.
4. Run biweekly drift review/recalibration package.
5. Run canonical gate post and close acta + decision.

## Rollback Rules

- If `t5_disable_total > 0`, stop promotion and rollback to last known good policy.
- If rusticl/clover ratio falls below policy floor, stop expansion.
- If canonical gate is not promote, block closure.

