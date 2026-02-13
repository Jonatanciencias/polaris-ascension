# Week 27 Release Package Runbook

- Date: 2026-02-12
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: cierre operativo del paquete de release post Week27 Block1/2/3.

## Preconditions

1. `week27_block1_monthly_continuity_decision.json` en `promote`.
2. `week27_block2_alert_bridge_observability_decision.json` en `promote`.
3. `week27_block3_biweekly_comparative_decision.json` en `promote`.
4. Gate canonico pre/post en verde para cada bloque.

## Promotion Flow

1. Verificar policy dual plataforma activa en `WEEK27_BLOCK3_PLATFORM_POLICY_DECISION.json`.
2. Validar deuda operativa en `WEEK27_BLOCK3_OPERATIONAL_DEBT_REVIEW.json` (`high/critical open = 0`).
3. Confirmar guardrails:
   - `split_t5_disable_total = 0`
   - `split_ratio_min >= 0.85`
   - `cycle_success_ratio = 1.0`
4. Registrar decision final de paquete y actualizar roadmap.

## Rollback Policy

- Si aparece `t5_disable_total > 0`: rollback inmediato a baseline Week26.
- Si `cycle_success_ratio < 0.95`: desactivar bridge live y volver a fallback operacional.
- Si ratio rusticl/clover cae bajo policy floor: congelar promoción multi-plataforma.
