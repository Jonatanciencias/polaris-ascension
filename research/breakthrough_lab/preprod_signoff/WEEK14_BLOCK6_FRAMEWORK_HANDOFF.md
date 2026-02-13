# Week14 Block6 Framework Handoff

## Objective

Consolidate a stable framework baseline for extension/plugin teams and dependent projects.

## Mandatory Operational Inputs

- Weekly policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- RX590 pre-release runbook: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK4_RX590_PRERELEASE_RUNBOOK.md`
- Plugin/project checklist: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK4_PLUGIN_PROJECT_BASE_CHECKLIST.md`
- Block5 go/no-go checklist: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md`
- Rollback SLA: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Handoff Rules

1. No plugin promotion without canonical gate in `promote`.
2. No extension can bypass policy guardrails (`T3/T4/T5`).
3. Every extension must emit JSON + MD artifacts and formal decision state.
4. Rollback path must remain executable before any scope expansion.
5. RX590 controlled profile remains baseline reference for compatibility.

## Recommended Starter Sequence

1. Implement plugin skeleton with deterministic seed protocol.
2. Add plugin to benchmark matrix with `auto_t3_controlled` and `auto_t5_guarded` cross-check.
3. Run canonical validation gate and dry-run profile.
4. Produce acta + decision JSON with promote/iterate outcome.

