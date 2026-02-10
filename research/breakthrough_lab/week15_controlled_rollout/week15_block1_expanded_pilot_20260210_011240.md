# Week14 Block5 RX590 Dry-Run

- Date: 2026-02-10T01:12:40.678339+00:00
- Scope: rx590_low_scope_dry_run
- Rollback SLA: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Artifacts

- Canary JSON: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_canary_20260210_011220.json`
- Canary MD: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_canary_20260210_011220.md`
- Pre-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_010820.json`
- Post-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_011240.json`
- Checklist: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week15_controlled_rollout/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md`

## Checks

| Check | Pass |
| --- | --- |
| pre_gate_promote | True |
| canary_returncode_zero | True |
| canary_promote | False |
| canary_t5_disable_zero | False |
| rollback_dry_run_ok | True |
| rollback_sla_exists | True |
| post_gate_promote | True |

## Decision

- Decision: `no-go`
- Failed checks: ['canary_promote', 'canary_t5_disable_zero']
- Rationale: One or more dry-run gates failed; keep no-go and resolve before expansion.

