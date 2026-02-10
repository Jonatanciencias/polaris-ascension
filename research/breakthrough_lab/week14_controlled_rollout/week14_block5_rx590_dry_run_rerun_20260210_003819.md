# Week14 Block5 RX590 Dry-Run

- Date: 2026-02-10T00:38:19.868061+00:00
- Scope: rx590_low_scope_dry_run
- Rollback SLA: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Artifacts

- Canary JSON: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_rerun_canary_20260210_003800.json`
- Canary MD: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_rerun_canary_20260210_003800.md`
- Pre-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_003459.json`
- Post-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_003819.json`
- Checklist: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md`

## Checks

| Check | Pass |
| --- | --- |
| pre_gate_promote | True |
| canary_returncode_zero | True |
| canary_promote | False |
| canary_t5_disable_zero | True |
| rollback_dry_run_ok | True |
| rollback_sla_exists | True |
| post_gate_promote | True |

## Decision

- Decision: `no-go`
- Failed checks: ['canary_promote']
- Rationale: One or more dry-run gates failed; keep no-go and resolve before expansion.

