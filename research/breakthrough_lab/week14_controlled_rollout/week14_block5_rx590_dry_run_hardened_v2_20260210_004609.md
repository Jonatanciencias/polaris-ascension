# Week14 Block5 RX590 Dry-Run

- Date: 2026-02-10T00:46:09.265484+00:00
- Scope: rx590_low_scope_dry_run
- Rollback SLA: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`

## Artifacts

- Canary JSON: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_hardened_v2_canary_20260210_004549.json`
- Canary MD: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block5_rx590_dry_run_hardened_v2_canary_20260210_004549.md`
- Pre-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_004249.json`
- Post-gate JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_004609.json`
- Checklist: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK5_GO_NO_GO_CHECKLIST.md`

## Checks

| Check | Pass |
| --- | --- |
| pre_gate_promote | True |
| canary_returncode_zero | True |
| canary_promote | True |
| canary_t5_disable_zero | True |
| rollback_dry_run_ok | True |
| rollback_sla_exists | True |
| post_gate_promote | True |

## Decision

- Decision: `go`
- Failed checks: []
- Rationale: Low-scope RX590 dry-run passed canary, canonical gates, and rollback readiness.

