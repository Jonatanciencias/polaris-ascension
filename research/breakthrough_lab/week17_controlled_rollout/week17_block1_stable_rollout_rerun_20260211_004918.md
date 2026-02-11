# Week17 Block1 Stable Rollout

- Date: 2026-02-11T00:49:18.064297+00:00
- Scope: week17_block1_initial_stable_rollout
- Stable tag: `v0.15.0`
- Snapshots target: 10

## Artifacts

- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_003858.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_004918.json`
- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.md`
- `checklist_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK17_BLOCK1_GO_NO_GO_CHECKLIST.md`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| stable_runbook_exists | True |
| pre_gate_promote | True |
| canary_returncode_zero | True |
| canary_promote | True |
| canary_t5_disable_zero | True |
| extended_snapshots_reached | True |
| rollback_dry_run_ok | True |
| rollback_sla_exists | True |
| post_gate_promote | True |

## Decision

- Decision: `go`
- Failed checks: []
- Rationale: Initial stable rollout reached extended horizon with green gates and rollback readiness.

