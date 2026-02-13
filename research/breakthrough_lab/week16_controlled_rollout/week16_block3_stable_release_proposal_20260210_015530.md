# Week 16 Block 3 - Stable Release Proposal

- Date: 2026-02-10T01:55:30.891534+00:00
- RC tag: `v0.15.0-rc1`
- Proposed tag: `v0.15.0`

## Checks

| Check | Pass |
| --- | --- |
| rc_manifest_exists | True |
| rc_source_is_expected | True |
| block1_promote | True |
| block2_promote | True |
| canonical_gate_promote | True |
| release_docs_written | True |

## Artifacts

- `block1_report_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block1_dependent_integration_20260210_014453.json`
- `block2_report_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_20260210_015504.json`
- `release_notes_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_RELEASE_NOTES.md`
- `release_checklist_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_RELEASE_CHECKLIST.md`
- `release_runbook_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_RELEASE_RUNBOOK.md`
- `stable_manifest_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json`
- `canonical_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_015530.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: RC lineage satisfies all gates and is ready to be proposed as stable v0.15.0.

