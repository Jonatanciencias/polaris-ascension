# Week 17 Block 2 - Second Dependent Pilot (Stable Manifest)

- Date: 2026-02-11T01:11:49.521939+00:00
- Stable tag: `v0.15.0`
- Dependent project dir: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/dependent_projects/rx590_stable_integration_pilot_v2`
- Plugin ID: `rx590_dependent_project_week17_block2`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| stable_status_proposed_or_stable | True |
| stable_manifest_references_exist | True |
| plugin_template_exists | True |
| integration_profile_written | True |
| plugin_pilot_promote | True |
| plugin_results_exists | True |
| pre_gate_promote | True |
| post_gate_promote | True |

## Artifacts

- `integration_profile_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/dependent_projects/rx590_stable_integration_pilot_v2/week17_block2_integration_profile.json`
- `plugin_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block2_second_dependent_pilot_plugin_20260211_011129.json`
- `plugin_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block2_second_dependent_pilot_plugin_20260211_011129.md`
- `plugin_results_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week15_block2_plugin_pilot_results.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_011026.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_011149.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Second dependent pilot is stable over v0.15.0 manifest with mandatory gates green.

