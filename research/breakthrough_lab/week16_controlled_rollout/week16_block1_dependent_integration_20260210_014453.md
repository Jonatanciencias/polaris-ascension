# Week 16 Block 1 - Dependent Project Integration Pilot

- Date: 2026-02-10T01:44:53.802010+00:00
- RC tag: `v0.15.0-rc1`
- Dependent project dir: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/dependent_projects/rx590_rc_integration_pilot`
- Plugin ID: `rx590_dependent_project_week16_block1`

## Checks

| Check | Pass |
| --- | --- |
| rc_manifest_exists | True |
| rc_manifest_status_candidate | True |
| rc_manifest_dependencies_exist | True |
| plugin_template_exists | True |
| integration_profile_written | True |
| plugin_pilot_promote | True |
| plugin_results_exists | True |
| pre_gate_promote | True |
| post_gate_promote | True |

## Artifacts

- `integration_profile_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/dependent_projects/rx590_rc_integration_pilot/week16_block1_integration_profile.json`
- `plugin_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block1_dependent_integration_plugin_pilot_20260210_014434.json`
- `plugin_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block1_dependent_integration_plugin_pilot_20260210_014434.md`
- `plugin_results_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week15_block2_plugin_pilot_results.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_014330.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_014453.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Dependent project consumed RC manifest and completed integration pilot with all mandatory gates green.

