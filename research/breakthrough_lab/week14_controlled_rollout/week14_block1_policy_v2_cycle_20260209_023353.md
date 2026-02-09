# Week 12 Block 1 - Weekly Replay Automation

- Date: 2026-02-09T02:38:20.431608+00:00
- Mode: `local`
- Policy: `research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Sizes: [1400, 2048, 3072]
- Snapshots: 8

## Steps

| Step | Return code | Decision |
| --- | ---: | --- |
| pre_validation | 0 | promote |
| canary_run | 0 | iterate |
| policy_eval | 0 | iterate |
| post_validation | 0 | promote |

## Artifacts

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_canary_20260209_023800.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_canary_20260209_023800.md`
- `eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_eval_20260209_023800.json`
- `eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_eval_20260209_023800.md`
- `pre_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_023413.json`
- `post_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_023820.json`

## Decision

- Decision: `iterate`
- Failed checks: ['policy_eval_not_promote']
- Rationale: Automated weekly replay found one or more non-promote steps.

