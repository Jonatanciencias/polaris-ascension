# Week 12 Block 1 - Weekly Replay Automation

- Date: 2026-02-09T13:25:38.772446+00:00
- Mode: `local`
- Policy: `research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Sizes: [1400, 2048, 3072]
- Snapshots: 10

## Steps

| Step | Return code | Decision |
| --- | ---: | --- |
| pre_validation | 0 | promote |
| canary_run | 0 | promote |
| policy_eval | 0 | promote |
| post_validation | 0 | promote |

## Artifacts

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.md`
- `eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_eval_20260209_132519.json`
- `eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_eval_20260209_132519.md`
- `pre_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_131909.json`
- `post_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_132538.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Automated weekly replay completed with promote in canary, policy evaluation, and both canonical gates.

