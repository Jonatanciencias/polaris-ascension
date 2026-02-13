# Week 12 Block 1 - Weekly Replay Automation

- Date: 2026-02-13T17:20:52.837738+00:00
- Mode: `local`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Sizes: [1400, 2048, 3072]
- Snapshots: 8

## Steps

| Step | Return code | Decision |
| --- | ---: | --- |
| pre_validation | 0 | promote |
| canary_run | 0 | promote |
| policy_eval | 0 | iterate |
| post_validation | 0 | promote |

## Artifacts

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_canary_20260213_172033.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_canary_20260213_172033.md`
- `eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_eval_20260213_172033.json`
- `eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_eval_20260213_172033.md`
- `pre_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_171737.json`
- `post_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_172052.json`

## Decision

- Decision: `iterate`
- Failed checks: ['policy_eval_not_promote']
- Rationale: Automated weekly replay found one or more non-promote steps.

