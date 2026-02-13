# Week 12 Block 1 - Weekly Replay Automation

- Date: 2026-02-11T01:54:42.753840+00:00
- Mode: `local`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Sizes: [1400, 2048, 3072]
- Snapshots: 6

## Steps

| Step | Return code | Decision |
| --- | ---: | --- |
| pre_validation | 0 | promote |
| canary_run | 0 | promote |
| policy_eval | 0 | promote |
| post_validation | 0 | promote |

## Artifacts

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_canary_20260211_015423.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_canary_20260211_015423.md`
- `eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_eval_20260211_015423.json`
- `eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_eval_20260211_015423.md`
- `pre_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_015134.json`
- `post_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_015442.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Automated weekly replay completed with promote in canary, policy evaluation, and both canonical gates.

