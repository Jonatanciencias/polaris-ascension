# Week 12 Block 1 - Weekly Replay Automation

- Date: 2026-02-11T14:05:43.104078+00:00
- Mode: `local`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
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

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_canary_20260211_140523.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_canary_20260211_140523.md`
- `eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_eval_20260211_140523.json`
- `eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_eval_20260211_140523.md`
- `pre_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140236.json`
- `post_validation_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140543.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Automated weekly replay completed with promote in canary, policy evaluation, and both canonical gates.

