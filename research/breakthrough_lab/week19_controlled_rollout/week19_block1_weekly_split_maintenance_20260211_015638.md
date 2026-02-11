# Week 19 Block 1 - Weekly Replay + Split Maintenance

- Date: 2026-02-11T01:56:38.890502+00:00
- Stable tag: `v0.15.0`
- Policy path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- T5 policy path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json`
- Baseline path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_canary_20260211_014235.json`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| weekly_replay_promote | True |
| weekly_replay_canary_promote | True |
| weekly_replay_eval_promote | True |
| split_canary_promote | True |
| split_eval_promote | True |
| split_required_sizes_present | True |
| split_ratio_floor | True |
| split_t5_guardrails | True |
| split_no_regression_vs_baseline | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Weekly replay decision: `promote`
- Split canary decision: `promote`
- Split eval decision: `promote`
- rusticl/clover ratio min: `0.922211`
- split T5 disable total: `0`
- split T5 overhead max: `1.306942`

## Artifacts

- `weekly_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_20260211_015114.json`
- `weekly_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_20260211_015114.md`
- `weekly_canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_canary_20260211_015423.json`
- `weekly_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_eval_20260211_015423.json`
- `split_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_canary_20260211_015619.json`
- `split_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_canary_20260211_015619.md`
- `split_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_eval_20260211_015619.json`
- `split_eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_eval_20260211_015619.md`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_015114.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_015638.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Week19 Block1 weekly replay and split maintenance remain stable on v0.15.0 with canonical gates green.

