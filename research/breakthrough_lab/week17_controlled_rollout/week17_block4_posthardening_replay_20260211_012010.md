# Week 17 Block 4 - Post-Hardening Weekly Replay

- Date: 2026-02-11T01:20:10.561181+00:00
- Stable tag: `v0.15.0`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- T5 policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| replay_automation_promote | True |
| replay_eval_promote | True |
| throughput_drift_bound | True |
| p95_drift_bound | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Drift

- max_abs_throughput_drift_percent: 0.6065
- max_positive_p95_drift_percent: 0.3241

## Artifacts

- `replay_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_20260211_011608.json`
- `replay_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_20260211_011608.md`
- `replay_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_eval_20260211_011930.json`
- `replay_canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_canary_20260211_011930.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_011608.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_012010.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Post-hardening weekly replay is stable with bounded drift and green pytest-tier gates.

