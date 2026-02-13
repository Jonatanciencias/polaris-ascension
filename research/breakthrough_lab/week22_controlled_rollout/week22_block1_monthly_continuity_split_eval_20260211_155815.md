# Week 12 Block 2 - Platform Split Policy Evaluation

- Date: 2026-02-11T15:58:15.989049+00:00
- Split artifact: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block1_monthly_continuity_split_canary_20260211_155815.json`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`

## Checks

| Check | Pass |
| --- | --- |
| all_runs_success | True |
| platform_split_present | True |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_overhead_bound | True |
| t5_disable_events_total_bound | True |
| clover_policy_rows | True |
| rusticl_ratio_vs_clover | True |
| required_sizes_present_on_split | True |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Platform split satisfies formal policy guardrails and rusticl/clover ratio requirements.

