# Week 12 Block 2 - Platform Split Policy Evaluation

- Date: 2026-02-13T17:29:06.496448+00:00
- Split artifact: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery2_split_canary_20260213_172906.json`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`

## Checks

| Check | Pass |
| --- | --- |
| all_runs_success | True |
| platform_split_present | True |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_overhead_bound | False |
| t5_disable_events_total_bound | True |
| clover_policy_rows | True |
| rusticl_ratio_vs_clover | True |
| required_sizes_present_on_split | True |

## Decision

- Decision: `iterate`
- Failed checks: ['t5_overhead_bound']
- Rationale: Platform split violated one or more policy guardrails/ratio checks.

