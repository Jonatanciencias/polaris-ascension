# Week 12 Block 2 - Platform Split Policy Evaluation

- Date: 2026-02-09T13:27:55.533866+00:00
- Split artifact: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_platform_split_20260209_132747.json`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`

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

