# Week 18 Block 2 - Stable Maintenance Split Canary

- Date: 2026-02-11T01:42:55.498689+00:00
- Stable tag: `v0.15.0`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- T5 policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json`
- Baseline: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.json`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| canary_promote | True |
| canary_split_present | True |
| canary_t5_guardrails | True |
| canary_no_regression_vs_baseline | True |
| split_eval_promote | True |
| split_required_sizes_present | True |
| split_ratio_meets_floor | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Split Metrics

- rusticl_ratio_min: 0.918116
- canary_t5_overhead_max: 1.402266
- canary_t5_disable_total: 0

## Artifacts

- `canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_canary_20260211_014235.json`
- `canary_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_canary_20260211_014235.md`
- `split_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_eval_20260211_014235.json`
- `split_eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week18_controlled_rollout/week18_block2_maintenance_split_eval_20260211_014235.md`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_014059.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_014255.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Stable split maintenance canary passed policy, guardrails, and mandatory canonical gates.

