# Week 30 Block 3 - Second Monthly Comparative + Platform Decision

- Date: 2026-02-13T17:56:05.054934+00:00
- Stable tag: `v0.15.0`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| baseline_cycle_report_exists | True |
| current_block1_promote | True |
| current_block2_promote | True |
| split_eval_promote | True |
| platform_policy_written | True |
| platform_policy_not_shadow_only | True |
| debt_review_written | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Baseline cycle decision: `promote`
- Current block1 decision: `promote`
- Current block2 decision: `promote`
- Platform policy: `dual_go_clover_rusticl`
- split_ratio_delta_percent: `-0.015232`
- t5_overhead_delta_percent: `-28.808600`
- t5_disable_delta: `0`

## Artifacts

- `dashboard_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week30_block3_biweekly_comparative_dashboard_20260213_175545.json`
- `dashboard_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week30_block3_biweekly_comparative_dashboard_20260213_175545.md`
- `platform_policy_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/WEEK30_BLOCK3_PLATFORM_POLICY_DECISION.json`
- `platform_policy_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/WEEK30_BLOCK3_PLATFORM_POLICY_DECISION.md`
- `debt_review_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/WEEK30_BLOCK3_OPERATIONAL_DEBT_REVIEW.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_175545.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_175605.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Second monthly comparative confirms continuity and emits a production-capable platform policy.

