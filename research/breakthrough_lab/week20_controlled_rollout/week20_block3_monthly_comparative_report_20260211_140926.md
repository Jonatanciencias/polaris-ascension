# Week 20 Block 3 - Monthly Comparative Report + Debt Review

- Date: 2026-02-11T14:09:46.134625+00:00
- Stable tag: `v0.15.0`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| baseline_block_decision_promote | True |
| current_block1_decision_promote | True |
| current_block2_decision_promote | True |
| current_block2_alerts_promote | True |
| current_split_ratio_floor | True |
| t5_overhead_regression_within_limit | True |
| t5_disable_events_not_regressed | True |
| comparative_dashboard_written | True |
| debt_review_written | True |
| block2_inputs_exist | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Baseline decision: `promote`
- Current block1 decision: `promote`
- Current block2 decision: `promote`
- Block2 alerts decision: `promote`
- split_ratio_min delta %: `0.149981`
- t5_overhead_max delta %: `4.198661`
- t5_disable_total delta: `0`

## Artifacts

- `dashboard_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block3_monthly_comparative_dashboard_20260211_140926.json`
- `dashboard_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block3_monthly_comparative_dashboard_20260211_140926.md`
- `debt_review_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK3_OPERATIONAL_DEBT_REVIEW.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140926.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140946.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Monthly comparative review confirms stable continuation with controlled debt and green gates.

