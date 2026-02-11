# Week 19 Block 2 - Biweekly Drift Recalibration

- Date: 2026-02-11T02:04:43.031755+00:00
- Stable tag: `v0.15.0`
- Base policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Recalibrated policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`

## Checks

| Check | Pass |
| --- | --- |
| stable_manifest_exists | True |
| stable_tag_v0_15_0 | True |
| recalibration_decision_promote | True |
| recalibrated_policy_exists | True |
| recalibrated_policy_eval_promote | True |
| global_abs_drift_conservative | True |
| global_p95_drift_conservative | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Recalibration decision: `promote`
- Recalibration action: `applied`
- Recalibrated policy eval decision: `promote`
- Observed max abs throughput drift: `0.690593`
- Observed max p95 drift: `0.324145`

## Artifacts

- `recalibration_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block2_biweekly_drift_recalibration_20260211_020423.json`
- `recalibration_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block2_biweekly_drift_recalibration_20260211_020423.md`
- `recalibration_policy_raw`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- `recalibrated_policy_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- `recalibrated_policy_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block2_recalibrated_policy_eval_20260211_020423.json`
- `recalibrated_policy_eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block2_recalibrated_policy_eval_20260211_020423.md`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_020423.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_020443.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Biweekly drift review confirms stable behavior and conservative recalibration remains policy-safe.

