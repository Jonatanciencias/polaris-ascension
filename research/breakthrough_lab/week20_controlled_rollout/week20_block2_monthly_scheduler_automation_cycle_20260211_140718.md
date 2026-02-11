# Week 20 Block 1 - Monthly Full Cycle

- Date: 2026-02-11T14:07:38.165492+00:00
- Stable tag: `v0.15.0`
- Policy path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- T5 policy path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json`
- Baseline path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_canary_20260211_015619.json`

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
| consolidation_artifacts_written | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Weekly replay decision: `promote`
- Split canary decision: `promote`
- Split eval decision: `promote`
- split_ratio_min: `0.923278`
- split_t5_overhead_max: `1.326084`
- split_t5_disable_total: `0`

## Artifacts

- `weekly_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_20260211_140217.json`
- `weekly_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_20260211_140217.md`
- `weekly_canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_canary_20260211_140523.json`
- `weekly_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_eval_20260211_140523.json`
- `split_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_split_canary_20260211_140718.json`
- `split_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_split_canary_20260211_140718.md`
- `split_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_split_eval_20260211_140718.json`
- `split_eval_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_split_eval_20260211_140718.md`
- `dashboard_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260211_140718.json`
- `dashboard_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block1_monthly_cycle_dashboard_20260211_140718.md`
- `runbook_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK1_MONTHLY_CYCLE_RUNBOOK.md`
- `checklist_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK1_MONTHLY_CYCLE_CHECKLIST.md`
- `debt_matrix_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK1_MONTHLY_CYCLE_LIVE_DEBT_MATRIX.json`
- `manifest_path`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK20_BLOCK1_MONTHLY_CYCLE_MANIFEST.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140217.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_140738.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Week20 Block1 monthly full cycle completed with replay/split/consolidation stable and canonical gates green.

