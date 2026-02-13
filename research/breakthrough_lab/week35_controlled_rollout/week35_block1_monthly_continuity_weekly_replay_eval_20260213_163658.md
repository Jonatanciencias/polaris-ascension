# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T16:36:58.899906+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week35_controlled_rollout/week35_block1_monthly_continuity_weekly_replay_canary_20260213_163658.json`

## Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots | True |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_disable_events_total_bound | True |
| t5_overhead_bound | True |
| all_policy_rows_present | True |
| per_kernel_size_slo | True |
| throughput_drift_abs_bound | True |
| p95_drift_bound | True |

## Rows

| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 868.991 | 6.164 | -0.286 | 0.202 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.282 | 22.174 | 0.279 | -0.327 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 710.938 | 72.083 | -0.334 | 0.022 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.771 | 6.006 | 0.335 | -1.014 | 0.0000 | 2.4137 | 0 |
| auto_t5_guarded | 2048 | 779.802 | 21.991 | -0.049 | 0.182 | 0.0000 | 1.1207 | 0 |
| auto_t5_guarded | 3072 | 804.547 | 71.994 | -0.028 | 0.077 | 0.0000 | 0.6829 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

