# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T14:05:23.400391+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block2_monthly_scheduler_automation_cycle_weekly_replay_canary_20260211_140523.json`

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
| auto_t3_controlled | 1400 | 873.845 | 6.122 | 0.440 | -1.019 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 771.725 | 22.218 | 0.123 | -0.027 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 702.238 | 72.122 | -0.386 | 0.004 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.644 | 5.993 | -0.334 | 0.544 | 0.0000 | 1.1294 | 0 |
| auto_t5_guarded | 2048 | 777.908 | 21.980 | 0.017 | -0.018 | 0.0000 | 0.5368 | 0 |
| auto_t5_guarded | 3072 | 804.050 | 71.954 | 0.182 | -0.068 | 0.0000 | 0.3345 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

