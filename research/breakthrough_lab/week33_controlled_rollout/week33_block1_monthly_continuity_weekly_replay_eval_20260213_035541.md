# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T03:55:41.570862+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week33_controlled_rollout/week33_block1_monthly_continuity_weekly_replay_canary_20260213_035541.json`

## Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots | True |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_disable_events_total_bound | True |
| t5_overhead_bound | False |
| all_policy_rows_present | True |
| per_kernel_size_slo | True |
| throughput_drift_abs_bound | True |
| p95_drift_bound | True |

## Rows

| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 874.744 | 6.165 | 1.224 | -1.079 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.025 | 22.213 | 0.037 | -0.066 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.221 | 72.086 | -0.351 | 0.113 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.187 | 5.994 | 0.456 | 0.210 | 0.0000 | 3.2060 | 0 |
| auto_t5_guarded | 2048 | 777.933 | 21.977 | 0.092 | -0.101 | 0.0000 | 0.5348 | 0 |
| auto_t5_guarded | 3072 | 803.876 | 71.985 | -0.006 | 0.071 | 0.0000 | 0.3334 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['t5_overhead_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

