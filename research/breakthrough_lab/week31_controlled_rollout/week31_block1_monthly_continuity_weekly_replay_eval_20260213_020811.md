# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T02:08:11.616108+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_weekly_replay_canary_20260213_020811.json`

## Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots | False |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_disable_events_total_bound | False |
| t5_overhead_bound | False |
| all_policy_rows_present | True |
| per_kernel_size_slo | True |
| throughput_drift_abs_bound | True |
| p95_drift_bound | True |

## Rows

| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 877.352 | 6.133 | 0.510 | -0.238 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.408 | 22.202 | 0.268 | -0.505 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.868 | 72.095 | -0.031 | 0.007 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 904.584 | 5.989 | -1.394 | 0.421 | 0.0000 | 10.8918 | 1 |
| auto_t5_guarded | 2048 | 777.616 | 21.981 | 0.059 | 0.049 | 0.0000 | 3.3259 | 0 |
| auto_t5_guarded | 3072 | 803.681 | 71.991 | 0.135 | 0.012 | 0.0000 | 0.7836 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['minimum_snapshots', 't5_disable_events_total_bound', 't5_overhead_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

