# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T02:37:10.500122+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_weekly_replay_canary_20260213_023710.json`

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
| auto_t3_controlled | 1400 | 875.287 | 6.136 | -0.738 | 0.276 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.856 | 22.184 | 0.020 | -0.122 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.775 | 72.098 | -0.569 | 0.007 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.052 | 5.989 | -0.292 | 0.371 | 0.0000 | 1.4863 | 0 |
| auto_t5_guarded | 2048 | 777.643 | 21.980 | -1.383 | -0.042 | 0.0000 | 4.3320 | 0 |
| auto_t5_guarded | 3072 | 803.796 | 71.980 | 0.033 | -0.014 | 0.0000 | 0.4500 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['t5_overhead_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

