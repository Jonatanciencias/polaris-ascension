# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T17:20:33.535487+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery_weekly_replay_canary_20260213_172033.json`

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
| throughput_drift_abs_bound | False |
| p95_drift_bound | True |

## Rows

| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 869.379 | 6.159 | 0.196 | 0.599 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.860 | 22.200 | 0.060 | -0.149 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 714.286 | 72.070 | -0.300 | -0.011 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 909.025 | 6.007 | -0.475 | 0.760 | 0.0000 | 2.5100 | 0 |
| auto_t5_guarded | 2048 | 777.175 | 21.984 | 2.813 | 0.001 | 0.0000 | 1.0913 | 0 |
| auto_t5_guarded | 3072 | 804.156 | 71.994 | 0.335 | 0.000 | 0.0000 | 0.6858 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['throughput_drift_abs_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

