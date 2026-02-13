# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T17:12:28.034192+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_weekly_replay_canary_20260213_171227.json`

## Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots | True |
| correctness_error_bound | True |
| t3_fallback_bound | True |
| t5_disable_events_total_bound | False |
| t5_overhead_bound | False |
| all_policy_rows_present | True |
| per_kernel_size_slo | True |
| throughput_drift_abs_bound | False |
| p95_drift_bound | True |

## Rows

| Kernel | Size | Avg GFLOPS | P95 ms | Thr drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 872.234 | 6.135 | -0.007 | -0.069 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.316 | 22.206 | -0.081 | 0.259 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 711.280 | 72.081 | 0.036 | -0.111 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 890.872 | 6.016 | -11.766 | 1.171 | 0.0000 | 37.0156 | 1 |
| auto_t5_guarded | 2048 | 779.922 | 21.985 | 0.045 | 0.045 | 0.0000 | 1.1670 | 0 |
| auto_t5_guarded | 3072 | 803.851 | 71.994 | 0.005 | 0.073 | 0.0000 | 0.6782 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['t5_disable_events_total_bound', 't5_overhead_bound', 'throughput_drift_abs_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

