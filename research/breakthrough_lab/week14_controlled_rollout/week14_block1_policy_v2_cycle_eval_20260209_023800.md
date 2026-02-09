# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T02:38:00.954667+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_canary_20260209_023800.json`

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
| auto_t3_controlled | 1400 | 885.047 | 6.123 | -0.480 | 0.326 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.114 | 22.159 | 0.068 | -0.159 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.239 | 72.103 | -0.279 | -0.023 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 894.024 | 6.005 | -3.705 | 0.149 | 0.0000 | 6.1523 | 1 |
| auto_t5_guarded | 2048 | 778.020 | 21.985 | 0.186 | -0.048 | 0.0000 | 0.6026 | 0 |
| auto_t5_guarded | 3072 | 804.013 | 71.971 | 0.098 | -0.058 | 0.0000 | 0.3718 | 0 |

## Decision

- Decision: `iterate`
- Failed checks: ['t5_disable_events_total_bound', 't5_overhead_bound', 'throughput_drift_abs_bound']
- Rationale: Weekly replay violated one or more formal SLO checks; keep iterate.

