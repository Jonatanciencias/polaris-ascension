# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T02:28:57.629556+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week20_controlled_rollout/week20_block1_monthly_cycle_weekly_replay_canary_20260211_022857.json`

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
| auto_t3_controlled | 1400 | 878.254 | 6.123 | -0.352 | 0.646 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.222 | 22.192 | -0.046 | 0.001 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 710.380 | 72.063 | -0.360 | -0.117 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.414 | 5.990 | 0.251 | -0.180 | 0.0000 | 1.1947 | 0 |
| auto_t5_guarded | 2048 | 777.954 | 21.988 | -0.027 | 0.045 | 0.0000 | 0.5657 | 0 |
| auto_t5_guarded | 3072 | 803.635 | 71.941 | -0.023 | 0.014 | 0.0000 | 0.3722 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

