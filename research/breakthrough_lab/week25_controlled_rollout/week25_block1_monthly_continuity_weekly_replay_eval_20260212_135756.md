# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T13:57:56.442766+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week25_controlled_rollout/week25_block1_monthly_continuity_weekly_replay_canary_20260212_135756.json`

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
| auto_t3_controlled | 1400 | 876.303 | 6.146 | -0.523 | 0.143 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.897 | 22.174 | -0.048 | -0.130 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.493 | 72.094 | -0.077 | -0.063 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.388 | 5.991 | -0.457 | -0.368 | 0.0000 | 1.1897 | 0 |
| auto_t5_guarded | 2048 | 777.919 | 21.989 | 0.040 | -0.044 | 0.0000 | 0.5242 | 0 |
| auto_t5_guarded | 3072 | 804.111 | 71.956 | -0.014 | 0.064 | 0.0000 | 0.3576 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

