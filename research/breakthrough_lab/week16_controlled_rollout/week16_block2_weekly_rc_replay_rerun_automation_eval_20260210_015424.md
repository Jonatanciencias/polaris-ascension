# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-10T01:54:24.835046+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_canary_20260210_015424.json`

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
| auto_t3_controlled | 1400 | 876.872 | 6.142 | -0.026 | 0.718 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.514 | 22.187 | -0.124 | 0.063 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 704.947 | 72.070 | -0.267 | 0.016 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 898.516 | 6.008 | -0.094 | 0.222 | 0.0000 | 1.1595 | 0 |
| auto_t5_guarded | 2048 | 777.864 | 21.974 | -0.052 | 0.067 | 0.0000 | 0.5823 | 0 |
| auto_t5_guarded | 3072 | 803.123 | 71.998 | 0.492 | -0.084 | 0.0000 | 0.3466 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

