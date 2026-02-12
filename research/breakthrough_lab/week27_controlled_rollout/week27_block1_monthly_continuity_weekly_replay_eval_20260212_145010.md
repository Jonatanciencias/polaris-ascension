# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T14:50:10.595201+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week27_controlled_rollout/week27_block1_monthly_continuity_weekly_replay_canary_20260212_145010.json`

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
| auto_t3_controlled | 1400 | 874.682 | 6.143 | -0.592 | 0.994 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.507 | 22.187 | -0.164 | -0.034 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 707.473 | 72.117 | -0.105 | 0.009 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.036 | 5.997 | 0.338 | -0.102 | 0.0000 | 1.1279 | 0 |
| auto_t5_guarded | 2048 | 776.782 | 21.978 | -0.978 | 0.115 | 0.0000 | 0.6102 | 0 |
| auto_t5_guarded | 3072 | 804.086 | 71.957 | 0.056 | -0.033 | 0.0000 | 0.3362 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

