# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T00:36:19.626738+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block1_monthly_continuity_weekly_replay_canary_20260212_003619.json`

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
| auto_t3_controlled | 1400 | 874.395 | 6.140 | -0.594 | -0.345 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.095 | 22.213 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.324 | 72.032 | 0.218 | -0.039 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.535 | 5.996 | 0.551 | -0.182 | 0.0000 | 1.1109 | 0 |
| auto_t5_guarded | 2048 | 776.294 | 21.981 | 0.120 | 0.017 | 0.0000 | 0.5341 | 0 |
| auto_t5_guarded | 3072 | 804.385 | 71.952 | 0.022 | 0.009 | 0.0000 | 0.3439 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

