# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T14:32:24.391921+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week26_controlled_rollout/week26_block1_monthly_continuity_weekly_replay_canary_20260212_143224.json`

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
| auto_t3_controlled | 1400 | 875.564 | 6.137 | -0.633 | -0.376 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.504 | 22.176 | 0.088 | -0.213 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.770 | 72.083 | -0.155 | -0.314 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.854 | 5.983 | -0.104 | 0.102 | 0.0000 | 1.1201 | 0 |
| auto_t5_guarded | 2048 | 778.022 | 21.979 | -0.079 | -0.095 | 0.0000 | 0.5293 | 0 |
| auto_t5_guarded | 3072 | 804.310 | 71.956 | -0.000 | 0.041 | 0.0000 | 0.3320 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

