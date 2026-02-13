# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T15:22:17.619660+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_weekly_replay_canary_20260212_152217.json`

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
| auto_t3_controlled | 1400 | 875.573 | 6.155 | -1.013 | -0.094 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 771.973 | 22.204 | -0.100 | 0.137 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 707.804 | 72.104 | -0.371 | 0.003 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.692 | 5.983 | -0.052 | -0.015 | 0.0000 | 1.1263 | 0 |
| auto_t5_guarded | 2048 | 778.118 | 21.972 | -0.048 | -0.073 | 0.0000 | 0.5825 | 0 |
| auto_t5_guarded | 3072 | 804.297 | 71.950 | -0.042 | -0.026 | 0.0000 | 0.3349 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

