# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T17:44:05.187577+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_1_monthly_continuity_targeted_weekly_replay_canary_20260213_174405.json`

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
| auto_t3_controlled | 1400 | 875.783 | 6.109 | -0.257 | 1.901 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.483 | 22.196 | -0.096 | 0.114 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.856 | 72.111 | 0.232 | 0.112 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 906.990 | 6.005 | 0.266 | -0.313 | 0.0000 | 1.9394 | 0 |
| auto_t5_guarded | 2048 | 778.862 | 21.980 | 0.024 | -0.076 | 0.0000 | 0.8484 | 0 |
| auto_t5_guarded | 3072 | 804.071 | 72.019 | 0.063 | -0.088 | 0.0000 | 0.5132 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

