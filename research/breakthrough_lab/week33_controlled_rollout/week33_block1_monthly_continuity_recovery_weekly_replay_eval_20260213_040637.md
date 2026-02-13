# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T04:06:37.133888+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week33_controlled_rollout/week33_block1_monthly_continuity_recovery_weekly_replay_canary_20260213_040637.json`

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
| auto_t3_controlled | 1400 | 868.291 | 6.170 | 0.038 | 1.165 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.625 | 22.205 | 0.152 | -0.393 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 710.175 | 72.071 | -0.052 | 0.109 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.958 | 6.011 | -0.164 | -0.120 | 0.0000 | 2.2986 | 0 |
| auto_t5_guarded | 2048 | 780.060 | 21.979 | -0.012 | 0.052 | 0.0000 | 1.0688 | 0 |
| auto_t5_guarded | 3072 | 803.856 | 71.995 | 0.134 | 0.018 | 0.0000 | 0.6839 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

