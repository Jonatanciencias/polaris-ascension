# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T04:26:03.969952+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week34_controlled_rollout/week34_block1_monthly_continuity_rc_canary_weekly_replay_canary_20260213_042603.json`

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
| auto_t3_controlled | 1400 | 867.364 | 6.160 | 0.063 | 0.507 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.650 | 22.200 | -0.071 | 0.079 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 710.577 | 72.092 | -0.172 | -0.299 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.783 | 5.998 | -0.686 | -0.265 | 0.0000 | 2.3030 | 0 |
| auto_t5_guarded | 2048 | 780.061 | 21.979 | 0.016 | -0.028 | 0.0000 | 1.0604 | 0 |
| auto_t5_guarded | 3072 | 804.089 | 71.988 | 0.021 | -0.025 | 0.0000 | 0.8091 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

