# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T02:43:42.598616+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week31_controlled_rollout/week31_block1_monthly_continuity_recovery_r2_weekly_replay_canary_20260213_024342.json`

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
| auto_t3_controlled | 1400 | 871.750 | 6.134 | 0.078 | -0.040 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.019 | 22.194 | 0.199 | -0.392 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 711.152 | 72.090 | 0.002 | -0.008 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.665 | 6.009 | -0.419 | 0.949 | 0.0000 | 2.2758 | 0 |
| auto_t5_guarded | 2048 | 779.196 | 21.977 | -0.031 | 0.040 | 0.0000 | 1.0617 | 0 |
| auto_t5_guarded | 3072 | 804.049 | 71.992 | -0.315 | 0.028 | 0.0000 | 0.7235 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

