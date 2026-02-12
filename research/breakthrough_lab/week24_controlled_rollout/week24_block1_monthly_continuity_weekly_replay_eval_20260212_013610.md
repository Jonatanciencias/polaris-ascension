# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T01:36:10.769896+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week24_controlled_rollout/week24_block1_monthly_continuity_weekly_replay_canary_20260212_013610.json`

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
| auto_t3_controlled | 1400 | 873.016 | 6.159 | 0.323 | 0.027 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.440 | 22.199 | -0.083 | 0.181 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.999 | 72.072 | -0.340 | -0.030 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 906.918 | 6.002 | -0.329 | -0.416 | 0.0000 | 1.2425 | 0 |
| auto_t5_guarded | 2048 | 777.999 | 21.979 | -0.046 | 0.067 | 0.0000 | 0.5415 | 0 |
| auto_t5_guarded | 3072 | 804.084 | 71.969 | 0.002 | -0.054 | 0.0000 | 0.3400 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

