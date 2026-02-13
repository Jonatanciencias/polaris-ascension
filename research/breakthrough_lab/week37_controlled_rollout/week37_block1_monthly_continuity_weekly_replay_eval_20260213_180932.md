# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T18:09:32.824620+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week37_controlled_rollout/week37_block1_monthly_continuity_weekly_replay_canary_20260213_180932.json`

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
| auto_t3_controlled | 1400 | 873.031 | 6.145 | -0.363 | 0.664 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.907 | 22.180 | 0.013 | -0.081 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 707.884 | 72.099 | 0.091 | 0.014 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.224 | 5.993 | -0.029 | -0.634 | 0.0000 | 1.7665 | 0 |
| auto_t5_guarded | 2048 | 778.833 | 21.982 | -0.021 | -0.013 | 0.0000 | 0.8311 | 0 |
| auto_t5_guarded | 3072 | 803.025 | 72.008 | -0.008 | 0.090 | 0.0000 | 0.5047 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

