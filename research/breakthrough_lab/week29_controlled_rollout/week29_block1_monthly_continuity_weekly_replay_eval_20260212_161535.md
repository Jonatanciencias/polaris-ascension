# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-12T16:15:35.792655+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week29_controlled_rollout/week29_block1_monthly_continuity_weekly_replay_canary_20260212_161535.json`

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
| auto_t3_controlled | 1400 | 875.450 | 6.163 | 0.318 | 0.335 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.646 | 22.186 | -0.105 | 0.038 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 707.787 | 72.105 | -0.157 | -0.024 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.336 | 5.994 | 0.404 | -0.225 | 0.0000 | 1.1126 | 0 |
| auto_t5_guarded | 2048 | 778.220 | 21.976 | -0.139 | 0.099 | 0.0000 | 0.5299 | 0 |
| auto_t5_guarded | 3072 | 803.858 | 71.947 | 0.220 | -0.016 | 0.0000 | 0.3374 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

