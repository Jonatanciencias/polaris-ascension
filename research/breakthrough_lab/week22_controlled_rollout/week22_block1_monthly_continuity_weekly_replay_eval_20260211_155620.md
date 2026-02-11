# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T15:56:20.988259+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block1_monthly_continuity_weekly_replay_canary_20260211_155620.json`

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
| auto_t3_controlled | 1400 | 875.703 | 6.152 | -0.745 | -0.402 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 771.815 | 22.208 | 0.005 | -0.269 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.621 | 72.091 | -0.171 | 0.092 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 909.502 | 5.982 | 0.092 | 0.580 | 0.0000 | 1.1642 | 0 |
| auto_t5_guarded | 2048 | 777.484 | 21.967 | 0.064 | -0.064 | 0.0000 | 0.5312 | 0 |
| auto_t5_guarded | 3072 | 804.049 | 71.974 | 0.013 | 0.004 | 0.0000 | 0.3335 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

