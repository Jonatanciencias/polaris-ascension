# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T17:27:33.904144+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week36_controlled_rollout/week36_block1_monthly_continuity_recovery2_weekly_replay_canary_20260213_172733.json`

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
| auto_t3_controlled | 1400 | 868.968 | 6.161 | 1.435 | -1.207 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.636 | 22.199 | 0.053 | 0.115 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.823 | 72.037 | -0.485 | -0.005 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.368 | 6.001 | 0.178 | 0.033 | 0.0000 | 2.7156 | 0 |
| auto_t5_guarded | 2048 | 780.303 | 21.975 | -0.013 | 0.041 | 0.0000 | 1.3689 | 0 |
| auto_t5_guarded | 3072 | 804.641 | 71.963 | 0.047 | 0.037 | 0.0000 | 0.8063 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

