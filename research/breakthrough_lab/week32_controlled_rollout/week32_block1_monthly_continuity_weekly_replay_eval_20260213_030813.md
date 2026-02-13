# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T03:08:13.042611+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week32_controlled_rollout/week32_block1_monthly_continuity_weekly_replay_canary_20260213_030812.json`

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
| auto_t3_controlled | 1400 | 875.875 | 6.128 | 0.800 | -1.248 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.648 | 22.180 | 0.125 | -0.178 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.987 | 72.062 | -0.294 | 0.153 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 909.067 | 5.993 | -0.233 | 0.099 | 0.0000 | 1.1509 | 0 |
| auto_t5_guarded | 2048 | 777.968 | 21.979 | -0.009 | -0.035 | 0.0000 | 0.5218 | 0 |
| auto_t5_guarded | 3072 | 804.023 | 71.991 | -0.027 | -0.055 | 0.0000 | 0.3350 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

