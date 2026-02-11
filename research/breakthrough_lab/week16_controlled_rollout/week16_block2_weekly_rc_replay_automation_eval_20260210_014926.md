# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-10T01:49:26.613584+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_canary_20260210_014926.json`

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
| auto_t3_controlled | 1400 | 876.946 | 6.140 | -0.812 | 2.171 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.439 | 22.193 | -0.028 | -0.131 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 705.553 | 72.048 | -0.199 | 0.049 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 906.112 | 6.004 | -0.170 | 0.332 | 0.0000 | 1.2423 | 0 |
| auto_t5_guarded | 2048 | 774.723 | 21.991 | 1.857 | -0.261 | 0.0000 | 0.5700 | 0 |
| auto_t5_guarded | 3072 | 803.547 | 71.967 | 0.310 | 0.110 | 0.0000 | 0.3495 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

