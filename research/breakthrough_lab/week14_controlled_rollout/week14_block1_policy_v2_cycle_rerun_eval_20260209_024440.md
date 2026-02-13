# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T02:44:40.385469+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_canary_20260209_024440.json`

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
| auto_t3_controlled | 1400 | 884.525 | 6.136 | -0.137 | -0.887 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.159 | 22.187 | -0.184 | 0.018 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 699.975 | 72.088 | -1.778 | -0.080 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 901.122 | 6.004 | -1.846 | -0.540 | 0.0000 | 1.3479 | 0 |
| auto_t5_guarded | 2048 | 778.130 | 21.988 | -0.016 | 0.016 | 0.0000 | 1.1311 | 0 |
| auto_t5_guarded | 3072 | 803.820 | 71.965 | -0.130 | 0.022 | 0.0000 | 0.3581 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

