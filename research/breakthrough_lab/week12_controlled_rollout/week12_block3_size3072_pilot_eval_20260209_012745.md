# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T01:27:45.261524+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json`

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
| auto_t3_controlled | 1400 | 881.899 | 6.148 | -0.029 | 1.287 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.432 | 22.188 | -0.005 | 0.371 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.618 | 72.092 | -0.163 | -0.172 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 903.671 | 6.005 | -2.742 | 0.498 | 0.0000 | 1.8956 | 0 |
| auto_t5_guarded | 2048 | 779.591 | 21.978 | -0.048 | 0.078 | 0.0000 | 0.8164 | 0 |
| auto_t5_guarded | 3072 | 803.701 | 71.981 | 0.029 | -0.060 | 0.0000 | 0.4978 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

