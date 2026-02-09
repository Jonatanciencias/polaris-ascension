# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T13:25:19.182935+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json`

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
| auto_t3_controlled | 1400 | 883.236 | 6.145 | 0.429 | -0.755 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.239 | 22.170 | 0.115 | -0.010 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.045 | 72.049 | -1.376 | -0.040 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.540 | 5.998 | 0.301 | -0.077 | 0.0000 | 1.4280 | 0 |
| auto_t5_guarded | 2048 | 778.059 | 21.988 | 0.368 | -0.296 | 0.0000 | 1.0287 | 0 |
| auto_t5_guarded | 3072 | 803.914 | 71.961 | 0.136 | -0.217 | 0.0000 | 0.5791 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

