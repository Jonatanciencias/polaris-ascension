# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T01:45:22.278984+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json`

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
| auto_t3_controlled | 1400 | 880.860 | 6.140 | 0.459 | -0.754 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.110 | 22.196 | 0.092 | -0.200 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 704.725 | 72.111 | -1.648 | 0.069 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.427 | 6.000 | 0.658 | 0.061 | 0.0000 | 1.4782 | 0 |
| auto_t5_guarded | 2048 | 777.284 | 21.994 | -0.928 | 0.054 | 0.0000 | 1.0476 | 0 |
| auto_t5_guarded | 3072 | 803.754 | 71.974 | 0.012 | -0.035 | 0.0000 | 0.6233 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

