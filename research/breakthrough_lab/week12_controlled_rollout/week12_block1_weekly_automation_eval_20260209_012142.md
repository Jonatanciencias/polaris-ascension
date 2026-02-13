# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T01:21:42.159671+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week12_controlled_rollout/week12_block1_weekly_automation_canary_20260209_012142.json`

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
| auto_t3_controlled | 1400 | 882.124 | 6.146 | 0.345 | 0.432 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.046 | 22.187 | 0.004 | 0.064 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 903.778 | 6.003 | 0.415 | -0.457 | 0.0000 | 1.3319 | 0 |
| auto_t5_guarded | 2048 | 778.306 | 21.987 | 0.051 | 0.059 | 0.0000 | 0.6064 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

