# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T02:00:50.026615+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json`

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
| auto_t3_controlled | 1400 | 881.746 | 6.140 | 0.729 | -0.577 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.246 | 22.178 | -0.087 | 0.013 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.514 | 72.072 | -0.500 | -0.013 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 901.821 | 6.003 | 0.238 | -0.072 | 0.0000 | 1.3788 | 0 |
| auto_t5_guarded | 2048 | 777.996 | 21.987 | 0.278 | -0.049 | 0.0000 | 0.6160 | 0 |
| auto_t5_guarded | 3072 | 803.699 | 71.974 | 0.160 | -0.002 | 0.0000 | 0.6139 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

