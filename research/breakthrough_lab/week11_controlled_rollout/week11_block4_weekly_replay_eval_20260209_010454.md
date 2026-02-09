# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-09T01:04:54.419544+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/week11_block4_weekly_replay_canary_20260209_010447.json`

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
| auto_t3_controlled | 1400 | 883.185 | 6.150 | -0.285 | 0.146 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 773.046 | 22.188 | 0.236 | -0.440 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 905.881 | 5.999 | -0.358 | 0.591 | 0.0000 | 1.3526 | 0 |
| auto_t5_guarded | 2048 | 777.040 | 21.995 | 1.015 | -0.323 | 0.0000 | 1.8522 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

