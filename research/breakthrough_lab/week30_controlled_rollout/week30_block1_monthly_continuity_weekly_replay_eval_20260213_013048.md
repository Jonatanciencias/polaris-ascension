# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-13T01:30:48.406808+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week30_controlled_rollout/week30_block1_monthly_continuity_weekly_replay_canary_20260213_013048.json`

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
| auto_t3_controlled | 1400 | 876.835 | 6.123 | -0.480 | 0.226 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.475 | 22.180 | 0.313 | -0.450 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.725 | 72.138 | -0.113 | 0.118 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 907.522 | 5.997 | 0.102 | 0.570 | 0.0000 | 1.1639 | 0 |
| auto_t5_guarded | 2048 | 777.019 | 21.977 | 0.011 | -0.011 | 0.0000 | 0.5313 | 0 |
| auto_t5_guarded | 3072 | 803.792 | 72.002 | 0.011 | 0.037 | 0.0000 | 0.3344 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

