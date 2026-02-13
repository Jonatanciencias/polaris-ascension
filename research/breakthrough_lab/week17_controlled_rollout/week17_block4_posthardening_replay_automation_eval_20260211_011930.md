# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T01:19:30.917190+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_canary_20260211_011930.json`

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
| auto_t3_controlled | 1400 | 875.923 | 6.150 | 0.372 | -1.049 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.503 | 22.180 | -0.015 | 0.050 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 708.917 | 72.076 | -0.157 | -0.012 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 908.456 | 5.995 | -0.277 | 0.324 | 0.0000 | 1.1928 | 0 |
| auto_t5_guarded | 2048 | 778.012 | 21.979 | 0.035 | -0.036 | 0.0000 | 0.5371 | 0 |
| auto_t5_guarded | 3072 | 802.676 | 71.996 | -0.607 | 0.235 | 0.0000 | 0.9113 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

