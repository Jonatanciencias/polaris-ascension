# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T02:04:23.480226+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_canary_20260211_015423.json`

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
| auto_t3_controlled | 1400 | 873.568 | 6.153 | -0.691 | 0.216 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 772.243 | 22.176 | 0.001 | 0.087 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 709.304 | 72.074 | -0.395 | -0.097 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 910.539 | 5.978 | -0.174 | 0.164 | 0.0000 | 1.2207 | 0 |
| auto_t5_guarded | 2048 | 775.807 | 21.983 | 0.010 | -0.037 | 0.0000 | 0.5217 | 0 |
| auto_t5_guarded | 3072 | 804.111 | 71.962 | -0.012 | 0.002 | 0.0000 | 0.3360 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

