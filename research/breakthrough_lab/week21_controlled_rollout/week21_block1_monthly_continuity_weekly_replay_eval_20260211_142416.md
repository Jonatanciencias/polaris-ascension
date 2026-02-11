# Week 11 Block 4 - Weekly Replay Evaluation

- Date: 2026-02-11T14:24:16.367982+00:00
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`
- Canary: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week21_controlled_rollout/week21_block1_monthly_continuity_weekly_replay_canary_20260211_142416.json`

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
| auto_t3_controlled | 1400 | 873.696 | 6.132 | 0.779 | -2.302 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 2048 | 771.873 | 22.196 | 0.037 | -0.219 | 0.0000 | 0.0000 | 0 |
| auto_t3_controlled | 3072 | 706.165 | 72.123 | -0.145 | -0.038 | 0.0000 | 0.0000 | 0 |
| auto_t5_guarded | 1400 | 893.284 | 5.982 | 0.575 | -0.221 | 0.0000 | 1.3060 | 0 |
| auto_t5_guarded | 2048 | 777.875 | 21.983 | -0.084 | 0.243 | 0.0000 | 0.5330 | 0 |
| auto_t5_guarded | 3072 | 803.752 | 71.954 | -0.053 | -0.080 | 0.0000 | 0.3688 | 0 |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay satisfies all global guardrails, per-kernel SLO rows, and drift bounds.

