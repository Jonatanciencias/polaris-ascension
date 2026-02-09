# Week 13 Block 3 - Drift Review and Recalibration

- Date: 2026-02-09T02:12:32.947779+00:00
- Base policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Recalibrated policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`

## Sustained Evidence Checks

| Check | Pass |
| --- | --- |
| all_windows_promote | True |
| global_abs_drift_stable | True |
| global_p95_drift_stable | True |
| policy_rows_sustained_stability | True |

## Per-Key Summary

| Key | Samples | Avg GFLOPS min | Avg GFLOPS cv | P95 max ms | P95 cv | Max abs thr drift % | Max p95 drift % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled:1400 | 3 | 880.860 | 0.0011 | 6.150 | 0.0007 | 0.729 | 0.146 |
| auto_t3_controlled:2048 | 3 | 773.046 | 0.0001 | 22.196 | 0.0003 | 0.236 | 0.013 |
| auto_t3_controlled:3072 | 2 | 704.725 | 0.0013 | 72.111 | 0.0003 | 1.648 | 0.069 |
| auto_t5_guarded:1400 | 3 | 901.821 | 0.0026 | 6.003 | 0.0003 | 0.658 | 0.591 |
| auto_t5_guarded:2048 | 3 | 777.040 | 0.0005 | 21.995 | 0.0002 | 1.015 | 0.054 |
| auto_t5_guarded:3072 | 2 | 803.699 | 0.0000 | 71.974 | 0.0000 | 0.160 | -0.002 |

## Decision

- Decision: `promote`
- Recalibration action: `applied`
- Failed checks: []
- Rationale: Sustained evidence conditions passed across windows; conservative tightening was applied.

