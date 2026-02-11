# Week 13 Block 3 - Drift Review and Recalibration

- Date: 2026-02-11T02:04:23.420447+00:00
- Base policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Recalibrated policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json`

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
| auto_t3_controlled:1400 | 2 | 873.568 | 0.0013 | 6.153 | 0.0002 | 0.691 | 0.216 |
| auto_t3_controlled:2048 | 2 | 772.243 | 0.0002 | 22.180 | 0.0001 | 0.015 | 0.087 |
| auto_t3_controlled:3072 | 2 | 708.917 | 0.0003 | 72.076 | 0.0000 | 0.395 | -0.012 |
| auto_t5_guarded:1400 | 2 | 908.456 | 0.0011 | 5.995 | 0.0015 | 0.277 | 0.324 |
| auto_t5_guarded:2048 | 2 | 775.807 | 0.0014 | 21.983 | 0.0001 | 0.035 | -0.036 |
| auto_t5_guarded:3072 | 2 | 802.676 | 0.0009 | 71.996 | 0.0002 | 0.607 | 0.235 |

## Decision

- Decision: `promote`
- Recalibration action: `applied`
- Failed checks: []
- Rationale: Sustained evidence conditions passed across windows; conservative tightening was applied.

