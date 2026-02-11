# Week 9 Block 6 - Wall-Clock Long Canary

- Date: 2026-02-11T00:38:31.942021+00:00
- Wall-clock target/actual (min): 2.0/2.0
- Snapshot interval (min): 1.0
- Snapshots: [1, 2]

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 2 | 2 | 0 |
| rusticl | 1 | 2 | 2 | 0 |
| Clover | 2 | 2 | 2 | 0 |
| rusticl | 2 | 2 | 2 | 0 |

## Checks

| Check | Pass |
| --- | --- |
| wallclock_duration_target | True |
| all_runs_success | True |
| pressure_failures_zero | True |
| platform_split_clover_and_rusticl | True |
| correctness_bound_all_runs | True |
| t3_guardrails_all_runs | True |
| t5_guardrails_all_runs | True |
| rusticl_peak_ratio_min | True |
| drift_abs_percent_bounded | True |
| no_regression_vs_block5_clover | True |

## Decision

- Decision: `promote`
- Rationale: Wall-clock canary passed with stable guardrails and platform split behavior.

