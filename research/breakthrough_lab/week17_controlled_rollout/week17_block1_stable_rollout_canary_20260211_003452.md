# Week 9 Block 6 - Wall-Clock Long Canary

- Date: 2026-02-11T00:34:52.644317+00:00
- Wall-clock target/actual (min): 10.0/10.0
- Snapshot interval (min): 1.0
- Snapshots: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 2 | 2 | 0 |
| rusticl | 1 | 2 | 2 | 0 |
| Clover | 2 | 2 | 2 | 0 |
| rusticl | 2 | 2 | 2 | 0 |
| Clover | 3 | 2 | 2 | 0 |
| rusticl | 3 | 2 | 2 | 0 |
| Clover | 4 | 2 | 2 | 0 |
| rusticl | 4 | 2 | 2 | 0 |
| Clover | 5 | 2 | 2 | 0 |
| rusticl | 5 | 2 | 2 | 0 |
| Clover | 6 | 2 | 2 | 0 |
| rusticl | 6 | 2 | 2 | 0 |
| Clover | 7 | 2 | 2 | 0 |
| rusticl | 7 | 2 | 2 | 0 |
| Clover | 8 | 2 | 2 | 0 |
| rusticl | 8 | 2 | 2 | 0 |
| Clover | 9 | 2 | 2 | 0 |
| rusticl | 9 | 2 | 2 | 0 |
| Clover | 10 | 2 | 2 | 0 |
| rusticl | 10 | 2 | 2 | 0 |

## Checks

| Check | Pass |
| --- | --- |
| wallclock_duration_target | True |
| all_runs_success | True |
| pressure_failures_zero | True |
| platform_split_clover_and_rusticl | True |
| correctness_bound_all_runs | True |
| t3_guardrails_all_runs | True |
| t5_guardrails_all_runs | False |
| rusticl_peak_ratio_min | True |
| drift_abs_percent_bounded | True |
| no_regression_vs_block5_clover | True |

## Decision

- Decision: `iterate`
- Rationale: Wall-clock canary found one or more guardrail/regression failures.

