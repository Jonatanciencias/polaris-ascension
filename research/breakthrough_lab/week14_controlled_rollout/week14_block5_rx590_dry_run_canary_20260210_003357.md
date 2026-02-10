# Week 9 Block 6 - Wall-Clock Long Canary

- Date: 2026-02-10T00:33:57.466374+00:00
- Wall-clock target/actual (min): 3.0/3.0
- Snapshot interval (min): 1.0
- Snapshots: [1, 2, 3]

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 1 | 1 | 0 |
| rusticl | 1 | 1 | 1 | 0 |
| Clover | 2 | 1 | 1 | 0 |
| rusticl | 2 | 1 | 1 | 0 |
| Clover | 3 | 1 | 1 | 0 |
| rusticl | 3 | 1 | 1 | 0 |

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

