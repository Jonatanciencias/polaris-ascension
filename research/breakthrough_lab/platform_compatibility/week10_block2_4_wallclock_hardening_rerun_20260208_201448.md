# Week 9 Block 6 - Wall-Clock Long Canary

- Date: 2026-02-08T20:14:48.918253+00:00
- Wall-clock target/actual (min): 40.0/40.0
- Snapshot interval (min): 5.0
- Snapshots: [1, 2, 3, 4, 5, 6, 7, 8]

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 3 | 3 | 0 |
| rusticl | 1 | 3 | 3 | 0 |
| Clover | 2 | 3 | 3 | 0 |
| rusticl | 2 | 3 | 3 | 0 |
| Clover | 3 | 3 | 3 | 0 |
| rusticl | 3 | 3 | 3 | 0 |
| Clover | 4 | 3 | 3 | 0 |
| rusticl | 4 | 3 | 3 | 0 |
| Clover | 5 | 3 | 3 | 0 |
| rusticl | 5 | 3 | 3 | 0 |
| Clover | 6 | 3 | 3 | 0 |
| rusticl | 6 | 3 | 3 | 0 |
| Clover | 7 | 3 | 3 | 0 |
| rusticl | 7 | 3 | 3 | 0 |
| Clover | 8 | 3 | 3 | 0 |
| rusticl | 8 | 3 | 3 | 0 |

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

