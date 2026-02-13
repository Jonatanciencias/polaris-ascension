# Week 10 Block 1.5 - Controlled Pre-Production Pilot (RX590)

- Date: 2026-02-08T16:56:31.279417+00:00
- Hourly snapshots (logical): [1, 2, 3, 4]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 2 | 2 | 0 |
| Clover | 2 | 2 | 2 | 0 |
| Clover | 3 | 2 | 2 | 0 |
| Clover | 4 | 2 | 2 | 0 |
| rusticl | 1 | 2 | 2 | 0 |
| rusticl | 2 | 2 | 2 | 0 |
| rusticl | 3 | 2 | 2 | 0 |
| rusticl | 4 | 2 | 2 | 0 |

## Pilot Matrix

| Platform | Snapshot | Kernel | Size | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clover | 1 | auto_t3_controlled | 1400 | 883.196 | 6.078 | 0.0003204 | 0.000 | 0 |
| Clover | 1 | auto_t3_controlled | 2048 | 772.402 | 22.144 | 0.0005493 | 0.000 | 0 |
| Clover | 1 | auto_t5_guarded | 1400 | 909.325 | 5.990 | 0.0002747 | 1.490 | 0 |
| Clover | 1 | auto_t5_guarded | 2048 | 777.889 | 21.975 | 0.0005188 | 0.643 | 0 |
| Clover | 2 | auto_t3_controlled | 1400 | 875.623 | 6.107 | 0.0002747 | 0.000 | 0 |
| Clover | 2 | auto_t3_controlled | 2048 | 773.814 | 22.136 | 0.0005035 | 0.000 | 0 |
| Clover | 2 | auto_t5_guarded | 1400 | 907.378 | 5.994 | 0.0003204 | 1.490 | 0 |
| Clover | 2 | auto_t5_guarded | 2048 | 778.150 | 21.991 | 0.0007019 | 0.664 | 0 |
| Clover | 3 | auto_t3_controlled | 1400 | 870.106 | 6.179 | 0.0002899 | 0.000 | 0 |
| Clover | 3 | auto_t3_controlled | 2048 | 773.841 | 22.137 | 0.0004730 | 0.000 | 0 |
| Clover | 3 | auto_t5_guarded | 1400 | 909.157 | 6.000 | 0.0003357 | 1.424 | 0 |
| Clover | 3 | auto_t5_guarded | 2048 | 777.154 | 22.017 | 0.0005493 | 0.662 | 0 |
| Clover | 4 | auto_t3_controlled | 1400 | 871.209 | 6.186 | 0.0004120 | 0.000 | 0 |
| Clover | 4 | auto_t3_controlled | 2048 | 771.948 | 22.185 | 0.0004425 | 0.000 | 0 |
| Clover | 4 | auto_t5_guarded | 1400 | 894.560 | 6.029 | 0.0003815 | 1.388 | 0 |
| Clover | 4 | auto_t5_guarded | 2048 | 777.696 | 21.969 | 0.0005035 | 0.663 | 0 |
| rusticl | 1 | auto_t3_controlled | 1400 | 884.018 | 6.107 | 0.0003204 | 0.000 | 0 |
| rusticl | 1 | auto_t3_controlled | 2048 | 712.529 | 24.055 | 0.0005493 | 0.000 | 0 |
| rusticl | 1 | auto_t5_guarded | 1400 | 921.671 | 5.935 | 0.0002747 | 1.688 | 0 |
| rusticl | 1 | auto_t5_guarded | 2048 | 719.508 | 23.763 | 0.0005188 | 0.609 | 0 |
| rusticl | 2 | auto_t3_controlled | 1400 | 872.557 | 6.153 | 0.0002747 | 0.000 | 0 |
| rusticl | 2 | auto_t3_controlled | 2048 | 711.165 | 24.045 | 0.0005035 | 0.000 | 0 |
| rusticl | 2 | auto_t5_guarded | 1400 | 921.072 | 5.934 | 0.0003204 | 1.624 | 0 |
| rusticl | 2 | auto_t5_guarded | 2048 | 718.855 | 23.736 | 0.0007019 | 0.602 | 0 |
| rusticl | 3 | auto_t3_controlled | 1400 | 877.764 | 6.146 | 0.0002899 | 0.000 | 0 |
| rusticl | 3 | auto_t3_controlled | 2048 | 712.166 | 24.045 | 0.0004730 | 0.000 | 0 |
| rusticl | 3 | auto_t5_guarded | 1400 | 920.599 | 5.912 | 0.0003357 | 1.659 | 0 |
| rusticl | 3 | auto_t5_guarded | 2048 | 718.822 | 23.764 | 0.0005493 | 0.582 | 0 |
| rusticl | 4 | auto_t3_controlled | 1400 | 870.750 | 6.185 | 0.0004120 | 0.000 | 0 |
| rusticl | 4 | auto_t3_controlled | 2048 | 711.755 | 24.059 | 0.0004425 | 0.000 | 0 |
| rusticl | 4 | auto_t5_guarded | 1400 | 922.537 | 5.908 | 0.0003815 | 1.713 | 0 |
| rusticl | 4 | auto_t5_guarded | 2048 | 716.430 | 23.781 | 0.0005035 | 0.813 | 0 |

## Checks

| Check | Pass |
| --- | --- |
| all_runs_success | True |
| pressure_failures_zero | True |
| platform_split_clover_and_rusticl | True |
| correctness_bound_all_runs | True |
| t3_guardrails_all_runs | True |
| t5_guardrails_all_runs | True |
| rusticl_peak_ratio_min | True |
| burnin_drift_abs_percent | True |
| no_regression_vs_block4_clover | True |

## Rollback

- Script: `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`
- Trigger condition: execute rollback if any guardrail check fails or rusticl ratio drops below threshold.

## Decision

- Decision: `promote`
- Rationale: Extended pilot passed pressure, split-platform and regression checks.

