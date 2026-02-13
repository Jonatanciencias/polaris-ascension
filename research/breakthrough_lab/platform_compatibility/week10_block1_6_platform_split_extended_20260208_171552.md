# Week 10 Block 1.6 - Controlled Pre-Production Pilot (RX590)

- Date: 2026-02-08T17:15:52.802268+00:00
- Hourly snapshots (logical): [1, 2, 3, 4, 5, 6, 7, 8]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 2 | 2 | 0 |
| Clover | 2 | 2 | 2 | 0 |
| Clover | 3 | 2 | 2 | 0 |
| Clover | 4 | 2 | 2 | 0 |
| Clover | 5 | 2 | 2 | 0 |
| Clover | 6 | 2 | 2 | 0 |
| Clover | 7 | 2 | 2 | 0 |
| Clover | 8 | 2 | 2 | 0 |
| rusticl | 1 | 2 | 2 | 0 |
| rusticl | 2 | 2 | 2 | 0 |
| rusticl | 3 | 2 | 2 | 0 |
| rusticl | 4 | 2 | 2 | 0 |
| rusticl | 5 | 2 | 2 | 0 |
| rusticl | 6 | 2 | 2 | 0 |
| rusticl | 7 | 2 | 2 | 0 |
| rusticl | 8 | 2 | 2 | 0 |

## Pilot Matrix

| Platform | Snapshot | Kernel | Size | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clover | 1 | auto_t3_controlled | 1400 | 873.670 | 6.171 | 0.0003204 | 0.000 | 0 |
| Clover | 1 | auto_t3_controlled | 2048 | 774.346 | 22.095 | 0.0005493 | 0.000 | 0 |
| Clover | 1 | auto_t5_guarded | 1400 | 908.099 | 5.999 | 0.0002747 | 1.514 | 0 |
| Clover | 1 | auto_t5_guarded | 2048 | 779.335 | 21.981 | 0.0005188 | 0.669 | 0 |
| Clover | 2 | auto_t3_controlled | 1400 | 873.089 | 6.170 | 0.0002747 | 0.000 | 0 |
| Clover | 2 | auto_t3_controlled | 2048 | 771.809 | 22.197 | 0.0005035 | 0.000 | 0 |
| Clover | 2 | auto_t5_guarded | 1400 | 909.053 | 5.991 | 0.0003204 | 1.436 | 0 |
| Clover | 2 | auto_t5_guarded | 2048 | 777.220 | 21.986 | 0.0007019 | 0.636 | 0 |
| Clover | 3 | auto_t3_controlled | 1400 | 884.176 | 6.035 | 0.0002899 | 0.000 | 0 |
| Clover | 3 | auto_t3_controlled | 2048 | 771.694 | 22.206 | 0.0004730 | 0.000 | 0 |
| Clover | 3 | auto_t5_guarded | 1400 | 909.950 | 5.973 | 0.0003357 | 1.505 | 0 |
| Clover | 3 | auto_t5_guarded | 2048 | 777.983 | 21.986 | 0.0005493 | 0.652 | 0 |
| Clover | 4 | auto_t3_controlled | 1400 | 880.513 | 6.102 | 0.0004120 | 0.000 | 0 |
| Clover | 4 | auto_t3_controlled | 2048 | 772.430 | 22.186 | 0.0004425 | 0.000 | 0 |
| Clover | 4 | auto_t5_guarded | 1400 | 907.490 | 5.989 | 0.0003815 | 1.471 | 0 |
| Clover | 4 | auto_t5_guarded | 2048 | 777.359 | 21.989 | 0.0005035 | 0.643 | 0 |
| Clover | 5 | auto_t3_controlled | 1400 | 875.166 | 6.138 | 0.0003281 | 0.000 | 0 |
| Clover | 5 | auto_t3_controlled | 2048 | 771.103 | 22.243 | 0.0005341 | 0.000 | 0 |
| Clover | 5 | auto_t5_guarded | 1400 | 912.670 | 5.970 | 0.0003967 | 1.429 | 0 |
| Clover | 5 | auto_t5_guarded | 2048 | 763.886 | 21.985 | 0.0005188 | 0.655 | 0 |
| Clover | 6 | auto_t3_controlled | 1400 | 881.742 | 6.109 | 0.0002975 | 0.000 | 0 |
| Clover | 6 | auto_t3_controlled | 2048 | 771.222 | 22.244 | 0.0005035 | 0.000 | 0 |
| Clover | 6 | auto_t5_guarded | 1400 | 912.286 | 5.977 | 0.0003204 | 1.502 | 0 |
| Clover | 6 | auto_t5_guarded | 2048 | 777.617 | 21.974 | 0.0005188 | 0.660 | 0 |
| Clover | 7 | auto_t3_controlled | 1400 | 873.025 | 6.172 | 0.0003967 | 0.000 | 0 |
| Clover | 7 | auto_t3_controlled | 2048 | 773.196 | 22.142 | 0.0005798 | 0.000 | 0 |
| Clover | 7 | auto_t5_guarded | 1400 | 911.569 | 5.941 | 0.0003052 | 1.553 | 0 |
| Clover | 7 | auto_t5_guarded | 2048 | 777.926 | 22.003 | 0.0005493 | 0.658 | 0 |
| Clover | 8 | auto_t3_controlled | 1400 | 874.758 | 6.132 | 0.0003204 | 0.000 | 0 |
| Clover | 8 | auto_t3_controlled | 2048 | 772.361 | 22.210 | 0.0006409 | 0.000 | 0 |
| Clover | 8 | auto_t5_guarded | 1400 | 911.791 | 5.978 | 0.0002747 | 1.481 | 0 |
| Clover | 8 | auto_t5_guarded | 2048 | 777.819 | 21.984 | 0.0004883 | 0.660 | 0 |
| rusticl | 1 | auto_t3_controlled | 1400 | 874.417 | 6.180 | 0.0003204 | 0.000 | 0 |
| rusticl | 1 | auto_t3_controlled | 2048 | 712.055 | 24.016 | 0.0005493 | 0.000 | 0 |
| rusticl | 1 | auto_t5_guarded | 1400 | 920.808 | 5.939 | 0.0002747 | 1.756 | 0 |
| rusticl | 1 | auto_t5_guarded | 2048 | 718.182 | 23.788 | 0.0005188 | 0.641 | 0 |
| rusticl | 2 | auto_t3_controlled | 1400 | 878.265 | 6.105 | 0.0002747 | 0.000 | 0 |
| rusticl | 2 | auto_t3_controlled | 2048 | 710.989 | 24.104 | 0.0005035 | 0.000 | 0 |
| rusticl | 2 | auto_t5_guarded | 1400 | 920.784 | 5.946 | 0.0003204 | 1.629 | 0 |
| rusticl | 2 | auto_t5_guarded | 2048 | 718.461 | 23.790 | 0.0007019 | 0.606 | 0 |
| rusticl | 3 | auto_t3_controlled | 1400 | 876.591 | 6.144 | 0.0002899 | 0.000 | 0 |
| rusticl | 3 | auto_t3_controlled | 2048 | 714.057 | 23.983 | 0.0004730 | 0.000 | 0 |
| rusticl | 3 | auto_t5_guarded | 1400 | 920.070 | 5.935 | 0.0003357 | 1.693 | 0 |
| rusticl | 3 | auto_t5_guarded | 2048 | 718.283 | 23.805 | 0.0005493 | 0.594 | 0 |
| rusticl | 4 | auto_t3_controlled | 1400 | 874.203 | 6.191 | 0.0004120 | 0.000 | 0 |
| rusticl | 4 | auto_t3_controlled | 2048 | 712.021 | 24.064 | 0.0004425 | 0.000 | 0 |
| rusticl | 4 | auto_t5_guarded | 1400 | 921.803 | 5.920 | 0.0003815 | 1.679 | 0 |
| rusticl | 4 | auto_t5_guarded | 2048 | 718.447 | 23.794 | 0.0005035 | 0.598 | 0 |
| rusticl | 5 | auto_t3_controlled | 1400 | 871.271 | 6.164 | 0.0003281 | 0.000 | 0 |
| rusticl | 5 | auto_t3_controlled | 2048 | 710.772 | 24.103 | 0.0005341 | 0.000 | 0 |
| rusticl | 5 | auto_t5_guarded | 1400 | 920.840 | 5.930 | 0.0003967 | 1.682 | 0 |
| rusticl | 5 | auto_t5_guarded | 2048 | 718.222 | 23.733 | 0.0005188 | 0.617 | 0 |
| rusticl | 6 | auto_t3_controlled | 1400 | 883.686 | 6.140 | 0.0002975 | 0.000 | 0 |
| rusticl | 6 | auto_t3_controlled | 2048 | 712.357 | 24.037 | 0.0005035 | 0.000 | 0 |
| rusticl | 6 | auto_t5_guarded | 1400 | 921.220 | 5.927 | 0.0003204 | 1.714 | 0 |
| rusticl | 6 | auto_t5_guarded | 2048 | 719.173 | 23.716 | 0.0005188 | 0.603 | 0 |
| rusticl | 7 | auto_t3_controlled | 1400 | 874.364 | 6.156 | 0.0003967 | 0.000 | 0 |
| rusticl | 7 | auto_t3_controlled | 2048 | 713.347 | 23.998 | 0.0005798 | 0.000 | 0 |
| rusticl | 7 | auto_t5_guarded | 1400 | 920.950 | 5.925 | 0.0003052 | 1.741 | 0 |
| rusticl | 7 | auto_t5_guarded | 2048 | 718.468 | 23.825 | 0.0005493 | 0.628 | 0 |
| rusticl | 8 | auto_t3_controlled | 1400 | 873.425 | 6.156 | 0.0003204 | 0.000 | 0 |
| rusticl | 8 | auto_t3_controlled | 2048 | 712.019 | 24.079 | 0.0006409 | 0.000 | 0 |
| rusticl | 8 | auto_t5_guarded | 1400 | 921.291 | 5.931 | 0.0002747 | 1.706 | 0 |
| rusticl | 8 | auto_t5_guarded | 2048 | 718.150 | 23.719 | 0.0004883 | 0.606 | 0 |

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

