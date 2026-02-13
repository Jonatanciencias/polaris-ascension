# Week 9 Block 5 - Controlled Pre-Production Pilot (RX590)

- Date: 2026-02-08T03:52:40.209121+00:00
- Hourly snapshots (logical): [1, 2, 3, 4, 5, 6]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']

## Pressure Summary

| Platform | Snapshot | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1 | 3 | 3 | 0 |
| Clover | 2 | 3 | 3 | 0 |
| Clover | 3 | 3 | 3 | 0 |
| Clover | 4 | 3 | 3 | 0 |
| Clover | 5 | 3 | 3 | 0 |
| Clover | 6 | 3 | 3 | 0 |
| rusticl | 1 | 3 | 3 | 0 |
| rusticl | 2 | 3 | 3 | 0 |
| rusticl | 3 | 3 | 3 | 0 |
| rusticl | 4 | 3 | 3 | 0 |
| rusticl | 5 | 3 | 3 | 0 |
| rusticl | 6 | 3 | 3 | 0 |

## Pilot Matrix

| Platform | Snapshot | Kernel | Size | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clover | 1 | auto_t3_controlled | 1400 | 883.498 | 6.077 | 0.0003204 | 0.000 | 0 |
| Clover | 1 | auto_t3_controlled | 2048 | 774.372 | 22.134 | 0.0005493 | 0.000 | 0 |
| Clover | 1 | auto_t5_guarded | 1400 | 906.683 | 6.003 | 0.0002747 | 2.227 | 0 |
| Clover | 1 | auto_t5_guarded | 2048 | 778.639 | 21.975 | 0.0005188 | 0.957 | 0 |
| Clover | 2 | auto_t3_controlled | 1400 | 875.150 | 6.123 | 0.0002747 | 0.000 | 0 |
| Clover | 2 | auto_t3_controlled | 2048 | 772.014 | 22.210 | 0.0005035 | 0.000 | 0 |
| Clover | 2 | auto_t5_guarded | 1400 | 909.258 | 5.991 | 0.0003204 | 2.139 | 0 |
| Clover | 2 | auto_t5_guarded | 2048 | 778.903 | 21.991 | 0.0007019 | 0.944 | 0 |
| Clover | 3 | auto_t3_controlled | 1400 | 873.934 | 6.112 | 0.0002899 | 0.000 | 0 |
| Clover | 3 | auto_t3_controlled | 2048 | 772.442 | 22.190 | 0.0004730 | 0.000 | 0 |
| Clover | 3 | auto_t5_guarded | 1400 | 910.626 | 5.972 | 0.0003357 | 2.241 | 0 |
| Clover | 3 | auto_t5_guarded | 2048 | 778.363 | 21.984 | 0.0005493 | 0.985 | 0 |
| Clover | 4 | auto_t3_controlled | 1400 | 874.025 | 6.122 | 0.0004120 | 0.000 | 0 |
| Clover | 4 | auto_t3_controlled | 2048 | 774.286 | 22.154 | 0.0004425 | 0.000 | 0 |
| Clover | 4 | auto_t5_guarded | 1400 | 906.759 | 6.028 | 0.0003815 | 2.096 | 0 |
| Clover | 4 | auto_t5_guarded | 2048 | 778.361 | 21.974 | 0.0005035 | 0.977 | 0 |
| Clover | 5 | auto_t3_controlled | 1400 | 873.675 | 6.157 | 0.0003281 | 0.000 | 0 |
| Clover | 5 | auto_t3_controlled | 2048 | 772.287 | 22.194 | 0.0005341 | 0.000 | 0 |
| Clover | 5 | auto_t5_guarded | 1400 | 909.634 | 5.977 | 0.0003967 | 2.147 | 0 |
| Clover | 5 | auto_t5_guarded | 2048 | 778.323 | 21.992 | 0.0005188 | 0.943 | 0 |
| Clover | 6 | auto_t3_controlled | 1400 | 871.296 | 6.183 | 0.0002975 | 0.000 | 0 |
| Clover | 6 | auto_t3_controlled | 2048 | 773.862 | 22.102 | 0.0005035 | 0.000 | 0 |
| Clover | 6 | auto_t5_guarded | 1400 | 911.388 | 5.936 | 0.0003204 | 2.315 | 0 |
| Clover | 6 | auto_t5_guarded | 2048 | 779.473 | 22.001 | 0.0005188 | 1.056 | 0 |
| rusticl | 1 | auto_t3_controlled | 1400 | 872.322 | 6.191 | 0.0003204 | 0.000 | 0 |
| rusticl | 1 | auto_t3_controlled | 2048 | 715.173 | 23.989 | 0.0005493 | 0.000 | 0 |
| rusticl | 1 | auto_t5_guarded | 1400 | 920.726 | 5.939 | 0.0002747 | 2.476 | 0 |
| rusticl | 1 | auto_t5_guarded | 2048 | 719.472 | 23.790 | 0.0005188 | 0.895 | 0 |
| rusticl | 2 | auto_t3_controlled | 1400 | 873.407 | 6.130 | 0.0002747 | 0.000 | 0 |
| rusticl | 2 | auto_t3_controlled | 2048 | 713.279 | 24.029 | 0.0005035 | 0.000 | 0 |
| rusticl | 2 | auto_t5_guarded | 1400 | 920.322 | 5.944 | 0.0003204 | 2.604 | 0 |
| rusticl | 2 | auto_t5_guarded | 2048 | 719.430 | 23.798 | 0.0007019 | 0.957 | 0 |
| rusticl | 3 | auto_t3_controlled | 1400 | 877.573 | 6.118 | 0.0002899 | 0.000 | 0 |
| rusticl | 3 | auto_t3_controlled | 2048 | 713.014 | 24.054 | 0.0004730 | 0.000 | 0 |
| rusticl | 3 | auto_t5_guarded | 1400 | 919.172 | 5.934 | 0.0003357 | 2.347 | 0 |
| rusticl | 3 | auto_t5_guarded | 2048 | 719.431 | 23.767 | 0.0005493 | 0.878 | 0 |
| rusticl | 4 | auto_t3_controlled | 1400 | 877.251 | 6.117 | 0.0004120 | 0.000 | 0 |
| rusticl | 4 | auto_t3_controlled | 2048 | 712.771 | 24.061 | 0.0004425 | 0.000 | 0 |
| rusticl | 4 | auto_t5_guarded | 1400 | 921.253 | 5.929 | 0.0003815 | 2.373 | 0 |
| rusticl | 4 | auto_t5_guarded | 2048 | 719.399 | 23.804 | 0.0005035 | 0.873 | 0 |
| rusticl | 5 | auto_t3_controlled | 1400 | 877.548 | 6.133 | 0.0003281 | 0.000 | 0 |
| rusticl | 5 | auto_t3_controlled | 2048 | 712.427 | 24.051 | 0.0005341 | 0.000 | 0 |
| rusticl | 5 | auto_t5_guarded | 1400 | 920.608 | 5.935 | 0.0003967 | 2.374 | 0 |
| rusticl | 5 | auto_t5_guarded | 2048 | 719.374 | 23.786 | 0.0005188 | 0.867 | 0 |
| rusticl | 6 | auto_t3_controlled | 1400 | 870.432 | 6.156 | 0.0002975 | 0.000 | 0 |
| rusticl | 6 | auto_t3_controlled | 2048 | 712.006 | 24.080 | 0.0005035 | 0.000 | 0 |
| rusticl | 6 | auto_t5_guarded | 1400 | 919.836 | 5.938 | 0.0003204 | 2.419 | 0 |
| rusticl | 6 | auto_t5_guarded | 2048 | 716.980 | 23.833 | 0.0005188 | 0.874 | 0 |

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

