# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T01:32:43.337867+00:00
- Seeds: [211, 509]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 2

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 211 | 2 | 2 | 0 |
| Clover | 509 | 2 | 2 | 0 |
| rusticl | 211 | 2 | 2 | 0 |
| rusticl | 509 | 2 | 2 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 211 | auto_t3_controlled | 1400 | ok | 873.700 | 6.127 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.988 | 22.203 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 706.075 | 72.156 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 909.888 | 5.983 | 0.0003357 | 1.090 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.852 | 21.975 | 0.0005264 | 0.526 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 803.896 | 71.985 | 0.0007935 | 0.345 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 876.497 | 6.088 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.829 | 22.171 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 706.127 | 72.180 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 908.076 | 5.974 | 0.0003204 | 1.111 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 778.966 | 21.975 | 0.0005646 | 0.506 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.512 | 71.954 | 0.0007477 | 0.333 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 880.168 | 6.137 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 714.404 | 23.997 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.371 | 77.781 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 916.232 | 5.964 | 0.0003357 | 1.366 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 694.102 | 23.901 | 0.0005264 | 0.454 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 745.716 | 77.556 | 0.0007935 | 0.307 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 881.512 | 6.089 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 711.125 | 24.051 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 657.320 | 77.723 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 915.018 | 5.976 | 0.0003204 | 1.277 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 716.234 | 23.878 | 0.0005646 | 0.479 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 745.561 | 77.598 | 0.0007477 | 0.306 | 0 |

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
| no_regression_vs_block3_clover | True |

## Decision

- Decision: `promote`
- Rationale: Stress replay passed queue-pressure and split-platform checks without regressions.

