# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T04:27:36.896228+00:00
- Seeds: [211, 509]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 4
- Queue pressure pulses per platform/seed: 0

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 211 | 0 | 0 | 0 |
| Clover | 509 | 0 | 0 | 0 |
| rusticl | 211 | 0 | 0 | 0 |
| rusticl | 509 | 0 | 0 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 211 | auto_t3_controlled | 1400 | ok | 878.041 | 6.055 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.241 | 22.173 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 734.068 | 72.040 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 908.660 | 5.984 | 0.0003357 | 2.185 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 780.148 | 21.968 | 0.0005264 | 1.046 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.692 | 71.985 | 0.0007935 | 0.665 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 881.353 | 6.026 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 773.711 | 22.131 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 709.565 | 72.096 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 910.281 | 6.016 | 0.0003204 | 2.232 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 779.907 | 21.990 | 0.0005646 | 1.028 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.629 | 71.977 | 0.0007477 | 0.660 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 865.757 | 6.224 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 714.043 | 24.039 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.947 | 77.752 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 913.902 | 5.979 | 0.0003357 | 2.602 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.741 | 23.868 | 0.0005264 | 0.945 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.065 | 77.620 | 0.0007935 | 0.616 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 871.334 | 6.142 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 714.337 | 23.995 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.257 | 77.751 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 915.282 | 5.969 | 0.0003204 | 2.649 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 717.998 | 23.862 | 0.0005646 | 0.954 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.701 | 77.609 | 0.0007477 | 0.616 | 0 |

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

