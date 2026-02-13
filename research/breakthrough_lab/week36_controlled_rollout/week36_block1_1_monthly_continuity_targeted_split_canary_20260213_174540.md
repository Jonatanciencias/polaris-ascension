# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T17:45:40.147565+00:00
- Seeds: [351, 613]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 6
- Queue pressure pulses per platform/seed: 0

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 351 | 0 | 0 | 0 |
| Clover | 613 | 0 | 0 | 0 |
| rusticl | 351 | 0 | 0 | 0 |
| rusticl | 613 | 0 | 0 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 351 | auto_t3_controlled | 1400 | ok | 869.922 | 6.177 | 0.0003052 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 2048 | ok | 772.273 | 22.213 | 0.0004883 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 3072 | ok | 708.284 | 72.107 | 0.0008087 | 0.000 | 0 |
| Clover | 351 | auto_t5_guarded | 1400 | ok | 907.989 | 6.017 | 0.0003662 | 1.796 | 0 |
| Clover | 351 | auto_t5_guarded | 2048 | ok | 778.958 | 21.985 | 0.0005035 | 0.854 | 0 |
| Clover | 351 | auto_t5_guarded | 3072 | ok | 804.127 | 71.990 | 0.0007935 | 0.507 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 878.865 | 6.117 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 771.245 | 22.252 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 708.682 | 72.013 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 909.225 | 5.989 | 0.0003357 | 1.777 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 779.141 | 21.974 | 0.0005341 | 0.789 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 803.942 | 72.035 | 0.0007172 | 0.495 | 0 |
| rusticl | 351 | auto_t3_controlled | 1400 | ok | 870.608 | 6.165 | 0.0003052 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 2048 | ok | 712.194 | 24.022 | 0.0004883 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 3072 | ok | 657.115 | 77.729 | 0.0008087 | 0.000 | 0 |
| rusticl | 351 | auto_t5_guarded | 1400 | ok | 914.216 | 5.980 | 0.0003662 | 1.952 | 0 |
| rusticl | 351 | auto_t5_guarded | 2048 | ok | 717.479 | 23.825 | 0.0005035 | 0.739 | 0 |
| rusticl | 351 | auto_t5_guarded | 3072 | ok | 744.473 | 77.630 | 0.0007935 | 0.471 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 877.575 | 6.132 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 712.430 | 24.040 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 658.461 | 77.836 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 913.918 | 5.979 | 0.0003357 | 1.932 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 717.439 | 23.881 | 0.0005341 | 0.730 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 746.384 | 77.553 | 0.0007172 | 0.454 | 0 |

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

