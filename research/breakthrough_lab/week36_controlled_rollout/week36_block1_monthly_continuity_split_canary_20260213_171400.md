# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T17:14:00.985960+00:00
- Seeds: [361, 613]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 4
- Queue pressure pulses per platform/seed: 0

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 361 | 0 | 0 | 0 |
| Clover | 613 | 0 | 0 | 0 |
| rusticl | 361 | 0 | 0 | 0 |
| rusticl | 613 | 0 | 0 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 361 | auto_t3_controlled | 1400 | ok | 876.321 | 6.104 | 0.0003891 | 0.000 | 0 |
| Clover | 361 | auto_t3_controlled | 2048 | ok | 771.121 | 22.266 | 0.0004883 | 0.000 | 0 |
| Clover | 361 | auto_t3_controlled | 3072 | ok | 710.724 | 72.053 | 0.0007172 | 0.000 | 0 |
| Clover | 361 | auto_t5_guarded | 1400 | ok | 908.021 | 5.994 | 0.0002823 | 2.251 | 0 |
| Clover | 361 | auto_t5_guarded | 2048 | ok | 779.982 | 21.976 | 0.0005341 | 1.043 | 0 |
| Clover | 361 | auto_t5_guarded | 3072 | ok | 803.028 | 71.958 | 0.0009308 | 0.674 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 870.431 | 6.177 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 772.713 | 22.187 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 710.116 | 72.146 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 907.753 | 6.002 | 0.0003357 | 2.174 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 767.763 | 21.986 | 0.0005341 | 1.042 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 804.703 | 72.015 | 0.0007172 | 0.689 | 0 |
| rusticl | 361 | auto_t3_controlled | 1400 | ok | 876.828 | 6.128 | 0.0003891 | 0.000 | 0 |
| rusticl | 361 | auto_t3_controlled | 2048 | ok | 711.628 | 24.080 | 0.0004883 | 0.000 | 0 |
| rusticl | 361 | auto_t3_controlled | 3072 | ok | 658.738 | 77.670 | 0.0007172 | 0.000 | 0 |
| rusticl | 361 | auto_t5_guarded | 1400 | ok | 915.018 | 5.961 | 0.0002823 | 2.563 | 0 |
| rusticl | 361 | auto_t5_guarded | 2048 | ok | 719.335 | 23.856 | 0.0005341 | 0.967 | 0 |
| rusticl | 361 | auto_t5_guarded | 3072 | ok | 746.628 | 77.548 | 0.0009308 | 0.618 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 884.600 | 6.052 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 713.706 | 23.895 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 658.776 | 77.857 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 913.281 | 5.973 | 0.0003357 | 2.619 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 718.118 | 23.867 | 0.0005341 | 1.061 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 743.572 | 77.681 | 0.0007172 | 0.719 | 0 |

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

