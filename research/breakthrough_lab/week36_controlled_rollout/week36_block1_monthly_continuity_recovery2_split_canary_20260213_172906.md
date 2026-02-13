# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T17:29:06.433562+00:00
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
| Clover | 361 | auto_t3_controlled | 1400 | ok | 869.780 | 6.137 | 0.0003891 | 0.000 | 0 |
| Clover | 361 | auto_t3_controlled | 2048 | ok | 773.203 | 22.205 | 0.0004883 | 0.000 | 0 |
| Clover | 361 | auto_t3_controlled | 3072 | ok | 708.119 | 72.094 | 0.0007172 | 0.000 | 0 |
| Clover | 361 | auto_t5_guarded | 1400 | ok | 907.081 | 5.984 | 0.0002823 | 2.657 | 0 |
| Clover | 361 | auto_t5_guarded | 2048 | ok | 780.415 | 21.968 | 0.0005341 | 1.268 | 0 |
| Clover | 361 | auto_t5_guarded | 3072 | ok | 802.706 | 72.003 | 0.0009308 | 0.754 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 872.132 | 6.146 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 771.704 | 22.223 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 708.885 | 72.041 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 911.602 | 5.978 | 0.0003357 | 2.654 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 780.301 | 21.974 | 0.0005341 | 1.182 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 804.758 | 71.968 | 0.0007172 | 0.733 | 0 |
| rusticl | 361 | auto_t3_controlled | 1400 | ok | 869.655 | 6.155 | 0.0003891 | 0.000 | 0 |
| rusticl | 361 | auto_t3_controlled | 2048 | ok | 713.962 | 23.997 | 0.0004883 | 0.000 | 0 |
| rusticl | 361 | auto_t3_controlled | 3072 | ok | 659.030 | 77.778 | 0.0007172 | 0.000 | 0 |
| rusticl | 361 | auto_t5_guarded | 1400 | ok | 913.372 | 5.984 | 0.0002823 | 3.137 | 0 |
| rusticl | 361 | auto_t5_guarded | 2048 | ok | 719.000 | 23.858 | 0.0005341 | 1.094 | 0 |
| rusticl | 361 | auto_t5_guarded | 3072 | ok | 744.108 | 77.561 | 0.0009308 | 0.687 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 865.782 | 6.174 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 711.613 | 24.081 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 658.956 | 77.708 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 913.242 | 5.990 | 0.0003357 | 3.190 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 718.515 | 23.887 | 0.0005341 | 1.120 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 746.701 | 77.576 | 0.0007172 | 0.687 | 0 |

## Checks

| Check | Pass |
| --- | --- |
| all_runs_success | True |
| pressure_failures_zero | True |
| platform_split_clover_and_rusticl | True |
| correctness_bound_all_runs | True |
| t3_guardrails_all_runs | True |
| t5_guardrails_all_runs | False |
| rusticl_peak_ratio_min | True |
| no_regression_vs_block3_clover | True |

## Decision

- Decision: `iterate`
- Rationale: Stress replay found one or more platform/guardrail/regression failures.

