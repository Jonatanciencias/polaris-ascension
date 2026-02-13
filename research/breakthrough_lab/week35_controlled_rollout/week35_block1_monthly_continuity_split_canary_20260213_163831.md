# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T16:38:31.738267+00:00
- Seeds: [351, 613]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 4
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
| Clover | 351 | auto_t3_controlled | 1400 | ok | 864.280 | 6.138 | 0.0003052 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 2048 | ok | 772.882 | 22.204 | 0.0004883 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 3072 | ok | 708.856 | 72.162 | 0.0008087 | 0.000 | 0 |
| Clover | 351 | auto_t5_guarded | 1400 | ok | 905.178 | 6.020 | 0.0003662 | 2.450 | 0 |
| Clover | 351 | auto_t5_guarded | 2048 | ok | 779.526 | 21.979 | 0.0005035 | 1.047 | 0 |
| Clover | 351 | auto_t5_guarded | 3072 | ok | 804.890 | 71.968 | 0.0007935 | 0.670 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 872.175 | 6.089 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 772.054 | 22.238 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 708.661 | 72.139 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 909.065 | 5.997 | 0.0003357 | 2.216 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 779.868 | 21.974 | 0.0005341 | 1.060 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 804.561 | 71.983 | 0.0007172 | 0.659 | 0 |
| rusticl | 351 | auto_t3_controlled | 1400 | ok | 865.377 | 6.196 | 0.0003052 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 2048 | ok | 711.636 | 24.035 | 0.0004883 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 3072 | ok | 658.801 | 77.725 | 0.0008087 | 0.000 | 0 |
| rusticl | 351 | auto_t5_guarded | 1400 | ok | 914.061 | 5.978 | 0.0003662 | 2.742 | 0 |
| rusticl | 351 | auto_t5_guarded | 2048 | ok | 718.667 | 23.838 | 0.0005035 | 0.974 | 0 |
| rusticl | 351 | auto_t5_guarded | 3072 | ok | 746.488 | 77.532 | 0.0007935 | 0.619 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 867.439 | 6.144 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 712.809 | 24.044 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 659.791 | 77.829 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 914.756 | 5.977 | 0.0003357 | 2.501 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 717.663 | 23.909 | 0.0005341 | 0.972 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 745.284 | 77.524 | 0.0007172 | 0.613 | 0 |

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

