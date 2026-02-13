# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T01:42:35.604518+00:00
- Seeds: [181, 313]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 2

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 181 | 2 | 2 | 0 |
| Clover | 313 | 2 | 2 | 0 |
| rusticl | 181 | 2 | 2 | 0 |
| rusticl | 313 | 2 | 2 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 181 | auto_t3_controlled | 1400 | ok | 877.924 | 6.111 | 0.0003204 | 0.000 | 0 |
| Clover | 181 | auto_t3_controlled | 2048 | ok | 771.899 | 22.192 | 0.0004883 | 0.000 | 0 |
| Clover | 181 | auto_t3_controlled | 3072 | ok | 710.119 | 72.005 | 0.0007782 | 0.000 | 0 |
| Clover | 181 | auto_t5_guarded | 1400 | ok | 904.685 | 6.020 | 0.0003052 | 1.165 | 0 |
| Clover | 181 | auto_t5_guarded | 2048 | ok | 778.505 | 21.965 | 0.0004883 | 0.516 | 0 |
| Clover | 181 | auto_t5_guarded | 3072 | ok | 804.181 | 71.975 | 0.0007477 | 0.353 | 0 |
| Clover | 313 | auto_t3_controlled | 1400 | ok | 878.782 | 6.054 | 0.0002975 | 0.000 | 0 |
| Clover | 313 | auto_t3_controlled | 2048 | ok | 773.155 | 22.092 | 0.0005493 | 0.000 | 0 |
| Clover | 313 | auto_t3_controlled | 3072 | ok | 709.122 | 72.100 | 0.0007477 | 0.000 | 0 |
| Clover | 313 | auto_t5_guarded | 1400 | ok | 904.231 | 6.020 | 0.0003204 | 1.077 | 0 |
| Clover | 313 | auto_t5_guarded | 2048 | ok | 778.313 | 21.967 | 0.0005035 | 0.520 | 0 |
| Clover | 313 | auto_t5_guarded | 3072 | ok | 804.360 | 71.961 | 0.0007629 | 0.330 | 0 |
| rusticl | 181 | auto_t3_controlled | 1400 | ok | 873.764 | 6.142 | 0.0003204 | 0.000 | 0 |
| rusticl | 181 | auto_t3_controlled | 2048 | ok | 713.062 | 23.976 | 0.0004883 | 0.000 | 0 |
| rusticl | 181 | auto_t3_controlled | 3072 | ok | 660.972 | 77.726 | 0.0007782 | 0.000 | 0 |
| rusticl | 181 | auto_t5_guarded | 1400 | ok | 835.198 | 5.943 | 0.0003052 | 1.159 | 0 |
| rusticl | 181 | auto_t5_guarded | 2048 | ok | 711.337 | 23.793 | 0.0004883 | 0.469 | 0 |
| rusticl | 181 | auto_t5_guarded | 3072 | ok | 746.379 | 77.509 | 0.0007477 | 0.313 | 0 |
| rusticl | 313 | auto_t3_controlled | 1400 | ok | 877.351 | 6.163 | 0.0002975 | 0.000 | 0 |
| rusticl | 313 | auto_t3_controlled | 2048 | ok | 712.175 | 24.056 | 0.0005493 | 0.000 | 0 |
| rusticl | 313 | auto_t3_controlled | 3072 | ok | 660.801 | 77.571 | 0.0007477 | 0.000 | 0 |
| rusticl | 313 | auto_t5_guarded | 1400 | ok | 919.788 | 5.939 | 0.0003204 | 1.402 | 0 |
| rusticl | 313 | auto_t5_guarded | 2048 | ok | 718.003 | 23.791 | 0.0005035 | 0.476 | 0 |
| rusticl | 313 | auto_t5_guarded | 3072 | ok | 744.883 | 77.456 | 0.0007629 | 0.338 | 0 |

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

