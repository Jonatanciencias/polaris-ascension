# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-09T02:46:57.326529+00:00
- Seeds: [1012, 1112]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 3

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1012 | 3 | 3 | 0 |
| Clover | 1112 | 3 | 3 | 0 |
| rusticl | 1012 | 3 | 3 | 0 |
| rusticl | 1112 | 3 | 3 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 1012 | auto_t3_controlled | 1400 | ok | 881.194 | 6.051 | 0.0003281 | 0.000 | 0 |
| Clover | 1012 | auto_t3_controlled | 2048 | ok | 772.802 | 22.139 | 0.0005646 | 0.000 | 0 |
| Clover | 1012 | auto_t3_controlled | 3072 | ok | 704.008 | 72.063 | 0.0007477 | 0.000 | 0 |
| Clover | 1012 | auto_t5_guarded | 1400 | ok | 905.013 | 6.010 | 0.0003204 | 1.300 | 0 |
| Clover | 1012 | auto_t5_guarded | 2048 | ok | 777.570 | 21.970 | 0.0005951 | 0.589 | 0 |
| Clover | 1012 | auto_t5_guarded | 3072 | ok | 804.162 | 71.957 | 0.0007324 | 0.395 | 0 |
| Clover | 1112 | auto_t3_controlled | 1400 | ok | 869.851 | 6.146 | 0.0003586 | 0.000 | 0 |
| Clover | 1112 | auto_t3_controlled | 2048 | ok | 773.693 | 22.136 | 0.0004272 | 0.000 | 0 |
| Clover | 1112 | auto_t3_controlled | 3072 | ok | 693.714 | 72.114 | 0.0007324 | 0.000 | 0 |
| Clover | 1112 | auto_t5_guarded | 1400 | ok | 906.535 | 5.985 | 0.0003052 | 1.213 | 0 |
| Clover | 1112 | auto_t5_guarded | 2048 | ok | 778.445 | 21.956 | 0.0005341 | 0.546 | 0 |
| Clover | 1112 | auto_t5_guarded | 3072 | ok | 803.185 | 71.988 | 0.0007935 | 0.352 | 0 |
| rusticl | 1012 | auto_t3_controlled | 1400 | ok | 877.925 | 6.137 | 0.0003281 | 0.000 | 0 |
| rusticl | 1012 | auto_t3_controlled | 2048 | ok | 714.212 | 24.001 | 0.0005646 | 0.000 | 0 |
| rusticl | 1012 | auto_t3_controlled | 3072 | ok | 655.133 | 77.739 | 0.0007477 | 0.000 | 0 |
| rusticl | 1012 | auto_t5_guarded | 1400 | ok | 921.672 | 5.917 | 0.0003204 | 1.615 | 0 |
| rusticl | 1012 | auto_t5_guarded | 2048 | ok | 717.888 | 23.822 | 0.0005951 | 0.517 | 0 |
| rusticl | 1012 | auto_t5_guarded | 3072 | ok | 745.820 | 77.574 | 0.0007324 | 0.327 | 0 |
| rusticl | 1112 | auto_t3_controlled | 1400 | ok | 876.129 | 6.165 | 0.0003586 | 0.000 | 0 |
| rusticl | 1112 | auto_t3_controlled | 2048 | ok | 711.656 | 24.055 | 0.0004272 | 0.000 | 0 |
| rusticl | 1112 | auto_t3_controlled | 3072 | ok | 655.828 | 77.787 | 0.0007324 | 0.000 | 0 |
| rusticl | 1112 | auto_t5_guarded | 1400 | ok | 835.831 | 5.941 | 0.0003052 | 1.215 | 0 |
| rusticl | 1112 | auto_t5_guarded | 2048 | ok | 719.067 | 23.790 | 0.0005341 | 0.518 | 0 |
| rusticl | 1112 | auto_t5_guarded | 3072 | ok | 746.361 | 77.461 | 0.0007935 | 0.336 | 0 |

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

