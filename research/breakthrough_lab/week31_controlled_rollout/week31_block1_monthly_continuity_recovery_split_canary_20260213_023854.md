# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T02:38:54.086478+00:00
- Seeds: [311, 607]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 6
- Queue pressure pulses per platform/seed: 1

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 311 | 1 | 1 | 0 |
| Clover | 607 | 1 | 1 | 0 |
| rusticl | 311 | 1 | 1 | 0 |
| rusticl | 607 | 1 | 1 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 311 | auto_t3_controlled | 1400 | ok | 869.411 | 6.165 | 0.0003204 | 0.000 | 0 |
| Clover | 311 | auto_t3_controlled | 2048 | ok | 773.431 | 22.138 | 0.0005035 | 0.000 | 0 |
| Clover | 311 | auto_t3_controlled | 3072 | ok | 708.295 | 72.159 | 0.0007477 | 0.000 | 0 |
| Clover | 311 | auto_t5_guarded | 1400 | ok | 904.235 | 6.019 | 0.0003052 | 1.445 | 0 |
| Clover | 311 | auto_t5_guarded | 2048 | ok | 778.282 | 21.981 | 0.0004883 | 0.712 | 0 |
| Clover | 311 | auto_t5_guarded | 3072 | ok | 804.430 | 72.006 | 0.0008240 | 0.440 | 0 |
| Clover | 607 | auto_t3_controlled | 1400 | ok | 870.702 | 6.218 | 0.0003510 | 0.000 | 0 |
| Clover | 607 | auto_t3_controlled | 2048 | ok | 774.249 | 22.168 | 0.0005035 | 0.000 | 0 |
| Clover | 607 | auto_t3_controlled | 3072 | ok | 706.325 | 72.102 | 0.0007477 | 0.000 | 0 |
| Clover | 607 | auto_t5_guarded | 1400 | ok | 905.792 | 6.011 | 0.0002899 | 1.458 | 0 |
| Clover | 607 | auto_t5_guarded | 2048 | ok | 778.572 | 21.985 | 0.0005035 | 0.700 | 0 |
| Clover | 607 | auto_t5_guarded | 3072 | ok | 804.223 | 71.944 | 0.0007324 | 0.444 | 0 |
| rusticl | 311 | auto_t3_controlled | 1400 | ok | 878.395 | 6.121 | 0.0003204 | 0.000 | 0 |
| rusticl | 311 | auto_t3_controlled | 2048 | ok | 711.075 | 24.085 | 0.0005035 | 0.000 | 0 |
| rusticl | 311 | auto_t3_controlled | 3072 | ok | 657.615 | 77.789 | 0.0007477 | 0.000 | 0 |
| rusticl | 311 | auto_t5_guarded | 1400 | ok | 888.789 | 5.949 | 0.0003052 | 1.467 | 0 |
| rusticl | 311 | auto_t5_guarded | 2048 | ok | 718.310 | 23.868 | 0.0004883 | 0.638 | 0 |
| rusticl | 311 | auto_t5_guarded | 3072 | ok | 745.946 | 77.551 | 0.0008240 | 0.408 | 0 |
| rusticl | 607 | auto_t3_controlled | 1400 | ok | 880.677 | 6.116 | 0.0003510 | 0.000 | 0 |
| rusticl | 607 | auto_t3_controlled | 2048 | ok | 713.193 | 23.989 | 0.0005035 | 0.000 | 0 |
| rusticl | 607 | auto_t3_controlled | 3072 | ok | 657.932 | 77.810 | 0.0007477 | 0.000 | 0 |
| rusticl | 607 | auto_t5_guarded | 1400 | ok | 912.406 | 5.961 | 0.0002899 | 1.716 | 0 |
| rusticl | 607 | auto_t5_guarded | 2048 | ok | 717.676 | 23.804 | 0.0005035 | 0.628 | 0 |
| rusticl | 607 | auto_t5_guarded | 3072 | ok | 745.844 | 77.602 | 0.0007324 | 0.414 | 0 |

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

