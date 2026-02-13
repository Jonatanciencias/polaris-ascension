# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-09T02:03:02.003315+00:00
- Seeds: [612, 712]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 3

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 612 | 3 | 3 | 0 |
| Clover | 712 | 3 | 3 | 0 |
| rusticl | 612 | 3 | 3 | 0 |
| rusticl | 712 | 3 | 3 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 612 | auto_t3_controlled | 1400 | ok | 885.190 | 6.090 | 0.0003357 | 0.000 | 0 |
| Clover | 612 | auto_t3_controlled | 2048 | ok | 772.192 | 22.182 | 0.0005341 | 0.000 | 0 |
| Clover | 612 | auto_t3_controlled | 3072 | ok | 705.961 | 72.078 | 0.0007172 | 0.000 | 0 |
| Clover | 612 | auto_t5_guarded | 1400 | ok | 911.239 | 5.977 | 0.0003510 | 1.282 | 0 |
| Clover | 612 | auto_t5_guarded | 2048 | ok | 777.491 | 21.993 | 0.0005035 | 0.585 | 0 |
| Clover | 612 | auto_t5_guarded | 3072 | ok | 804.559 | 71.943 | 0.0007782 | 0.367 | 0 |
| Clover | 712 | auto_t3_controlled | 1400 | ok | 871.524 | 6.135 | 0.0002823 | 0.000 | 0 |
| Clover | 712 | auto_t3_controlled | 2048 | ok | 772.168 | 22.203 | 0.0004883 | 0.000 | 0 |
| Clover | 712 | auto_t3_controlled | 3072 | ok | 705.651 | 72.064 | 0.0007782 | 0.000 | 0 |
| Clover | 712 | auto_t5_guarded | 1400 | ok | 910.050 | 5.969 | 0.0003662 | 1.277 | 0 |
| Clover | 712 | auto_t5_guarded | 2048 | ok | 777.509 | 21.988 | 0.0004883 | 0.591 | 0 |
| Clover | 712 | auto_t5_guarded | 3072 | ok | 803.368 | 71.941 | 0.0008698 | 0.363 | 0 |
| rusticl | 612 | auto_t3_controlled | 1400 | ok | 881.226 | 6.123 | 0.0003357 | 0.000 | 0 |
| rusticl | 612 | auto_t3_controlled | 2048 | ok | 712.657 | 24.032 | 0.0005341 | 0.000 | 0 |
| rusticl | 612 | auto_t3_controlled | 3072 | ok | 656.305 | 77.860 | 0.0007172 | 0.000 | 0 |
| rusticl | 612 | auto_t5_guarded | 1400 | ok | 920.991 | 5.933 | 0.0003510 | 1.409 | 0 |
| rusticl | 612 | auto_t5_guarded | 2048 | ok | 718.742 | 23.752 | 0.0005035 | 0.534 | 0 |
| rusticl | 612 | auto_t5_guarded | 3072 | ok | 745.006 | 77.427 | 0.0007782 | 0.346 | 0 |
| rusticl | 712 | auto_t3_controlled | 1400 | ok | 874.902 | 6.134 | 0.0002823 | 0.000 | 0 |
| rusticl | 712 | auto_t3_controlled | 2048 | ok | 712.423 | 24.034 | 0.0004883 | 0.000 | 0 |
| rusticl | 712 | auto_t3_controlled | 3072 | ok | 656.888 | 77.619 | 0.0007782 | 0.000 | 0 |
| rusticl | 712 | auto_t5_guarded | 1400 | ok | 920.836 | 5.934 | 0.0003662 | 1.472 | 0 |
| rusticl | 712 | auto_t5_guarded | 2048 | ok | 718.585 | 23.757 | 0.0004883 | 0.546 | 0 |
| rusticl | 712 | auto_t5_guarded | 3072 | ok | 743.117 | 77.482 | 0.0008698 | 0.360 | 0 |

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

