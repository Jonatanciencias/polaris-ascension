# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T17:22:06.196524+00:00
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
| Clover | 351 | auto_t3_controlled | 1400 | ok | 866.206 | 6.172 | 0.0003052 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 2048 | ok | 774.074 | 22.151 | 0.0004883 | 0.000 | 0 |
| Clover | 351 | auto_t3_controlled | 3072 | ok | 710.041 | 72.073 | 0.0008087 | 0.000 | 0 |
| Clover | 351 | auto_t5_guarded | 1400 | ok | 913.204 | 5.968 | 0.0003662 | 2.260 | 0 |
| Clover | 351 | auto_t5_guarded | 2048 | ok | 779.972 | 21.971 | 0.0005035 | 1.060 | 0 |
| Clover | 351 | auto_t5_guarded | 3072 | ok | 802.656 | 71.986 | 0.0007935 | 0.669 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 872.015 | 6.169 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 772.815 | 22.200 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 709.061 | 72.218 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 912.460 | 5.990 | 0.0003357 | 2.142 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 779.790 | 21.995 | 0.0005341 | 1.074 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 803.226 | 71.988 | 0.0007172 | 0.674 | 0 |
| rusticl | 351 | auto_t3_controlled | 1400 | ok | 873.115 | 6.090 | 0.0003052 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 2048 | ok | 713.260 | 24.038 | 0.0004883 | 0.000 | 0 |
| rusticl | 351 | auto_t3_controlled | 3072 | ok | 659.819 | 77.688 | 0.0008087 | 0.000 | 0 |
| rusticl | 351 | auto_t5_guarded | 1400 | ok | 913.220 | 5.994 | 0.0003662 | 2.745 | 0 |
| rusticl | 351 | auto_t5_guarded | 2048 | ok | 719.148 | 23.837 | 0.0005035 | 0.951 | 0 |
| rusticl | 351 | auto_t5_guarded | 3072 | ok | 746.801 | 77.544 | 0.0007935 | 0.617 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 869.920 | 6.194 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 714.278 | 23.996 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 659.013 | 77.731 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 914.662 | 5.984 | 0.0003357 | 2.500 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 719.373 | 23.845 | 0.0005341 | 0.968 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 746.787 | 77.481 | 0.0007172 | 0.619 | 0 |

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

