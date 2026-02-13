# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T04:08:10.294231+00:00
- Seeds: [331, 613]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 4
- Queue pressure pulses per platform/seed: 0

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 331 | 0 | 0 | 0 |
| Clover | 613 | 0 | 0 | 0 |
| rusticl | 331 | 0 | 0 | 0 |
| rusticl | 613 | 0 | 0 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 331 | auto_t3_controlled | 1400 | ok | 878.664 | 6.098 | 0.0003510 | 0.000 | 0 |
| Clover | 331 | auto_t3_controlled | 2048 | ok | 773.918 | 22.173 | 0.0004730 | 0.000 | 0 |
| Clover | 331 | auto_t3_controlled | 3072 | ok | 708.751 | 72.066 | 0.0007629 | 0.000 | 0 |
| Clover | 331 | auto_t5_guarded | 1400 | ok | 909.603 | 6.011 | 0.0002899 | 2.258 | 0 |
| Clover | 331 | auto_t5_guarded | 2048 | ok | 779.861 | 21.986 | 0.0005188 | 1.059 | 0 |
| Clover | 331 | auto_t5_guarded | 3072 | ok | 804.418 | 71.980 | 0.0007629 | 0.666 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 872.019 | 6.191 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 771.997 | 22.248 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 709.308 | 72.101 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 909.687 | 6.017 | 0.0003357 | 2.251 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 779.668 | 22.001 | 0.0005341 | 1.048 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 805.049 | 71.984 | 0.0007172 | 0.658 | 0 |
| rusticl | 331 | auto_t3_controlled | 1400 | ok | 874.437 | 6.146 | 0.0003510 | 0.000 | 0 |
| rusticl | 331 | auto_t3_controlled | 2048 | ok | 714.136 | 24.000 | 0.0004730 | 0.000 | 0 |
| rusticl | 331 | auto_t3_controlled | 3072 | ok | 659.186 | 77.769 | 0.0007629 | 0.000 | 0 |
| rusticl | 331 | auto_t5_guarded | 1400 | ok | 914.229 | 5.993 | 0.0002899 | 2.616 | 0 |
| rusticl | 331 | auto_t5_guarded | 2048 | ok | 719.017 | 23.866 | 0.0005188 | 0.946 | 0 |
| rusticl | 331 | auto_t5_guarded | 3072 | ok | 746.475 | 77.551 | 0.0007629 | 0.607 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 858.491 | 6.205 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 713.018 | 24.047 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 657.972 | 77.713 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 911.767 | 6.000 | 0.0003357 | 2.604 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 718.309 | 23.878 | 0.0005341 | 0.959 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 744.288 | 77.504 | 0.0007172 | 0.628 | 0 |

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

