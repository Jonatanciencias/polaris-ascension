# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T02:45:15.677286+00:00
- Seeds: [313, 613]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 4
- Queue pressure pulses per platform/seed: 0

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 313 | 0 | 0 | 0 |
| Clover | 613 | 0 | 0 | 0 |
| rusticl | 313 | 0 | 0 | 0 |
| rusticl | 613 | 0 | 0 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 313 | auto_t3_controlled | 1400 | ok | 861.917 | 6.221 | 0.0002975 | 0.000 | 0 |
| Clover | 313 | auto_t3_controlled | 2048 | ok | 774.879 | 22.109 | 0.0005493 | 0.000 | 0 |
| Clover | 313 | auto_t3_controlled | 3072 | ok | 709.044 | 72.101 | 0.0007477 | 0.000 | 0 |
| Clover | 313 | auto_t5_guarded | 1400 | ok | 906.979 | 5.995 | 0.0003204 | 2.221 | 0 |
| Clover | 313 | auto_t5_guarded | 2048 | ok | 779.350 | 21.989 | 0.0005035 | 1.083 | 0 |
| Clover | 313 | auto_t5_guarded | 3072 | ok | 804.509 | 71.987 | 0.0007629 | 0.666 | 0 |
| Clover | 613 | auto_t3_controlled | 1400 | ok | 867.651 | 6.168 | 0.0004120 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 2048 | ok | 773.069 | 22.165 | 0.0006104 | 0.000 | 0 |
| Clover | 613 | auto_t3_controlled | 3072 | ok | 707.289 | 72.076 | 0.0007782 | 0.000 | 0 |
| Clover | 613 | auto_t5_guarded | 1400 | ok | 905.602 | 6.037 | 0.0003357 | 2.294 | 0 |
| Clover | 613 | auto_t5_guarded | 2048 | ok | 779.689 | 21.981 | 0.0005341 | 1.038 | 0 |
| Clover | 613 | auto_t5_guarded | 3072 | ok | 804.508 | 71.985 | 0.0007172 | 0.661 | 0 |
| rusticl | 313 | auto_t3_controlled | 1400 | ok | 875.253 | 6.066 | 0.0002975 | 0.000 | 0 |
| rusticl | 313 | auto_t3_controlled | 2048 | ok | 715.436 | 23.930 | 0.0005493 | 0.000 | 0 |
| rusticl | 313 | auto_t3_controlled | 3072 | ok | 658.772 | 77.874 | 0.0007477 | 0.000 | 0 |
| rusticl | 313 | auto_t5_guarded | 1400 | ok | 915.564 | 5.970 | 0.0003204 | 2.568 | 0 |
| rusticl | 313 | auto_t5_guarded | 2048 | ok | 718.700 | 23.827 | 0.0005035 | 0.939 | 0 |
| rusticl | 313 | auto_t5_guarded | 3072 | ok | 746.444 | 77.564 | 0.0007629 | 0.612 | 0 |
| rusticl | 613 | auto_t3_controlled | 1400 | ok | 874.135 | 6.135 | 0.0004120 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 2048 | ok | 714.160 | 23.990 | 0.0006104 | 0.000 | 0 |
| rusticl | 613 | auto_t3_controlled | 3072 | ok | 658.836 | 77.774 | 0.0007782 | 0.000 | 0 |
| rusticl | 613 | auto_t5_guarded | 1400 | ok | 913.282 | 5.991 | 0.0003357 | 2.749 | 0 |
| rusticl | 613 | auto_t5_guarded | 2048 | ok | 718.915 | 23.842 | 0.0005341 | 0.954 | 0 |
| rusticl | 613 | auto_t5_guarded | 3072 | ok | 746.121 | 77.641 | 0.0007172 | 0.612 | 0 |

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

