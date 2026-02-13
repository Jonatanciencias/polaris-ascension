# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T15:24:12.516841+00:00
- Seeds: [211, 509]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 2

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 211 | 2 | 2 | 0 |
| Clover | 509 | 2 | 2 | 0 |
| rusticl | 211 | 2 | 2 | 0 |
| rusticl | 509 | 2 | 2 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 211 | auto_t3_controlled | 1400 | ok | 868.976 | 6.197 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.085 | 22.229 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 705.633 | 72.129 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 907.518 | 5.980 | 0.0003357 | 1.103 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.783 | 22.009 | 0.0005264 | 0.532 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 802.561 | 71.985 | 0.0007935 | 0.334 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 877.225 | 6.093 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.212 | 22.228 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 707.555 | 72.173 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 906.593 | 5.995 | 0.0003204 | 1.110 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 773.786 | 21.972 | 0.0005646 | 0.531 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.968 | 71.971 | 0.0007477 | 0.347 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 870.339 | 6.179 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 711.941 | 24.065 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.928 | 77.871 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 919.066 | 5.945 | 0.0003357 | 1.241 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.221 | 23.807 | 0.0005264 | 0.480 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.233 | 77.586 | 0.0007935 | 0.303 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 885.932 | 6.070 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.364 | 24.033 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.994 | 77.675 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 920.744 | 5.934 | 0.0003204 | 1.309 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.445 | 23.792 | 0.0005646 | 0.481 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.677 | 77.459 | 0.0007477 | 0.303 | 0 |

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

