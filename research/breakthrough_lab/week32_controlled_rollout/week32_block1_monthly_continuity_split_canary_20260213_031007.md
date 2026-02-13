# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T03:10:07.472399+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 884.531 | 6.088 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.317 | 22.197 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 707.848 | 72.077 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 911.832 | 5.963 | 0.0003357 | 1.115 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.718 | 21.998 | 0.0005264 | 0.523 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.097 | 71.986 | 0.0007935 | 0.332 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 876.407 | 6.158 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 773.717 | 22.155 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 707.473 | 72.147 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 910.144 | 6.007 | 0.0003204 | 1.117 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.676 | 21.987 | 0.0005646 | 0.517 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.092 | 72.005 | 0.0007477 | 0.336 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 877.645 | 6.138 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 711.685 | 24.056 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.991 | 77.659 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 913.863 | 5.973 | 0.0003357 | 1.242 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 708.746 | 23.884 | 0.0005264 | 0.474 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 744.936 | 77.627 | 0.0007935 | 0.308 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 883.296 | 6.065 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 713.556 | 24.021 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 657.737 | 77.631 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 915.162 | 5.980 | 0.0003204 | 1.330 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 716.522 | 23.872 | 0.0005646 | 0.477 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.096 | 77.560 | 0.0007477 | 0.306 | 0 |

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

