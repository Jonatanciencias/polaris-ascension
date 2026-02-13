# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T03:57:36.288747+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 879.166 | 6.158 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 773.153 | 22.134 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 707.362 | 72.006 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 910.047 | 5.926 | 0.0003357 | 1.109 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.976 | 22.014 | 0.0005264 | 0.523 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 803.976 | 71.995 | 0.0007935 | 0.350 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 878.032 | 6.149 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.239 | 22.178 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 707.481 | 72.130 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 907.760 | 5.994 | 0.0003204 | 1.094 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.803 | 21.991 | 0.0005646 | 0.507 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.613 | 72.016 | 0.0007477 | 0.332 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 867.916 | 6.203 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.378 | 24.031 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.234 | 77.684 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 915.010 | 5.966 | 0.0003357 | 1.292 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 717.254 | 23.866 | 0.0005264 | 0.486 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.003 | 77.608 | 0.0007935 | 0.304 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 886.990 | 6.056 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 710.835 | 24.116 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 659.134 | 77.666 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 915.974 | 5.974 | 0.0003204 | 1.278 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 717.018 | 23.836 | 0.0005646 | 0.481 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.122 | 77.590 | 0.0007477 | 0.321 | 0 |

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

