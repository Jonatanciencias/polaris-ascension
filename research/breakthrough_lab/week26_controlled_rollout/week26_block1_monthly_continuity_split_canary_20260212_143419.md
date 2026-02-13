# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T14:34:19.200339+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 874.058 | 6.162 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.147 | 22.192 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 708.332 | 72.098 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 907.761 | 5.970 | 0.0003357 | 1.106 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.749 | 21.966 | 0.0005264 | 0.517 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.114 | 71.972 | 0.0007935 | 0.334 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 875.647 | 6.138 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.027 | 22.226 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 704.902 | 72.030 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 908.579 | 6.010 | 0.0003204 | 1.188 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.948 | 21.979 | 0.0005646 | 0.523 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.333 | 71.983 | 0.0007477 | 0.333 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 885.102 | 6.088 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.911 | 24.025 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.428 | 77.629 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 920.853 | 5.935 | 0.0003357 | 1.340 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.318 | 23.778 | 0.0005264 | 0.471 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.487 | 77.537 | 0.0007935 | 0.312 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 875.295 | 6.156 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 711.576 | 24.026 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.493 | 77.682 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 920.384 | 5.943 | 0.0003204 | 1.307 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.910 | 23.759 | 0.0005646 | 0.481 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.181 | 77.511 | 0.0007477 | 0.320 | 0 |

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

