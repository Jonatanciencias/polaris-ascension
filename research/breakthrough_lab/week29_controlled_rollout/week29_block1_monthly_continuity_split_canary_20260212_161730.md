# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T16:17:30.830629+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 873.985 | 6.116 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.397 | 22.189 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 706.096 | 72.111 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 908.767 | 5.977 | 0.0003357 | 1.119 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.687 | 21.975 | 0.0005264 | 0.513 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 803.984 | 71.951 | 0.0007935 | 0.331 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 877.631 | 6.129 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.082 | 22.170 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 704.882 | 72.072 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 902.917 | 6.030 | 0.0003204 | 1.085 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 778.144 | 21.965 | 0.0005646 | 0.505 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.390 | 71.967 | 0.0007477 | 0.334 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 875.157 | 6.155 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 711.702 | 24.083 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.741 | 77.639 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 920.262 | 5.926 | 0.0003357 | 1.220 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 719.175 | 23.794 | 0.0005264 | 0.472 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 744.654 | 77.439 | 0.0007935 | 0.309 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 875.960 | 6.139 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.767 | 24.022 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.419 | 77.730 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 919.598 | 5.933 | 0.0003204 | 1.255 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.685 | 23.767 | 0.0005646 | 0.466 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 744.962 | 77.510 | 0.0007477 | 0.311 | 0 |

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

