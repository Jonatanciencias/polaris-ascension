# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T13:59:51.483439+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 877.445 | 6.119 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.854 | 22.190 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 707.034 | 72.063 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 906.369 | 6.018 | 0.0003357 | 1.175 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 778.062 | 22.012 | 0.0005264 | 0.529 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.086 | 71.954 | 0.0007935 | 0.332 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 879.384 | 6.051 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.166 | 22.165 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 705.710 | 72.048 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 905.139 | 5.999 | 0.0003204 | 1.077 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.039 | 21.983 | 0.0005646 | 0.510 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.265 | 71.943 | 0.0007477 | 0.330 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 873.080 | 6.146 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.820 | 24.068 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.376 | 77.681 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 921.437 | 5.923 | 0.0003357 | 1.408 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.210 | 23.807 | 0.0005264 | 0.474 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.201 | 77.494 | 0.0007935 | 0.306 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 878.745 | 6.145 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.382 | 24.057 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 659.589 | 77.668 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 823.200 | 5.961 | 0.0003204 | 1.204 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 719.224 | 23.772 | 0.0005646 | 0.475 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.109 | 77.526 | 0.0007477 | 0.305 | 0 |

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

