# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T02:30:53.661602+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 873.558 | 6.129 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.369 | 22.161 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 706.978 | 72.084 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 905.813 | 5.998 | 0.0003357 | 1.361 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 778.721 | 21.973 | 0.0005264 | 0.522 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 802.909 | 71.955 | 0.0007935 | 0.338 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 876.779 | 6.157 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.689 | 22.215 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 708.680 | 72.103 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 909.272 | 5.983 | 0.0003204 | 1.096 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.810 | 21.998 | 0.0005646 | 0.529 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.978 | 71.916 | 0.0007477 | 0.334 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 872.094 | 6.195 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.055 | 24.097 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 660.895 | 77.739 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 920.503 | 5.944 | 0.0003357 | 1.362 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.314 | 23.816 | 0.0005264 | 0.476 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 745.184 | 77.532 | 0.0007935 | 0.305 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 885.281 | 6.072 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 715.563 | 23.941 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 661.187 | 77.574 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 921.008 | 5.928 | 0.0003204 | 1.284 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 719.289 | 23.746 | 0.0005646 | 0.478 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.302 | 77.493 | 0.0007477 | 0.310 | 0 |

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

