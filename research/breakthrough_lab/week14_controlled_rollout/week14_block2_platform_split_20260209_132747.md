# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-09T13:27:47.669376+00:00
- Seeds: [1212, 1312]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 4

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 1212 | 4 | 4 | 0 |
| Clover | 1312 | 4 | 4 | 0 |
| rusticl | 1212 | 4 | 4 | 0 |
| rusticl | 1312 | 4 | 4 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 1212 | auto_t3_controlled | 1400 | ok | 881.964 | 6.106 | 0.0003510 | 0.000 | 0 |
| Clover | 1212 | auto_t3_controlled | 2048 | ok | 776.644 | 22.046 | 0.0005569 | 0.000 | 0 |
| Clover | 1212 | auto_t3_controlled | 3072 | ok | 708.615 | 71.952 | 0.0007935 | 0.000 | 0 |
| Clover | 1212 | auto_t5_guarded | 1400 | ok | 914.628 | 5.979 | 0.0003052 | 1.245 | 0 |
| Clover | 1212 | auto_t5_guarded | 2048 | ok | 778.682 | 21.939 | 0.0005798 | 0.557 | 0 |
| Clover | 1212 | auto_t5_guarded | 3072 | ok | 803.627 | 71.884 | 0.0007935 | 0.346 | 0 |
| Clover | 1312 | auto_t3_controlled | 1400 | ok | 865.607 | 6.234 | 0.0003052 | 0.000 | 0 |
| Clover | 1312 | auto_t3_controlled | 2048 | ok | 769.360 | 22.285 | 0.0005341 | 0.000 | 0 |
| Clover | 1312 | auto_t3_controlled | 3072 | ok | 703.246 | 72.274 | 0.0008087 | 0.000 | 0 |
| Clover | 1312 | auto_t5_guarded | 1400 | ok | 905.511 | 6.018 | 0.0003204 | 1.224 | 0 |
| Clover | 1312 | auto_t5_guarded | 2048 | ok | 776.977 | 22.001 | 0.0005493 | 0.559 | 0 |
| Clover | 1312 | auto_t5_guarded | 3072 | ok | 803.476 | 72.029 | 0.0007935 | 0.352 | 0 |
| rusticl | 1212 | auto_t3_controlled | 1400 | ok | 868.833 | 6.222 | 0.0003510 | 0.000 | 0 |
| rusticl | 1212 | auto_t3_controlled | 2048 | ok | 710.230 | 24.102 | 0.0005569 | 0.000 | 0 |
| rusticl | 1212 | auto_t3_controlled | 3072 | ok | 655.961 | 78.007 | 0.0007935 | 0.000 | 0 |
| rusticl | 1212 | auto_t5_guarded | 1400 | ok | 917.821 | 5.960 | 0.0003052 | 1.419 | 0 |
| rusticl | 1212 | auto_t5_guarded | 2048 | ok | 717.119 | 23.812 | 0.0005798 | 0.519 | 0 |
| rusticl | 1212 | auto_t5_guarded | 3072 | ok | 745.421 | 77.561 | 0.0007935 | 0.333 | 0 |
| rusticl | 1312 | auto_t3_controlled | 1400 | ok | 875.959 | 6.155 | 0.0003052 | 0.000 | 0 |
| rusticl | 1312 | auto_t3_controlled | 2048 | ok | 711.902 | 24.091 | 0.0005341 | 0.000 | 0 |
| rusticl | 1312 | auto_t3_controlled | 3072 | ok | 672.596 | 76.009 | 0.0008087 | 0.000 | 0 |
| rusticl | 1312 | auto_t5_guarded | 1400 | ok | 934.307 | 5.837 | 0.0003204 | 1.502 | 0 |
| rusticl | 1312 | auto_t5_guarded | 2048 | ok | 727.331 | 23.501 | 0.0005493 | 0.522 | 0 |
| rusticl | 1312 | auto_t5_guarded | 3072 | ok | 752.121 | 76.805 | 0.0007935 | 0.324 | 0 |

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

