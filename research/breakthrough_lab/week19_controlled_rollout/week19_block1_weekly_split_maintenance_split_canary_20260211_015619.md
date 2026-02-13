# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T01:56:19.092378+00:00
- Seeds: [191, 419]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 2

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 191 | 2 | 2 | 0 |
| Clover | 419 | 2 | 2 | 0 |
| rusticl | 191 | 2 | 2 | 0 |
| rusticl | 419 | 2 | 2 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 191 | auto_t3_controlled | 1400 | ok | 873.974 | 6.155 | 0.0003815 | 0.000 | 0 |
| Clover | 191 | auto_t3_controlled | 2048 | ok | 774.256 | 22.089 | 0.0004883 | 0.000 | 0 |
| Clover | 191 | auto_t3_controlled | 3072 | ok | 708.586 | 72.152 | 0.0008240 | 0.000 | 0 |
| Clover | 191 | auto_t5_guarded | 1400 | ok | 908.623 | 6.011 | 0.0003510 | 1.114 | 0 |
| Clover | 191 | auto_t5_guarded | 2048 | ok | 777.770 | 21.981 | 0.0004883 | 0.539 | 0 |
| Clover | 191 | auto_t5_guarded | 3072 | ok | 803.380 | 71.991 | 0.0007172 | 0.333 | 0 |
| Clover | 419 | auto_t3_controlled | 1400 | ok | 877.479 | 6.098 | 0.0002747 | 0.000 | 0 |
| Clover | 419 | auto_t3_controlled | 2048 | ok | 773.187 | 22.133 | 0.0006256 | 0.000 | 0 |
| Clover | 419 | auto_t3_controlled | 3072 | ok | 708.264 | 72.117 | 0.0007019 | 0.000 | 0 |
| Clover | 419 | auto_t5_guarded | 1400 | ok | 907.209 | 6.015 | 0.0002861 | 1.096 | 0 |
| Clover | 419 | auto_t5_guarded | 2048 | ok | 777.696 | 21.994 | 0.0004883 | 0.511 | 0 |
| Clover | 419 | auto_t5_guarded | 3072 | ok | 804.035 | 71.989 | 0.0008087 | 0.332 | 0 |
| rusticl | 191 | auto_t3_controlled | 1400 | ok | 878.831 | 6.141 | 0.0003815 | 0.000 | 0 |
| rusticl | 191 | auto_t3_controlled | 2048 | ok | 714.061 | 23.920 | 0.0004883 | 0.000 | 0 |
| rusticl | 191 | auto_t3_controlled | 3072 | ok | 659.973 | 77.720 | 0.0008240 | 0.000 | 0 |
| rusticl | 191 | auto_t5_guarded | 1400 | ok | 921.028 | 5.928 | 0.0003510 | 1.261 | 0 |
| rusticl | 191 | auto_t5_guarded | 2048 | ok | 719.138 | 23.766 | 0.0004883 | 0.473 | 0 |
| rusticl | 191 | auto_t5_guarded | 3072 | ok | 745.988 | 77.544 | 0.0007172 | 0.304 | 0 |
| rusticl | 419 | auto_t3_controlled | 1400 | ok | 886.474 | 6.047 | 0.0002747 | 0.000 | 0 |
| rusticl | 419 | auto_t3_controlled | 2048 | ok | 713.009 | 24.022 | 0.0006256 | 0.000 | 0 |
| rusticl | 419 | auto_t3_controlled | 3072 | ok | 660.054 | 77.617 | 0.0007019 | 0.000 | 0 |
| rusticl | 419 | auto_t5_guarded | 1400 | ok | 920.796 | 5.932 | 0.0002861 | 1.307 | 0 |
| rusticl | 419 | auto_t5_guarded | 2048 | ok | 717.899 | 23.841 | 0.0004883 | 0.582 | 0 |
| rusticl | 419 | auto_t5_guarded | 3072 | ok | 746.293 | 77.542 | 0.0008087 | 0.308 | 0 |

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

