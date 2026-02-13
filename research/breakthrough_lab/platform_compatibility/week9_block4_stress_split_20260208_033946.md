# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-08T03:39:46.858207+00:00
- Seeds: [11, 77]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 6
- Queue pressure pulses per platform/seed: 2

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 11 | 2 | 2 | 0 |
| Clover | 77 | 2 | 2 | 0 |
| rusticl | 11 | 2 | 2 | 0 |
| rusticl | 77 | 2 | 2 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 11 | auto_t3_controlled | 1400 | ok | 870.862 | 6.193 | 0.0002899 | 0.000 | 0 |
| Clover | 11 | auto_t3_controlled | 2048 | ok | 772.593 | 22.184 | 0.0005646 | 0.000 | 0 |
| Clover | 11 | auto_t5_guarded | 1400 | ok | 910.275 | 6.014 | 0.0002899 | 2.264 | 0 |
| Clover | 11 | auto_t5_guarded | 2048 | ok | 778.788 | 21.982 | 0.0005035 | 0.948 | 0 |
| Clover | 77 | auto_t3_controlled | 1400 | ok | 878.615 | 6.122 | 0.0004349 | 0.000 | 0 |
| Clover | 77 | auto_t3_controlled | 2048 | ok | 772.638 | 22.203 | 0.0004730 | 0.000 | 0 |
| Clover | 77 | auto_t5_guarded | 1400 | ok | 892.551 | 6.015 | 0.0003510 | 2.583 | 0 |
| Clover | 77 | auto_t5_guarded | 2048 | ok | 778.395 | 21.997 | 0.0005493 | 0.970 | 0 |
| rusticl | 11 | auto_t3_controlled | 1400 | ok | 878.567 | 6.062 | 0.0002899 | 0.000 | 0 |
| rusticl | 11 | auto_t3_controlled | 2048 | ok | 713.307 | 24.013 | 0.0005646 | 0.000 | 0 |
| rusticl | 11 | auto_t5_guarded | 1400 | ok | 917.797 | 5.948 | 0.0002899 | 2.398 | 0 |
| rusticl | 11 | auto_t5_guarded | 2048 | ok | 719.094 | 23.759 | 0.0005035 | 0.877 | 0 |
| rusticl | 77 | auto_t3_controlled | 1400 | ok | 883.485 | 6.073 | 0.0004349 | 0.000 | 0 |
| rusticl | 77 | auto_t3_controlled | 2048 | ok | 714.577 | 24.001 | 0.0004730 | 0.000 | 0 |
| rusticl | 77 | auto_t5_guarded | 1400 | ok | 919.133 | 5.935 | 0.0003510 | 2.394 | 0 |
| rusticl | 77 | auto_t5_guarded | 2048 | ok | 720.130 | 23.747 | 0.0005493 | 0.867 | 0 |

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

