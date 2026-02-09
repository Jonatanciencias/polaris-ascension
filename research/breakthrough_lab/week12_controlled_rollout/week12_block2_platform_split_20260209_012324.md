# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-09T01:23:24.944407+00:00
- Seeds: [212, 312]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 3

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 212 | 3 | 3 | 0 |
| Clover | 312 | 3 | 3 | 0 |
| rusticl | 212 | 3 | 3 | 0 |
| rusticl | 312 | 3 | 3 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 212 | auto_t3_controlled | 1400 | ok | 868.456 | 6.197 | 0.0003052 | 0.000 | 0 |
| Clover | 212 | auto_t3_controlled | 2048 | ok | 772.966 | 22.181 | 0.0005035 | 0.000 | 0 |
| Clover | 212 | auto_t5_guarded | 1400 | ok | 910.054 | 5.981 | 0.0003281 | 1.286 | 0 |
| Clover | 212 | auto_t5_guarded | 2048 | ok | 778.034 | 21.972 | 0.0005035 | 0.579 | 0 |
| Clover | 312 | auto_t3_controlled | 1400 | ok | 876.760 | 6.153 | 0.0003204 | 0.000 | 0 |
| Clover | 312 | auto_t3_controlled | 2048 | ok | 771.996 | 22.122 | 0.0005035 | 0.000 | 0 |
| Clover | 312 | auto_t5_guarded | 1400 | ok | 907.761 | 5.992 | 0.0003204 | 1.310 | 0 |
| Clover | 312 | auto_t5_guarded | 2048 | ok | 778.111 | 21.977 | 0.0005035 | 0.600 | 0 |
| rusticl | 212 | auto_t3_controlled | 1400 | ok | 866.380 | 6.197 | 0.0003052 | 0.000 | 0 |
| rusticl | 212 | auto_t3_controlled | 2048 | ok | 712.517 | 24.023 | 0.0005035 | 0.000 | 0 |
| rusticl | 212 | auto_t5_guarded | 1400 | ok | 920.423 | 5.938 | 0.0003281 | 1.466 | 0 |
| rusticl | 212 | auto_t5_guarded | 2048 | ok | 718.484 | 23.816 | 0.0005035 | 0.551 | 0 |
| rusticl | 312 | auto_t3_controlled | 1400 | ok | 872.627 | 6.172 | 0.0003204 | 0.000 | 0 |
| rusticl | 312 | auto_t3_controlled | 2048 | ok | 713.640 | 23.958 | 0.0005035 | 0.000 | 0 |
| rusticl | 312 | auto_t5_guarded | 1400 | ok | 919.291 | 5.929 | 0.0003204 | 1.428 | 0 |
| rusticl | 312 | auto_t5_guarded | 2048 | ok | 719.135 | 23.750 | 0.0005035 | 0.534 | 0 |

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

