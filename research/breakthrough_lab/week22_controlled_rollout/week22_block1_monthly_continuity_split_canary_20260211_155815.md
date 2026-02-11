# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T15:58:15.927121+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 876.836 | 6.110 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.383 | 22.203 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 705.685 | 72.040 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 906.866 | 6.014 | 0.0003357 | 1.088 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.768 | 21.983 | 0.0005264 | 0.511 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.230 | 71.968 | 0.0007935 | 0.331 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 872.384 | 6.205 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.082 | 22.227 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 707.279 | 72.073 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 903.574 | 6.004 | 0.0003204 | 1.085 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 778.201 | 21.971 | 0.0005646 | 0.524 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.250 | 71.965 | 0.0007477 | 0.330 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 875.076 | 6.137 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 711.924 | 24.056 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.728 | 77.788 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 919.094 | 5.945 | 0.0003357 | 1.324 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 717.993 | 23.764 | 0.0005264 | 0.480 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.381 | 77.518 | 0.0007935 | 0.306 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 875.453 | 6.169 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.686 | 24.014 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 656.853 | 77.753 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 919.309 | 5.934 | 0.0003204 | 1.318 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.744 | 23.789 | 0.0005646 | 0.477 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.162 | 77.440 | 0.0007477 | 0.308 | 0 |

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

