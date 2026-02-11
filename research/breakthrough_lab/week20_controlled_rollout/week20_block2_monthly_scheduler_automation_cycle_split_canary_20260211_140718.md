# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T14:07:18.624764+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 880.260 | 6.145 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 771.589 | 22.250 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 702.745 | 72.095 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 905.342 | 6.008 | 0.0003357 | 1.089 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.836 | 22.020 | 0.0005264 | 0.512 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.167 | 71.950 | 0.0007935 | 0.333 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 873.077 | 6.147 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.741 | 22.205 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 704.307 | 72.069 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 910.202 | 6.003 | 0.0003204 | 1.074 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.841 | 21.979 | 0.0005646 | 0.525 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.227 | 71.998 | 0.0007477 | 0.329 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 881.814 | 6.119 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.449 | 24.045 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 655.236 | 77.692 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 920.648 | 5.934 | 0.0003357 | 1.326 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.666 | 23.761 | 0.0005264 | 0.483 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.173 | 77.524 | 0.0007935 | 0.309 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 878.211 | 6.099 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.475 | 24.042 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 654.393 | 77.819 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 920.444 | 5.921 | 0.0003204 | 1.314 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 719.193 | 23.807 | 0.0005646 | 0.473 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 745.590 | 77.565 | 0.0007477 | 0.311 | 0 |

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

