# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T00:38:14.796200+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 873.173 | 6.157 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 771.834 | 22.223 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 704.822 | 72.105 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 906.496 | 6.011 | 0.0003357 | 1.109 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.809 | 21.977 | 0.0005264 | 0.532 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 803.634 | 71.923 | 0.0007935 | 0.333 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 877.970 | 6.076 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.547 | 22.224 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 706.949 | 72.105 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 911.335 | 5.977 | 0.0003204 | 1.106 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 778.105 | 21.977 | 0.0005646 | 0.533 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.686 | 71.936 | 0.0007477 | 0.332 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 876.996 | 6.140 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 714.378 | 24.001 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 658.828 | 77.756 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 921.404 | 5.939 | 0.0003357 | 1.272 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 719.466 | 23.739 | 0.0005264 | 0.481 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 742.599 | 77.560 | 0.0007935 | 0.313 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 874.562 | 6.163 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.160 | 24.040 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.352 | 77.665 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 920.963 | 5.925 | 0.0003204 | 1.302 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 719.800 | 23.776 | 0.0005646 | 0.487 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.207 | 77.468 | 0.0007477 | 0.315 | 0 |

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

