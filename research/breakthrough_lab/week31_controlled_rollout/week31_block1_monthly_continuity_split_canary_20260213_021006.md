# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-13T02:10:06.663685+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 873.241 | 6.185 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 773.467 | 22.162 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 706.269 | 72.099 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 906.914 | 6.008 | 0.0003357 | 1.109 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.834 | 21.976 | 0.0005264 | 0.519 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 803.940 | 71.991 | 0.0007935 | 0.327 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 875.800 | 6.087 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.838 | 22.218 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 706.073 | 72.088 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 908.279 | 6.020 | 0.0003204 | 1.078 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 778.013 | 21.975 | 0.0005646 | 0.529 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 802.112 | 72.064 | 0.0007477 | 0.331 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 881.819 | 6.088 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 710.303 | 24.098 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 659.368 | 77.674 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 915.518 | 5.976 | 0.0003357 | 1.308 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 716.340 | 23.864 | 0.0005264 | 0.479 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 745.894 | 77.579 | 0.0007935 | 0.305 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 880.302 | 6.098 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 713.275 | 24.028 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.622 | 77.612 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 916.069 | 5.955 | 0.0003204 | 1.343 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 704.460 | 23.861 | 0.0005646 | 2.146 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 744.859 | 77.492 | 0.0007477 | 0.301 | 0 |

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

