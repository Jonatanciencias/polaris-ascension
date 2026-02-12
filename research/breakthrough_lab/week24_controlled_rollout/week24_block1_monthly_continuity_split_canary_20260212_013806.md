# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T01:38:06.116925+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 874.159 | 6.191 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 771.726 | 22.213 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 706.324 | 71.991 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 906.471 | 6.033 | 0.0003357 | 1.127 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.744 | 21.992 | 0.0005264 | 0.525 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.095 | 71.967 | 0.0007935 | 0.336 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 873.965 | 6.148 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 772.737 | 22.142 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 707.300 | 72.049 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 911.136 | 5.996 | 0.0003204 | 1.129 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 769.373 | 21.979 | 0.0005646 | 0.536 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 802.659 | 71.942 | 0.0007477 | 0.336 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 871.465 | 6.171 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 711.567 | 24.103 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.200 | 77.845 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 920.252 | 5.941 | 0.0003357 | 1.327 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 719.407 | 23.795 | 0.0005264 | 0.492 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 744.573 | 77.542 | 0.0007935 | 0.316 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 870.157 | 6.175 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 713.811 | 24.007 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 656.902 | 77.721 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 918.400 | 5.937 | 0.0003204 | 1.345 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.356 | 23.784 | 0.0005646 | 0.482 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 744.989 | 77.471 | 0.0007477 | 0.314 | 0 |

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

