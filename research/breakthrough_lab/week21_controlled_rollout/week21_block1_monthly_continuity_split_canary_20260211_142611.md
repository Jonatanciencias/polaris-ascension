# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-11T14:26:11.325880+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 876.638 | 6.157 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.891 | 22.151 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 705.103 | 72.112 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 908.970 | 5.986 | 0.0003357 | 1.117 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 777.945 | 21.968 | 0.0005264 | 0.520 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 802.169 | 71.934 | 0.0007935 | 0.332 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 872.839 | 6.172 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.581 | 22.177 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 704.729 | 72.134 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 886.056 | 5.989 | 0.0003204 | 1.118 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.964 | 21.970 | 0.0005646 | 0.532 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 803.140 | 71.912 | 0.0007477 | 0.330 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 879.502 | 6.091 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 713.487 | 23.983 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.102 | 77.722 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 921.442 | 5.932 | 0.0003357 | 1.305 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.638 | 23.781 | 0.0005264 | 0.478 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.036 | 77.519 | 0.0007935 | 0.309 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 886.211 | 6.086 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 711.631 | 24.060 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 656.992 | 77.734 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 922.162 | 5.922 | 0.0003204 | 1.313 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.141 | 23.803 | 0.0005646 | 0.478 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 745.542 | 77.514 | 0.0007477 | 0.308 | 0 |

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

