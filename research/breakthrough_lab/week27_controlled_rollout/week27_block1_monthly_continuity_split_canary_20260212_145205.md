# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-12T14:52:05.523867+00:00
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
| Clover | 211 | auto_t3_controlled | 1400 | ok | 879.818 | 6.058 | 0.0003281 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 2048 | ok | 772.707 | 22.167 | 0.0005035 | 0.000 | 0 |
| Clover | 211 | auto_t3_controlled | 3072 | ok | 707.577 | 72.090 | 0.0007477 | 0.000 | 0 |
| Clover | 211 | auto_t5_guarded | 1400 | ok | 908.626 | 5.983 | 0.0003357 | 1.080 | 0 |
| Clover | 211 | auto_t5_guarded | 2048 | ok | 779.628 | 21.959 | 0.0005264 | 0.514 | 0 |
| Clover | 211 | auto_t5_guarded | 3072 | ok | 804.386 | 71.942 | 0.0007935 | 0.331 | 0 |
| Clover | 509 | auto_t3_controlled | 1400 | ok | 880.966 | 6.070 | 0.0003052 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 2048 | ok | 771.874 | 22.186 | 0.0004578 | 0.000 | 0 |
| Clover | 509 | auto_t3_controlled | 3072 | ok | 705.380 | 72.055 | 0.0007935 | 0.000 | 0 |
| Clover | 509 | auto_t5_guarded | 1400 | ok | 905.757 | 5.981 | 0.0003204 | 1.225 | 0 |
| Clover | 509 | auto_t5_guarded | 2048 | ok | 777.830 | 21.980 | 0.0005646 | 0.517 | 0 |
| Clover | 509 | auto_t5_guarded | 3072 | ok | 804.236 | 71.940 | 0.0007477 | 0.329 | 0 |
| rusticl | 211 | auto_t3_controlled | 1400 | ok | 879.744 | 6.129 | 0.0003281 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 2048 | ok | 712.019 | 24.045 | 0.0005035 | 0.000 | 0 |
| rusticl | 211 | auto_t3_controlled | 3072 | ok | 657.980 | 77.979 | 0.0007477 | 0.000 | 0 |
| rusticl | 211 | auto_t5_guarded | 1400 | ok | 918.675 | 5.951 | 0.0003357 | 1.300 | 0 |
| rusticl | 211 | auto_t5_guarded | 2048 | ok | 718.953 | 23.779 | 0.0005264 | 0.465 | 0 |
| rusticl | 211 | auto_t5_guarded | 3072 | ok | 746.171 | 77.457 | 0.0007935 | 0.304 | 0 |
| rusticl | 509 | auto_t3_controlled | 1400 | ok | 875.954 | 6.121 | 0.0003052 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 2048 | ok | 712.495 | 24.063 | 0.0004578 | 0.000 | 0 |
| rusticl | 509 | auto_t3_controlled | 3072 | ok | 658.209 | 77.933 | 0.0007935 | 0.000 | 0 |
| rusticl | 509 | auto_t5_guarded | 1400 | ok | 920.116 | 5.930 | 0.0003204 | 1.293 | 0 |
| rusticl | 509 | auto_t5_guarded | 2048 | ok | 718.976 | 23.815 | 0.0005646 | 0.474 | 0 |
| rusticl | 509 | auto_t5_guarded | 3072 | ok | 746.159 | 77.376 | 0.0007477 | 0.307 | 0 |

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

