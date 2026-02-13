# Week 9 Block 4 - Stress Replay (Queue Pulses + Platform Split)

- Date: 2026-02-09T01:38:14.452407+00:00
- Seeds: [412, 512]
- Sizes: [1400, 2048, 3072]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 8
- Queue pressure pulses per platform/seed: 3

## Pressure Summary

| Platform | Seed | Requested | Completed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Clover | 412 | 3 | 3 | 0 |
| Clover | 512 | 3 | 3 | 0 |
| rusticl | 412 | 3 | 3 | 0 |
| rusticl | 512 | 3 | 3 | 0 |

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 412 | auto_t3_controlled | 1400 | ok | 869.978 | 6.143 | 0.0003662 | 0.000 | 0 |
| Clover | 412 | auto_t3_controlled | 2048 | ok | 771.716 | 22.212 | 0.0005493 | 0.000 | 0 |
| Clover | 412 | auto_t3_controlled | 3072 | ok | 706.929 | 72.091 | 0.0007782 | 0.000 | 0 |
| Clover | 412 | auto_t5_guarded | 1400 | ok | 911.004 | 5.955 | 0.0002899 | 1.271 | 0 |
| Clover | 412 | auto_t5_guarded | 2048 | ok | 778.028 | 21.989 | 0.0004578 | 0.589 | 0 |
| Clover | 412 | auto_t5_guarded | 3072 | ok | 801.069 | 72.057 | 0.0007172 | 0.363 | 0 |
| Clover | 512 | auto_t3_controlled | 1400 | ok | 870.293 | 6.206 | 0.0004578 | 0.000 | 0 |
| Clover | 512 | auto_t3_controlled | 2048 | ok | 772.294 | 22.169 | 0.0005646 | 0.000 | 0 |
| Clover | 512 | auto_t3_controlled | 3072 | ok | 707.562 | 72.071 | 0.0007172 | 0.000 | 0 |
| Clover | 512 | auto_t5_guarded | 1400 | ok | 908.844 | 5.986 | 0.0003357 | 1.370 | 0 |
| Clover | 512 | auto_t5_guarded | 2048 | ok | 777.575 | 22.011 | 0.0004730 | 0.602 | 0 |
| Clover | 512 | auto_t5_guarded | 3072 | ok | 804.294 | 71.969 | 0.0008392 | 0.363 | 0 |
| rusticl | 412 | auto_t3_controlled | 1400 | ok | 876.781 | 6.140 | 0.0003662 | 0.000 | 0 |
| rusticl | 412 | auto_t3_controlled | 2048 | ok | 714.579 | 23.967 | 0.0005493 | 0.000 | 0 |
| rusticl | 412 | auto_t3_controlled | 3072 | ok | 659.846 | 77.621 | 0.0007782 | 0.000 | 0 |
| rusticl | 412 | auto_t5_guarded | 1400 | ok | 920.089 | 5.942 | 0.0002899 | 1.437 | 0 |
| rusticl | 412 | auto_t5_guarded | 2048 | ok | 718.614 | 23.767 | 0.0004578 | 0.550 | 0 |
| rusticl | 412 | auto_t5_guarded | 3072 | ok | 746.351 | 77.483 | 0.0007172 | 0.341 | 0 |
| rusticl | 512 | auto_t3_controlled | 1400 | ok | 874.756 | 6.158 | 0.0004578 | 0.000 | 0 |
| rusticl | 512 | auto_t3_controlled | 2048 | ok | 713.537 | 24.026 | 0.0005646 | 0.000 | 0 |
| rusticl | 512 | auto_t3_controlled | 3072 | ok | 657.565 | 77.796 | 0.0007172 | 0.000 | 0 |
| rusticl | 512 | auto_t5_guarded | 1400 | ok | 920.337 | 5.929 | 0.0003357 | 1.417 | 0 |
| rusticl | 512 | auto_t5_guarded | 2048 | ok | 719.082 | 23.825 | 0.0004730 | 0.544 | 0 |
| rusticl | 512 | auto_t5_guarded | 3072 | ok | 746.243 | 77.509 | 0.0008392 | 0.341 | 0 |

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

