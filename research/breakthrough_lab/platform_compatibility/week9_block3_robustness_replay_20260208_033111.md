# Week 9 Block 3 - Robustness Replay (Seeds + Platform Split)

- Date: 2026-02-08T03:31:11.572231+00:00
- Seeds: [7, 42, 1337]
- Sizes: [1400, 2048]
- Kernels: ['auto_t3_controlled', 'auto_t5_guarded']
- Sessions: 1 | Iterations: 6

## Run Matrix

| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Clover | 7 | auto_t3_controlled | 1400 | ok | 873.901 | 6.147 | 0.0003510 | 0.000 | 0 |
| Clover | 7 | auto_t3_controlled | 2048 | ok | 773.697 | 22.159 | 0.0004730 | 0.000 | 0 |
| Clover | 7 | auto_t5_guarded | 1400 | ok | 906.699 | 6.029 | 0.0003052 | 2.146 | 0 |
| Clover | 7 | auto_t5_guarded | 2048 | ok | 778.896 | 22.007 | 0.0005646 | 1.010 | 0 |
| Clover | 42 | auto_t3_controlled | 1400 | ok | 875.910 | 6.066 | 0.0003357 | 0.000 | 0 |
| Clover | 42 | auto_t3_controlled | 2048 | ok | 771.995 | 22.214 | 0.0005188 | 0.000 | 0 |
| Clover | 42 | auto_t5_guarded | 1400 | ok | 911.405 | 5.990 | 0.0002975 | 2.140 | 0 |
| Clover | 42 | auto_t5_guarded | 2048 | ok | 779.207 | 21.978 | 0.0006409 | 0.945 | 0 |
| Clover | 1337 | auto_t3_controlled | 1400 | ok | 874.953 | 6.095 | 0.0002747 | 0.000 | 0 |
| Clover | 1337 | auto_t3_controlled | 2048 | ok | 772.373 | 22.186 | 0.0005035 | 0.000 | 0 |
| Clover | 1337 | auto_t5_guarded | 1400 | ok | 907.413 | 5.985 | 0.0003052 | 2.144 | 0 |
| Clover | 1337 | auto_t5_guarded | 2048 | ok | 778.847 | 21.971 | 0.0005341 | 0.971 | 0 |
| rusticl | 7 | auto_t3_controlled | 1400 | ok | 870.602 | 6.136 | 0.0003510 | 0.000 | 0 |
| rusticl | 7 | auto_t3_controlled | 2048 | ok | 712.837 | 24.015 | 0.0004730 | 0.000 | 0 |
| rusticl | 7 | auto_t5_guarded | 1400 | ok | 920.281 | 5.944 | 0.0003052 | 2.426 | 0 |
| rusticl | 7 | auto_t5_guarded | 2048 | ok | 718.242 | 23.782 | 0.0005646 | 0.903 | 0 |
| rusticl | 42 | auto_t3_controlled | 1400 | ok | 880.462 | 6.101 | 0.0003357 | 0.000 | 0 |
| rusticl | 42 | auto_t3_controlled | 2048 | ok | 713.557 | 24.045 | 0.0005188 | 0.000 | 0 |
| rusticl | 42 | auto_t5_guarded | 1400 | ok | 920.742 | 5.934 | 0.0002975 | 2.476 | 0 |
| rusticl | 42 | auto_t5_guarded | 2048 | ok | 719.160 | 23.828 | 0.0006409 | 1.174 | 0 |
| rusticl | 1337 | auto_t3_controlled | 1400 | ok | 874.476 | 6.138 | 0.0002747 | 0.000 | 0 |
| rusticl | 1337 | auto_t3_controlled | 2048 | ok | 712.823 | 24.044 | 0.0005035 | 0.000 | 0 |
| rusticl | 1337 | auto_t5_guarded | 1400 | ok | 919.069 | 5.927 | 0.0003052 | 2.440 | 0 |
| rusticl | 1337 | auto_t5_guarded | 2048 | ok | 693.932 | 23.860 | 0.0005341 | 0.855 | 0 |

## Rusticl vs Clover Ratio

| Seed | Kernel | Size | Ratio |
| ---: | --- | ---: | ---: |
| 7 | auto_t3_controlled | 1400 | 1.002 |
| 7 | auto_t5_guarded | 1400 | 1.014 |
| 7 | auto_t3_controlled | 2048 | 0.923 |
| 7 | auto_t5_guarded | 2048 | 0.925 |
| 42 | auto_t3_controlled | 1400 | 0.994 |
| 42 | auto_t5_guarded | 1400 | 1.009 |
| 42 | auto_t3_controlled | 2048 | 0.924 |
| 42 | auto_t5_guarded | 2048 | 0.922 |
| 1337 | auto_t3_controlled | 1400 | 0.993 |
| 1337 | auto_t5_guarded | 1400 | 1.010 |
| 1337 | auto_t3_controlled | 2048 | 0.923 |
| 1337 | auto_t5_guarded | 2048 | 0.921 |

## Regression vs Week9 Block2 (Clover Aggregate)

| Kernel | Size | Throughput delta % | P95 delta % |
| --- | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | -0.175 | -0.528 |
| auto_t3_controlled | 2048 | -0.023 | +0.080 |
| auto_t5_guarded | 1400 | +0.688 | +0.355 |
| auto_t5_guarded | 2048 | +0.036 | +0.094 |

## Checks

| Check | Pass |
| --- | --- |
| all_runs_success | True |
| platform_split_clover_and_rusticl | True |
| correctness_bound_all_runs | True |
| t3_guardrails_all_runs | True |
| t5_guardrails_all_runs | True |
| rusticl_peak_ratio_min | True |
| t5_no_regression_vs_block2_clover | True |

## Decision

- Decision: `promote`
- Rationale: Alternate-seed replay and short platform split passed with no post-hardening regressions.

