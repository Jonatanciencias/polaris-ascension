# Week 8 Block 6 - Platform Canary (Critical Sizes)

- Date: 2026-02-08T02:46:25.002510+00:00
- Sizes: [1400, 2048]
- Kernels: ['auto', 'auto_t3_controlled', 'auto_t5_guarded']

## Run Matrix

| Platform selector | Kernel | Size | Status | Peak mean GFLOPS | P95 ms | Max error |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Clover | auto | 1400 | ok | 897.811 | 6.139 | 0.0002975 |
| Clover | auto | 2048 | ok | 775.326 | 22.198 | 0.0004425 |
| Clover | auto_t3_controlled | 1400 | ok | 897.541 | 6.143 | 0.0003395 |
| Clover | auto_t3_controlled | 2048 | ok | 774.270 | 22.233 | 0.0005341 |
| Clover | auto_t5_guarded | 1400 | ok | 914.463 | 6.007 | 0.0003281 |
| Clover | auto_t5_guarded | 2048 | ok | 781.593 | 21.986 | 0.0005493 |
| rusticl | auto | 1400 | ok | 905.837 | 6.187 | 0.0002899 |
| rusticl | auto | 2048 | ok | 716.167 | 23.991 | 0.0005646 |
| rusticl | auto_t3_controlled | 1400 | ok | 908.153 | 6.158 | 0.0003967 |
| rusticl | auto_t3_controlled | 2048 | ok | 718.772 | 23.903 | 0.0005035 |
| rusticl | auto_t5_guarded | 1400 | ok | 925.907 | 5.929 | 0.0003357 |
| rusticl | auto_t5_guarded | 2048 | ok | 721.303 | 23.838 | 0.0006104 |

## Rusticl/Clover Peak Ratio

| Size | Kernel | Ratio |
| ---: | --- | ---: |
| 1400 | auto | 1.009 |
| 1400 | auto_t3_controlled | 1.012 |
| 1400 | auto_t5_guarded | 1.013 |
| 2048 | auto | 0.924 |
| 2048 | auto_t3_controlled | 0.928 |
| 2048 | auto_t5_guarded | 0.923 |

## Checks

| Check | Pass |
| --- | --- |
| clover_selection_all_runs | True |
| rusticl_selection_all_runs | True |
| correctness_bound_all_runs | True |
| rusticl_peak_ratio_min | True |
| t3_guardrails_all_platforms | True |
| t5_guardrails_all_platforms | True |

## Decision

- Decision: `promote`
- Rationale: Critical-size short canary validates both platforms with bounded correctness and guardrails.

