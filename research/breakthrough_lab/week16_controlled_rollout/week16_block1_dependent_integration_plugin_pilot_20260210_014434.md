# Week 15 Block 2 - Plugin Pilot

- Date: 2026-02-10T01:44:34.253443+00:00
- Plugin ID: `rx590_dependent_project_week16_block1`
- Owner: `dependent-project-team`

## Runs

| Kernel | Size | Avg GFLOPS | P95 ms | Max error |
| --- | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 867.920 | 6.117 | 0.0003052 |
| auto_t5_guarded | 1400 | 908.698 | 5.959 | 0.0003662 |
| auto_t3_controlled | 2048 | 771.359 | 22.245 | 0.0004578 |
| auto_t5_guarded | 2048 | 777.703 | 21.977 | 0.0005188 |
| auto_t3_controlled | 3072 | 708.093 | 72.095 | 0.0006714 |
| auto_t5_guarded | 3072 | 803.082 | 71.958 | 0.0009155 |

## Checks

| Check | Pass |
| --- | --- |
| template_exists | True |
| driver_smoke_good | True |
| all_runs_success | True |
| correctness_bound | True |
| t3_fallback_bound | True |
| t5_disable_zero | True |
| pre_gate_promote | True |
| post_gate_promote | True |

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Plugin pilot satisfies template contracts and performance/correctness guardrails.

