# Week 15 Block 2 - Plugin Pilot

- Date: 2026-02-11T01:11:29.695854+00:00
- Plugin ID: `rx590_dependent_project_week17_block2`
- Owner: `dependent-project-team-v2`

## Runs

| Kernel | Size | Avg GFLOPS | P95 ms | Max error |
| --- | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 875.245 | 6.101 | 0.0003052 |
| auto_t5_guarded | 1400 | 907.647 | 5.978 | 0.0002899 |
| auto_t3_controlled | 2048 | 772.774 | 22.139 | 0.0005188 |
| auto_t5_guarded | 2048 | 777.601 | 21.978 | 0.0004883 |
| auto_t3_controlled | 3072 | 709.998 | 72.064 | 0.0007629 |
| auto_t5_guarded | 3072 | 804.094 | 72.000 | 0.0007706 |

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

