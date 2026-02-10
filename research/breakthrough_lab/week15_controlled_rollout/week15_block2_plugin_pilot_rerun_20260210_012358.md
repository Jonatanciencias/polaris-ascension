# Week 15 Block 2 - Plugin Pilot

- Date: 2026-02-10T01:23:58.382840+00:00
- Plugin ID: `rx590_plugin_pilot_v1`
- Owner: `gpu-lab`

## Runs

| Kernel | Size | Avg GFLOPS | P95 ms | Max error |
| --- | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 876.548 | 6.148 | 0.0003204 |
| auto_t5_guarded | 1400 | 904.223 | 6.028 | 0.0003204 |
| auto_t3_controlled | 2048 | 773.155 | 22.141 | 0.0005798 |
| auto_t5_guarded | 2048 | 777.484 | 21.976 | 0.0004883 |

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

