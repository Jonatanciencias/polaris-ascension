# Week 15 Block 2 - Plugin Pilot

- Date: 2026-02-10T01:22:47.988219+00:00
- Plugin ID: `rx590_plugin_pilot_v1`
- Owner: `gpu-lab`

## Runs

| Kernel | Size | Avg GFLOPS | P95 ms | Max error |
| --- | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 874.822 | 6.107 | 0.0003204 |
| auto_t5_guarded | 1400 | 905.213 | 6.007 | 0.0003204 |
| auto_t3_controlled | 2048 | 773.430 | 22.105 | 0.0005035 |
| auto_t5_guarded | 2048 | 777.758 | 21.978 | 0.0004883 |

## Checks

| Check | Pass |
| --- | --- |
| template_exists | True |
| driver_smoke_good | False |
| all_runs_success | True |
| correctness_bound | True |
| t3_fallback_bound | True |
| t5_disable_zero | True |
| pre_gate_promote | True |
| post_gate_promote | True |

## Decision

- Decision: `iterate`
- Failed checks: ['driver_smoke_good']
- Rationale: Plugin pilot did not satisfy one or more mandatory contracts/guardrails.

