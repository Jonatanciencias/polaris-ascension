# T4 Week 4 Block 3 - Policy Gating Report

- Date: 2026-02-07T22:42:56.404813+00:00
- Families: ['dense_random', 'compressible_lowrank'] | Sizes: [512, 1024, 1400] | Sessions: 6
- Contract: error_budget=0.005, precheck_energy_threshold=0.95

## Summary

- Contract compliance: 1.000
- Post-fallback violation rate: 0.000
- Fallback rate: 0.000
- Policy exact-route rate: 0.500
- Approximate-attempt rate: 0.500
- Compressible speedup vs exact: 3.022x
- Stop rule triggered: False

## Family Metrics

| Family | Runs | Speedup vs Exact | Contract | Policy Exact Route | Approx Attempts | Fallback | Raw Error Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_random | 18 | 0.809x | 1.000 | 1.000 | 0.000 | 0.000 | 0.000000 |
| compressible_lowrank | 18 | 3.022x | 1.000 | 0.000 | 1.000 | 0.000 | 0.002493 |

## Decision

- Suggested decision: `promote`
- Promotion gate (scoped): True
- Rationale: Compressibility-gated policy is contract-safe with no post-fallback escapes, zero fallback triggers and strong speedup on eligible workloads.
