# T4 Week 5 Block 2 - Controlled Integration Report

- Date: 2026-02-07T23:30:25.966135+00:00
- Policy: `t4-approx-gating-block3-2026-02-07`
- Sizes: [512, 1024, 1400]
- Families: ['dense_random', 'compressible_lowrank']
- Sessions=8 | Seed=42 | Error budget=0.005

## Aggregate Metrics

- Contract compliance rate: 1.000
- Post-fallback violation rate: 0.000
- Fallback rate: 0.000
- Policy exact-route rate: 0.500
- Compressible speedup vs exact: 2.852x

## Guardrail Evaluation

| Guardrail | Observed | Threshold | Comparator | Pass |
| --- | ---: | ---: | --- | --- |
| post_fallback_violation_rate | 0.000000 | 0.010000 | <= | True |
| contract_compliance_rate | 1.000000 | 0.990000 | >= | True |
| fallback_rate | 0.000000 | 0.100000 | <= | True |
| compressible_speedup_vs_exact_mean | 2.851803 | 2.000000 | >= | True |

- Disable signal: False
- Fallback action: `continue_controlled_integration_ready_for_next_gate`

## Decision

- Decision: `promote`
- Rationale: Controlled rerun passed all guardrails with deterministic behavior; policy remains valid for controlled progression.

