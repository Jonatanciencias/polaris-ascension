# T4 Week 8 Block 4 - Mixed Policy Refinement Report

- Date: 2026-02-08T02:44:14.978432+00:00
- Baseline policy: `t4-approx-gating-block3-2026-02-07`
- Candidate policy: `t4-approx-gating-week8-block4-2026-02-08`
- Families: ['dense_random', 'compressible_lowrank'] | Sizes: [512, 768, 1024]
- Sessions=6 | Seed=42 | Noise=0.02

## Baseline vs Candidate

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| contract_compliance_rate | 1.000000 | 1.000000 |
| post_fallback_violation_rate | 0.000000 | 0.000000 |
| fallback_rate | 0.194444 | 0.000000 |
| policy_exact_route_rate | 0.500000 | 0.500000 |
| approximate_attempt_rate | 0.500000 | 0.500000 |
| compressible_speedup_vs_exact_mean | 1.364489 | 1.982434 |
| delta_vs_exact_percent | 13.191789 | 39.733669 |
| fallback_reduction_abs | - | 0.194444 |

## Guardrail Evaluation

| Guardrail | Observed | Threshold | Comparator | Pass |
| --- | ---: | ---: | --- | --- |
| candidate_contract_compliance | 1.000000 | 0.990000 | >= | True |
| candidate_post_fallback_violation_rate | 0.000000 | 0.000000 | <= | True |
| candidate_fallback_rate | 0.000000 | 0.100000 | <= | True |
| candidate_compressible_speedup | 1.982434 | 1.500000 | >= | True |
| fallback_reduction_vs_baseline | 0.194444 | 0.100000 | >= | True |
| policy_exact_route_cap | 0.500000 | 0.800000 | <= | True |

## Decision

- Decision: `promote`
- Rationale: Fallback reduction and safety/performance guardrails passed on mixed workload campaign.

