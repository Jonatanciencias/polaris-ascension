# T5 Week 4 Block 4 - Shadow Canary Integration Report

- Date: 2026-02-07T22:29:47.505408+00:00
- Policy: `t5-abft-detect-only-block3-2026-02-07`
- Sizes: [1400, 2048]
- Sessions=12 | Iterations=24 | Warmup=2
- Shadow sampling period: `8` (periodic_8)

## Canary Metrics

- Effective overhead: 1.284%
- False positive rate: 0.000
- Critical recall: 1.000
- Uniform recall: 0.972
- Critical misses: 0
- Max correctness error: 0.0005646

## Guardrail Evaluation

| Guardrail | Observed | Threshold | Comparator | Pass |
| --- | ---: | ---: | --- | --- |
| false_positive_rate | 0.000000 | 0.050000 | <= | True |
| effective_overhead_percent | 1.283708 | 3.000000 | <= | True |
| correctness_error | 0.000565 | 0.001000 | <= | True |
| uniform_recall | 0.972222 | 0.950000 | >= | True |
| critical_recall | 1.000000 | 0.990000 | >= | True |

- All guardrails passed: True
- Disable signal: False
- Fallback action: `continue_shadow_canary_ready_for_gate_review`

## Decision

- Decision: `promote`
- Rationale: Shadow canary satisfies all guardrails with deterministic evidence; track is ready for promotion gate review.

