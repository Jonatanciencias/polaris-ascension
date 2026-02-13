# T5 Week 5 Block 3 - Production Wiring Report

- Date: 2026-02-07T23:41:33.065607+00:00
- Policy: `t5-abft-detect-only-block3-2026-02-07`
- Sizes: [1400, 2048]
- Sessions/size=8 | Iterations/session=16 | Seed=42

## Aggregate Metrics

- Kernel avg GFLOPS mean: 844.417
- Effective overhead percent: 1.221
- False positive rate: 0.000
- Correctness max error: 0.0005951
- Disable events: 0

## Per-Size

| Size | Kernel Avg GFLOPS | Overhead % | False Pos Rate | Max Error | Disable Events |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1400 | 910.821 | 1.575 | 0.000 | 0.0003510 | 0 |
| 2048 | 778.013 | 0.868 | 0.000 | 0.0005951 | 0 |

## Guardrail Evaluation

| Guardrail | Observed | Threshold | Comparator | Pass |
| --- | ---: | ---: | --- | --- |
| false_positive_rate | 0.000000 | 0.050000 | <= | True |
| effective_overhead_percent | 1.221293 | 3.000000 | <= | True |
| correctness_error | 0.000595 | 0.001000 | <= | True |
| uniform_recall_reference | 0.966667 | 0.950000 | >= | True |
| critical_recall_reference | 1.000000 | 0.990000 | >= | True |

- Disable signal: False
- Fallback action: `keep_t5_abft_runtime_guarded`

## Decision

- Decision: `promote`
- Rationale: T5 guarded runtime wiring passed all guardrails with zero auto-disable events.

