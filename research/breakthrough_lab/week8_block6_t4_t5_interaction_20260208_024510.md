# Week 8 Block 6 - T4+T5 Interaction Report

- Date: 2026-02-08T02:45:10.399477+00:00
- Sizes: [1400, 2048]
- Sessions=3 | Iterations=8 | Seed=42

## T5 Baseline vs Combined

- Baseline avg GFLOPS: 841.470
- Combined avg GFLOPS: 843.509
- Baseline p95 ms: 13.995
- Combined p95 ms: 13.973
- Overhead delta: +0.069%
- P95 delta: -0.159%
- Avg GFLOPS delta: +0.242%

## T4 Combined State

- Contract compliance: 1.000
- Post-fallback violation rate: 0.000
- Fallback rate: 0.000

## Guardrail Checks

| Check | Pass |
| --- | --- |
| t4_contract_compliance | True |
| t4_post_fallback_violations | True |
| t5_correctness_combined | True |
| t5_overhead_cross_delta | True |
| t5_p95_cross_delta | True |
| t5_avg_gflops_cross_drop | True |

## Decision

- Decision: `promote`
- Rationale: Combined T4+T5 profile stayed within cross-effect overhead/latency bounds and preserved correctness.

