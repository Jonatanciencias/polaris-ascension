# T3 Report - Week 5 Block 1 (Controlled Production Integration)

- Status: completed
- Decision: iterate
- Promotion gate: PARTIAL (guardrails pass, uplift gate pending)

## Summary
- Controlled integration executed via production benchmark path (`auto_t3_controlled`) with strict deterministic protocol.
- Executed steps: 32/32 sessions across 8 scope sizes.
- Portfolio uplift vs static auto: +2.053%.
- P95 latency delta vs static: -1.343%.
- Fallback rate: 0.000 (threshold 0.100).
- Correctness failures: 0.
- Disable events: 0.

## Interpretation
- Runtime guardrails behaved as intended: no correctness escapes and no forced disables.
- Latency behavior remains stable/improved under controlled mode.
- Portfolio uplift is positive but below the +5.0% gate; refinement is required before promotion.

## Evidence
- research/breakthrough_lab/t3_online_control/policy_controlled_block1.json
- research/breakthrough_lab/t3_online_control/week5_t3_controlled_production_20260207_231451.json
- research/breakthrough_lab/t3_online_control/week5_t3_controlled_production_20260207_231451.md
- research/breakthrough_lab/t3_online_control/results.json
- research/breakthrough_lab/t3_online_control/run_week5_t3_controlled_production.py
- src/optimization_engines/t3_controlled_policy.py
- src/optimization_engines/adaptive_kernel_selector.py
- src/benchmarking/production_kernel_benchmark.py
