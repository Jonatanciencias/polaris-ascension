# T4 Report - Week 5 Block 2 (Controlled Integration with policy_activation_block3)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (controlled)

## Summary
- Contract compliance rate: `1.000`.
- Post-fallback violation rate: `0.000`.
- Fallback rate: `0.000`.
- Policy exact-route rate: `0.500`.
- Approximate-attempt rate: `0.500`.
- Compressible workload speedup vs exact: `2.852x`.
- Guardrails passed: `True` (disable signal `False`).

## Interpretation
- The scoped activation policy remains stable under strict deterministic rerun.
- No contract escapes were observed after fallback semantics.
- Controlled integration stays eligible for next gate progression.

## Evidence
- research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json
- research/breakthrough_lab/t4_approximate_gemm/run_week5_t4_controlled_integration.py
- research/breakthrough_lab/t4_approximate_gemm/week5_t4_controlled_integration_20260207_233025.json
- research/breakthrough_lab/t4_approximate_gemm/week5_t4_controlled_integration_20260207_233025.md
- research/breakthrough_lab/t4_approximate_gemm/results.json
- research/breakthrough_lab/ACTA_WEEK5_BLOCK2_T4_CONTROLLED_INTEGRATION_2026-02-07.md
