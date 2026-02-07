# T4 Report - Week 3 Block 2 (Approximate GEMM Contract)

- Status: completed
- Decision: iterate
- Promotion gate: NOT YET

## Summary
- Contract compliance rate: `1.000`.
- Post-fallback violation rate: `0.000`.
- Fallback rate: `0.500`.
- Compressible workload speedup vs exact: `2.972x`.
- Dense-random workload behavior: full fallback (`1.000`), no contract escapes.

## Interpretation
- The contract + fallback mechanism behaves safely and deterministically.
- Performance benefit is validated for compressible matrices, not for dense-random general case.
- This supports continuation as a scoped path, not broad promotion yet.

## Evidence
- research/breakthrough_lab/t4_approximate_gemm/week3_t4_contract_run_20260207_200118.json
- research/breakthrough_lab/t4_approximate_gemm/week3_t4_contract_run_20260207_200118.md
- research/breakthrough_lab/t4_approximate_gemm/results.json
- research/breakthrough_lab/t4_approximate_gemm/run_t4_error_contract.py
