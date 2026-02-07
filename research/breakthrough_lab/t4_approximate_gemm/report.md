# T4 Report - Week 4 Block 3 (Policy Gating by Compressibility)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (scoped)

## Summary
- Contract compliance rate: `1.000`.
- Post-fallback violation rate: `0.000`.
- Fallback rate: `0.000` (fallback reservado a violación real de contrato).
- Policy exact-route rate: `0.500`.
- Approximate-attempt rate: `0.500`.
- Compressible workload speedup vs exact: `3.022x`.
- Dense-random workload behavior: routed by policy to exact path (`policy_exact_route=1.000`), no contract escapes.

## Interpretation
- The compressibility-gated activation policy removes unnecessary fallback pressure.
- Safety remains deterministic (`contract=1.000`, `post_fallback_violation=0.000`).
- Promotion is approved as a scoped path: approximate only on eligible compressible workloads, exact route otherwise.

## Evidence
- research/breakthrough_lab/t4_approximate_gemm/week4_t4_policy_gating_20260207_224256.json
- research/breakthrough_lab/t4_approximate_gemm/week4_t4_policy_gating_20260207_224256.md
- research/breakthrough_lab/t4_approximate_gemm/policy_activation_block3.json
- research/breakthrough_lab/t4_approximate_gemm/results.json
- research/breakthrough_lab/t4_approximate_gemm/run_t4_policy_gating.py
- research/breakthrough_lab/ACTA_WEEK4_BLOCK3_T4_POLICY_GATING_2026-02-07.md
