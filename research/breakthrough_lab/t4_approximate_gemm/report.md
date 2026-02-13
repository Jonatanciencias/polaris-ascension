# T4 Report - Week 8 Block 4 (Mixed Policy Refinement)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (mixed-workload refinement)

## Summary
- Baseline policy: `policy_activation_block3.json` (target_rank `16`)
- Candidate policy: `policy_activation_block4.json` (target_rank `18`)
- Workload campaign: `dense_random + compressible_lowrank`, sizes `512/768/1024`, `noise_scale=0.02`.

Baseline vs candidate:
- Contract compliance rate: `1.000 -> 1.000`
- Post-fallback violation rate: `0.000 -> 0.000`
- Fallback rate: `0.194 -> 0.000` (reduction `0.194`)
- Compressible speedup vs exact: `1.370x -> 1.999x`
- Delta vs exact (%): `12.173 -> 39.540`

## Interpretation
- The refined activation policy reduced fallback events to zero without contract escapes.
- Guardrails passed with rollback-safe thresholds for mixed workload operation.
- Candidate remains within routing cap (exact-route `0.500` <= `0.800`) and keeps positive acceleration envelope.

## Evidence
- `research/breakthrough_lab/t4_approximate_gemm/policy_activation_block4.json`
- `research/breakthrough_lab/t4_approximate_gemm/run_week8_t4_mixed_campaign.py`
- `research/breakthrough_lab/t4_approximate_gemm/week8_t4_mixed_campaign_20260208_021541.json`
- `research/breakthrough_lab/t4_approximate_gemm/week8_t4_mixed_campaign_20260208_021541.md`
- `research/breakthrough_lab/t4_approximate_gemm/results.json`
- `research/breakthrough_lab/ACTA_WEEK8_BLOCK4_T4_MIXED_POLICY_REFINEMENT_2026-02-08.md`
