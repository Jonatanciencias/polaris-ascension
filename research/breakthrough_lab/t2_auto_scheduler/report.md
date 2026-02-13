# T2 Report - Week 2 Block 2 Partial (Expanded Space + Strict Deterministic)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (scoped candidate)

## Summary
- Deterministic strict mode maintained:
  - deterministic seed per config/session
  - canonical input distribution (`standard_normal`)
  - strict filter (`max_error <= 1e-3`)
- Search space expanded to dimensions:
  - `vector_width`: `{4, 8}`
  - `unroll_k`: `{0, 4, 8, 10}`
  - `local_size`: `{5x5, 10x10, 12x12}`
- Budget executed: `6 configs x 3 sizes x 12 runs`
- Valid candidates after strict filter: `16/18`
- Promoted candidate (scoped):
  - `t20_v3vec_v4_u0_l10 @ 1400`
  - `926.303 GFLOPS` replay mean
  - `+15.838%` vs baseline
  - `max_error max = 0.000336` (pass)
  - `cv = 0.00421` (pass)

## Evidence
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_184001.json
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_184001.md
- research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py
- research/auto_tuner/gemm_auto_tuner.py

## Next
- Integrate `t20_v3vec_v4_u0_l10` as size-scoped scheduler candidate (1400 class).
- Preserve baseline `tile24` for large sizes to avoid regressions.
- Open a focused validation sweep around `1200-1600` and `1536-2048` boundary policy.
