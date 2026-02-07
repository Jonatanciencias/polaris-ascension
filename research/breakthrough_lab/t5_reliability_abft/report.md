# T5 Report - Week 4 Block 2 (ABFT-lite Coverage Refinement)

- Status: completed
- Decision: iterate
- Promotion gate: NOT YET

## Summary
- Block 2 objective executed: improve `uniform_random` recall while preserving low overhead.
- Verifier refinement applied:
  - ABFT row/column sampled checks (existing)
  - projection-check bank (`projection_count=4`) for broader fault observability
  - sparse periodic verification modes (`periodic_4`, `periodic_8`)
- Recommended mode: `periodic_8`.
- Effective overhead (`periodic_8`): `1.206%`.
- Critical recall (`periodic_8`): `1.000` with `0` misses.
- Uniform-random recall (`periodic_8`): `1.000`.
- False positive rate: `0.000`.
- Correctness guard passed: `max_error=0.0005493` (`<=1e-3`).

## Interpretation
- The Block 2 objective is satisfied: random-fault detection coverage increased strongly without losing overhead budget.
- Both tested periodic modes pass the detect-only quality gate.
- Track should proceed to integration hardening and longer stress campaigns before promotion.

## Evidence
- research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_205124.json
- research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_205124.md
- research/breakthrough_lab/t5_reliability_abft/results.json
- research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py
