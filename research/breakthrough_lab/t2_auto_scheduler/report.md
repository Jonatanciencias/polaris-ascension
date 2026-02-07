# T2 Report - Week 2 Block 2 Partial (Hardening + Strict Rerun)

- Status: completed
- Decision: iterate
- Promotion gate: FAILED (performance threshold not met, +10% required)

## Summary
- Hardening applied before rerun:
  - deterministic seed per config/session
  - canonical input distribution (`standard_normal`) aligned with baseline protocol
- Expanded search budget executed: `2 kernels x 3 sizes x 12 runs`
- Strict ranking filter applied first: `max_error <= 1e-3`
- Valid candidates after strict filter: `6/6`
- Best replayed candidate for uplift (`tile24@2048`):
  - `783.923 GFLOPS`
  - `+6.026%` vs baseline
  - `max_error max = 0.000610` (pass)
  - `cv = 0.00039` (pass)

## Evidence
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_183138.json
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_183138.md
- research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py
- research/auto_tuner/gemm_auto_tuner.py

## Next
- Keep strict correctness-first ranking active (already enforced).
- Expand search dimensions now that correctness gate is stable:
  - vector width
  - unroll
  - local-size candidates
- Target next checkpoint: close the remaining gap from `+6.026%` to `>= +10%`.
