# T2 Report - Week 2 Block 1 (Strict Filtered Search)

- Status: completed
- Decision: refine
- Promotion gate: FAILED (strict correctness threshold not met)

## Summary
- Expanded search budget executed: `2 kernels x 3 sizes x 12 runs`
- Strict ranking filter applied first: `max_error <= 1e-3`
- Valid candidates after strict filter: `0/6`
- Best replayed candidate by throughput (`tile20@1400`): `802.609 GFLOPS`, `+0.111%`, `cv=0.00190`
- Correctness still fails under strict threshold (`max_error max = 0.001099`)

## Evidence
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_174303.json
- research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_174303.md

## Next
- Add precision-first ranking: reject any candidate with `max_error > 1e-3` before replay.
- Investigate numeric drift sources for tile20/tile24 at 1400/2048 (accumulation and build flags).
- Re-run bounded search after precision fixes, then expand search dimensions (vector/unroll/local-size variants).
