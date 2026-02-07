# T5 Report - Week 4 Block 1 (ABFT-lite Detect-only)

- Status: completed
- Decision: iterate
- Promotion gate: NOT YET

## Summary
- ABFT-lite detect-only campaign executed with deterministic fault injection at sizes `1400` and `2048`.
- Modes evaluated:
  - `always` (sampling coverage `1.000`)
  - `periodic_4` (sampling coverage `0.250`)
- Recommended mode: `periodic_4`.
- Effective overhead (`periodic_4`): `0.973%`.
- Critical fault recall (`periodic_4`): `1.000` with `0` misses.
- False positive rate on clean outputs: `0.000`.
- Correctness guard (`max_error <= 1e-3`) passed: `True` (`max_error=0.0005646`).

## Interpretation
- The detect-only ABFT prototype is operationally viable in overhead and critical detection under monitored-fault scope.
- Sparse checksum sampling leaves low coverage for uniform-random fault space (`uniform recall=0.000` in recommended mode).
- The track should continue with guardrail refinement (coverage policy and sampling strategy) before any promotion claim.

## Evidence
- research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_203936.json
- research/breakthrough_lab/t5_reliability_abft/week4_t5_abft_detect_only_20260207_203936.md
- research/breakthrough_lab/t5_reliability_abft/results.json
- research/breakthrough_lab/t5_reliability_abft/run_t5_abft_detect_only.py
