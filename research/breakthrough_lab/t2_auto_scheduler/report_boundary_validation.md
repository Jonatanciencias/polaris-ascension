# T2 Report - Week 2 Block 2 (Boundary Validation)

- Status: completed
- Decision: iterate
- Integration scope in production: keep `size == 1400` promoted path + fallback

## Objective
Validate boundary behavior after T2 production integration using strict deterministic search on:
- `1200-1600`
- `1536-2048`

## Protocol
- Command:
  - `./venv/bin/python research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py --search-space expanded --kernels tile20 tile24 --sizes 1200 1280 1400 1536 1600 1792 1920 2048 --runs-per-config 12 --replay-sessions 5 --replay-runs 10 --top-k 4 --correctness-threshold 1e-3 --seed 42 --input-distribution standard_normal`
- Strict filter: `max_error <= 1e-3`
- Determinism: fixed seed + canonical `standard_normal` inputs

## Results
- Search completed: `48/48` candidates executed, `46/48` valid after strict filter.
- Replay decision hint: `promote` (there are replay candidates passing all gates).

Key boundary findings:
- `t20_v3vec_v4_u0_l10` stays strong on several boundary points:
  - `1200`: `+23.846%`
  - `1280`: `+32.061%`
  - `1400`: `+16.385%`
  - `1600`: `+26.887%`
  - `1792`: `+40.352%`
  - `1920`: `+14.863%` vs `tile24` baseline
- Critical non-uniformity:
  - `1536`: `tile24` is better than `t20_v3` in this search (`795.765` vs `615.643` GFLOPS)
  - `2048`: `t20_v3` regresses hard (`-53.953%` vs baseline `tile24`)

## Interpretation
- The promoted T2 candidate is robustly beneficial in the validated 1400-class scope.
- Boundary behavior is non-monotonic; broad range promotion without refined policy would be risky.

## Formal Decision
- Keep current production scope unchanged (`1400` exact + fallback).
- Mark boundary validation as completed.
- Open follow-up for selector boundary-policy refinement (shape/range-aware, with dedicated replay).

## Evidence
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.md`
- `research/breakthrough_lab/ACTA_WEEK2_BLOCK2_T2_BOUNDARY_VALIDATION_2026-02-07.md`
