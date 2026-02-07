# Acta Week 2 - Block 2 (T2 Boundary Validation & Closure)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: cierre del pendiente de validación de frontera T2 antes de cierre total del bloque.

## Context
Pending action from block-2 partial approval:
1. integrate promoted candidate in production scheduler,
2. keep fallback for larger sizes,
3. execute boundary validation on `1200-1600` and `1536-2048`.

Items (1) and (2) were completed earlier. This acta closes item (3).

## Executed Validation

Command:
- `./venv/bin/python research/breakthrough_lab/t2_auto_scheduler/run_week2_t2_search.py --search-space expanded --kernels tile20 tile24 --sizes 1200 1280 1400 1536 1600 1792 1920 2048 --runs-per-config 12 --replay-sessions 5 --replay-runs 10 --top-k 4 --correctness-threshold 1e-3 --seed 42 --input-distribution standard_normal`

Artifacts:
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.md`

## Quantitative Summary

- Completed search points: `48/48`
- Strict-valid points (`max_error <= 1e-3`): `46/48`
- Replay decision hint: `promote`

Boundary highlights:
- `t20_v3vec_v4_u0_l10` is strongly positive at `1200/1280/1400/1600/1792/1920`.
- `1536`: `t24_prod_v4_u0_l12` outperforms `t20_v3vec_v4_u0_l10` in this run.
- `2048`: `t20_v3vec_v4_u0_l10` is strongly negative vs baseline (`-53.953%`).

## Risk Assessment

- Positive: promoted 1400-class path remains justified.
- Risk: expanding T2 promoted scope as a contiguous range would introduce regressions due non-uniform boundary behavior.

## Formal Decision

Track `t2_auto_scheduler`: **iterate** (policy refinement), not broadening production scope yet.

Resolution:
- Keep production integration as currently approved: `size == 1400` promoted + fallback.
- Open next technical task: boundary-policy refinement (size/shape-aware rule) with strict replay before any scope expansion.

## Block Status

`Week 2 - Block 2` is now formally closed with boundary evidence included.
