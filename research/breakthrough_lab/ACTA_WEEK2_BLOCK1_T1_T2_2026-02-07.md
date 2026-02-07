# Acta Week 2 - Block 1 (T1/T2)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: execution and formal review of Week 2 first block for tracks `t1_io_aware` and `t2_auto_scheduler`.

## Evidence Reviewed

- `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_174226.json`
- `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_174226.md`
- `research/breakthrough_lab/t1_io_aware/results.json`
- `research/breakthrough_lab/t1_io_aware/report.md`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_174303.json`
- `research/breakthrough_lab/t2_auto_scheduler/week2_t2_bounded_search_20260207_174303.md`
- `research/breakthrough_lab/t2_auto_scheduler/results.json`
- `research/breakthrough_lab/t2_auto_scheduler/report.md`
- `research/breakthrough_lab/PROMOTION_GATE_CHECKLIST.md`

## Review Criteria

- Performance uplift vs baseline.
- Correctness under strict threshold (`max_error <= 1e-3`).
- Stability (`cv_peak <= 0.03`).
- Promotion gate readiness.

## Track Decisions

| Track | Current State | Key Findings | Decision | Rationale |
| --- | --- | --- | --- | --- |
| `t1_io_aware` | 2 IO-aware variants implemented and benchmarked (10x20, 1400/2048/3072) | `io_prefetch_v1` reached `+4.708%` at 1400 with correctness/stability pass, but both variants regress hard at 2048/3072 | `refine` | Keep track active due partial uplift at 1400, but require redesign for large-size behavior before any promotion discussion |
| `t2_auto_scheduler` | Expanded bounded search executed with strict filtered ranking | `0/6` candidates passed strict correctness filter; best replay throughput still failed `max_error <= 1e-3` | `refine` | Precision quality is the blocker; ranking policy is correct now but numeric path must be hardened before expanding search space |

Summary:
- `continue`: 0
- `refine`: 2
- `stop`: 0

## Approved Actions (Next Block)

1. T1: add a third IO-aware variant with explicit large-size strategy (2048/3072), then re-run 10x20.
2. T1: enforce weighted objective across 1400/2048/3072 to block one-size overfitting.
3. T2: apply precision-focused fixes (accumulation/build options) and rerun strict-filter search.
4. T2: only if strict correctness passes, expand search dimensions (vector/unroll/local-size).

## Formal Resolution

`Week 2 - Block 1 (T1/T2)` is marked as completed.

No track is stopped at this checkpoint. Both tracks remain in `refine` state.
