# T1 Report - Week 2 Block 3 (Closure Attempt)

- Status: completed
- Decision: stop
- Promotion gate: FAILED (insufficient uplift)

## Summary
- Last technical attempt executed with 3 variants and weighted objective `{1400:0.2, 2048:0.4, 3072:0.4}`.
- Best candidate: `io_hybrid_sizeaware_v1`.
- Weighted delta vs baseline: `+0.766%` (target was materially higher for promotion).
- Unweighted mean delta: `+1.454%`.
- Correctness/stability: pass on all sizes (`max_error <= 1e-3`, `cv_peak <= 0.03`).
- Stop rule triggered: **True** (after 3 variants, no candidate reached `+5%` at size `1400` or `2048`).

## Per-size (best candidate)
- `1400`: `+4.891%`
- `2048`: `-0.156%`
- `3072`: `-0.375%`

## Evidence
- `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_193227.json`
- `research/breakthrough_lab/t1_io_aware/week2_t1_io_campaign_20260207_193227.md`
- `research/breakthrough_lab/t1_io_aware/results.json`
- `research/breakthrough_lab/t1_io_aware/run_week2_t1_campaign.py`

## Closure
Track `t1_io_aware` is closed in current form and removed from active promotion path.
Any future re-entry must start as a new hypothesis/card with a different mechanism, not incremental tuning of the current variants.
