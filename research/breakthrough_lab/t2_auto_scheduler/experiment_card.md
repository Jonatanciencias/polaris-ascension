# Experiment Card - T2 Auto Scheduler

## Metadata

- Experiment ID: `t2-001`
- Track: `t2_auto_scheduler`
- Owner: `track-t2`
- Status: `planned`

## Hypothesis

An automated schedule search with bounded space can discover kernel configurations that outperform manual defaults while reducing tuning time.

## Method

- Baseline: canonical baseline from `../BASELINE_RUNBOOK.md`.
- Define bounded search space for:
  - tile size
  - local size
  - vector width
  - unroll factor
- Run fixed-budget search and replay top candidates.

## Variables

- independent:
  - search space dimensions
  - scoring objective (peak vs sustained)
  - exploration budget
- controlled:
  - fixed benchmark protocol and hardware

## Success Metrics

- best candidate `delta_vs_baseline_percent >= 10`
- same or better stability (`cv_peak <= baseline_cv`)
- search runtime acceptable for iteration cadence

## Stop Rule

Stop when search budget is exhausted or 3 consecutive iterations do not improve top score by >= 1%.

## Artifacts

- `results.json`
- candidate leaderboard (JSON/MD)
- final selected candidate config

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
