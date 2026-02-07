# T2 Report - P0-11 Bounded Search Dry-Run

- Status: completed
- Decision: iterate
- Promotion gate: FAILED (performance and correctness thresholds not met)

## Summary
- Best candidate: tile24@2048
- Peak GFLOPS mean (replay): 786.056
- Delta vs baseline: +1.518%
- CV peak: 0.000137
- Max error mean: 0.001689

## Evidence
- research/breakthrough_lab/t2_auto_scheduler/dry_run_leaderboard.json
- research/breakthrough_lab/t2_auto_scheduler/dry_run_report.md
- research/breakthrough_lab/t2_auto_scheduler/dry_run_validation.json

## Next
- Expand bounded search space (tile/workgroup/vector/unroll).
- Add strict error filtering in candidate ranking.
- Re-run with higher configuration budget and deterministic replay.
