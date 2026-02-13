# Week 18 Block 1 - Comparative Update

- Date: 2026-02-11T01:31:41.461884+00:00
- Baseline: `week16_block2_weekly_rc_replay_rerun`
- Current: `week17_block4_posthardening_replay`

## Decision Chain

- `week17_block1`: `go`
- `week17_block2`: `promote`
- `week17_block3`: `promote`
- `week17_block4`: `promote`

## Drift Delta

- Throughput drift max abs: 4.4509% -> 0.6065% (delta -3.8444%)
- P95 drift max: 0.6219% -> 0.3241% (delta -0.2978%)

## Stability Highlights

- Week17 Block1 snapshots: 10
- Week17 Block1 T5 disable total: 0
- Week17 Block3 repeat campaign: 20 passed / 0 failed

## Conclusions

- `drift_improved`: `True`
- `pytest_flake_stabilized`: `True`
- `stable_rollout_go`: `True`

