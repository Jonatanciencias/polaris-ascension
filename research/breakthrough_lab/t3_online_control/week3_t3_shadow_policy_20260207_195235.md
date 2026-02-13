# T3 Week 3 Shadow Policy Report

- Date: 2026-02-07T19:52:35.158590+00:00
- Protocol: epochs=2, runs_per_decision=8, warmup=2, seed=42
- Epsilon: 0.15, fallback_regression_limit=0.1, max_fallback_rate=0.2

## Summary

- Static mean GFLOPS: 766.131
- Shadow executed mean GFLOPS: 766.131
- Mean uplift vs static: +0.000%
- P95 latency delta vs static: +0.000%
- Fallback rate: 0.333
- Correctness failures: 0
- Stop rule triggered: True
- Decision hint: drop

## Policy Snapshot

```json
{
  "small": {
    "tile20": {
      "count": 3,
      "mean_reward": 766.1306773832004
    },
    "tile20_v3_1400": {
      "count": 0,
      "mean_reward": 0.0
    },
    "tile24": {
      "count": 0,
      "mean_reward": 0.0
    }
  }
}
```
