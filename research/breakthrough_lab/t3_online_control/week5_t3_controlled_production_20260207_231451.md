# T3 Week 5 Block 1 - Controlled Production Integration Report

- Date: 2026-02-07T23:14:51.636446+00:00
- Policy: `t3-controlled-week5-block1-2026-02-07`
- Sizes: [1200, 1280, 1400, 1536, 1600, 1792, 1920, 2048]
- Sessions=4 | Iterations=12 | Seed=42

## Aggregate Metrics

- Static avg GFLOPS mean: 753.667
- Controlled avg GFLOPS mean: 769.142
- Delta vs static: +2.053%
- P95 latency delta: -1.343%
- Fallback rate: 0.000
- Correctness failures: 0
- Disable events: 0

## Per-Size Summary

| Size | Static Avg GFLOPS | Controlled Avg GFLOPS | Delta | Fallback Rate | Correctness Fails |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1200 | 708.065 | 708.065 | +0.000% | 0.000 | 0 |
| 1280 | 610.134 | 733.936 | +20.291% | 0.000 | 0 |
| 1400 | 888.910 | 888.910 | +0.000% | 0.000 | 0 |
| 1536 | 770.081 | 770.081 | +0.000% | 0.000 | 0 |
| 1600 | 764.407 | 764.407 | +0.000% | 0.000 | 0 |
| 1792 | 769.913 | 769.913 | +0.000% | 0.000 | 0 |
| 1920 | 785.118 | 785.118 | +0.000% | 0.000 | 0 |
| 2048 | 732.704 | 732.704 | +0.000% | 0.000 | 0 |

## Gate Evaluation

```json
{
  "min_uplift_percent": {
    "threshold": 5.0,
    "observed": 2.053342218782413,
    "comparator": ">=",
    "pass": false
  },
  "max_p95_latency_delta_percent": {
    "threshold": 3.0,
    "observed": -1.3429396115653578,
    "comparator": "<=",
    "pass": true
  },
  "max_fallback_rate": {
    "threshold": 0.1,
    "observed": 0.0,
    "comparator": "<=",
    "pass": true
  },
  "max_correctness_failures": {
    "threshold": 0,
    "observed": 0,
    "comparator": "<=",
    "pass": true
  }
}
```

## Decision

- Decision: `iterate`
- Rationale: Uplift gate not reached; maintain controlled rollout and gather more data.

