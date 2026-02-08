# T3 Week 8 Block 3 - Drift Campaign Report

- Date: 2026-02-08T02:01:48.401359+00:00
- Policy: `t3-controlled-week8-block3-2026-02-08`
- Sizes: [1400, 1536, 2048]
- Sessions=2 | Iterations=6 | Seed=42

## Scenario Summary

| Scenario | Auto avg GFLOPS | T3 avg GFLOPS | T3 delta vs auto | T3 p95 delta vs auto | T3 fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 675.709 | 809.771 | +19.840% | -20.566% | 0.000 |
| warm | 676.969 | 806.897 | +19.193% | -20.070% | 0.000 |
| warm_queue_pressure | 676.397 | 808.205 | +19.487% | -20.313% | 0.000 |

## Drift Drop vs Cold

```json
{
  "auto": -0.10182849128358912,
  "auto_t3_controlled": 0.19340545274946863
}
```

## Queue Pressure Pulses

```json
{
  "cold": {
    "requested": 0,
    "completed": 0,
    "failures": 0,
    "errors": []
  },
  "warm": {
    "requested": 0,
    "completed": 0,
    "failures": 0,
    "errors": []
  },
  "warm_queue_pressure": {
    "requested": 12,
    "completed": 12,
    "failures": 0,
    "errors": []
  }
}
```

## Gate Evaluation

```json
{
  "checks": {
    "correctness_guard": {
      "observed_max_error": 0.000640869140625,
      "threshold_max_error": 0.001,
      "observed_correctness_failures": 0,
      "threshold_correctness_failures": 0,
      "pass": true
    },
    "pressure_t3_fallback_rate": {
      "observed": 0.0,
      "threshold_max": 0.08,
      "pass": true
    },
    "pressure_t3_p95_delta_vs_auto": {
      "observed_percent": -20.312766780136634,
      "threshold_max_percent": 4.0,
      "pass": true
    },
    "pressure_t3_drop_vs_cold": {
      "observed_percent": 0.19340545274946863,
      "threshold_max_percent": 20.0,
      "pass": true
    },
    "pressure_auto_drop_vs_cold": {
      "observed_percent": -0.10182849128358912,
      "threshold_max_percent": 25.0,
      "pass": true
    },
    "pressure_t3_delta_vs_auto": {
      "observed_percent": 19.486641020685266,
      "threshold_min_percent": -1.0,
      "pass": true
    },
    "policy_not_disabled": {
      "observed": false,
      "required": false,
      "pass": true
    }
  },
  "failed_checks": []
}
```

## Decision

- Decision: `promote`
- Rationale: Drift campaign passed correctness and rollback-safe guardrails under warm/cold/pressure scenarios.

