# Week 9 Block 1 - Long Mixed Canary Report

- Date: 2026-02-08T03:20:17.380949+00:00
- Sizes: [1400, 2048]
- Batches=24 | Sessions/batch=1 | Iterations/session=8

## Queue Pressure

- Pulses requested/completed/failures: 48/48/0

## Group Summary

| Kernel | Size | Avg GFLOPS | P95 ms | Max error | Drift % | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| auto_t3_controlled | 1400 | 876.455 | 6.135 | 0.0003891 | -0.002 | fallback_mean=0.000, disabled=0 |
| auto_t3_controlled | 2048 | 772.866 | 22.169 | 0.0006104 | +0.062 | fallback_mean=0.000, disabled=0 |
| auto_t5_guarded | 1400 | 902.298 | 5.980 | 0.0003815 | -1.500 | overhead=1.861%, fp=0.000, disable=0 |
| auto_t5_guarded | 2048 | 778.705 | 21.965 | 0.0005798 | -0.009 | overhead=0.765%, fp=0.000, disable=0 |

## Guardrail Checks

| Check | Pass |
| --- | --- |
| pressure_failures_zero | True |
| t3_correctness_bound | True |
| t3_fallback_rate_mean | True |
| t3_policy_not_disabled | True |
| t3_drift_abs_percent | True |
| t5_correctness_bound | True |
| t5_overhead_mean_percent | True |
| t5_false_positive_rate_mean | True |
| t5_disable_events_zero | True |
| t5_drift_abs_percent | True |

## Decision

- Decision: `promote`
- Rationale: Long canary passed queue-pressure, drift and guardrail checks for T3/T5.

