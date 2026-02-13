# Week 9 Block 1 - Long Mixed Canary Report

- Date: 2026-02-08T03:08:16.444097+00:00
- Sizes: [1400, 2048]
- Batches=24 | Sessions/batch=1 | Iterations/session=8

## Queue Pressure

- Pulses requested/completed/failures: 48/48/0

## Group Summary

| Kernel | Size | Avg GFLOPS | P95 ms | Max error | Drift % | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| auto_t3_controlled | 1400 | 876.480 | 6.137 | 0.0003891 | +0.260 | fallback_mean=0.000, disabled=0 |
| auto_t3_controlled | 2048 | 773.579 | 22.147 | 0.0006104 | -0.090 | fallback_mean=0.000, disabled=0 |
| auto_t5_guarded | 1400 | 898.196 | 5.985 | 0.0003815 | -2.026 | overhead=1.924%, fp=0.000, disable=1 |
| auto_t5_guarded | 2048 | 778.710 | 21.966 | 0.0005798 | -0.060 | overhead=0.754%, fp=0.000, disable=0 |

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
| t5_disable_events_zero | False |
| t5_drift_abs_percent | True |

## Decision

- Decision: `iterate`
- Rationale: Long canary is operational but one or more drift/guardrail checks failed.

