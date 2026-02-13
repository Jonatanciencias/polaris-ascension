# Week 30 Block 2 - Alert Bridge Observability

- Date: 2026-02-13T17:48:06.156390+00:00
- Endpoint: `http://127.0.0.1:8875/webhook`
- Health URL: `http://127.0.0.1:8875/health`
- Cycles: `3`

## Checks

| Check | Pass |
| --- | --- |
| workflow_exists | True |
| scheduler_spec_exists | True |
| scheduler_health_all_checks | True |
| source_cycle_promote | True |
| source_alerts_promote | True |
| endpoint_present | True |
| cycle_success_ratio_threshold | True |
| latency_p95_threshold | True |
| degradation_high_alerts_none | True |
| retry_path_exercised | True |
| artifacts_written | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Observability Metrics

- cycle_success_ratio: `1.000000`
- attempt_success_ratio: `0.750000`
- dispatch_success_latency_p95_ms: `11.887622900030692`
- dispatch_success_latency_max_ms: `11.8908360000205`
- retries_rate: `0.333333`

## Alerts

- none

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Bridge observability metrics stay healthy with degradation alerts under threshold and canonical gates green.

