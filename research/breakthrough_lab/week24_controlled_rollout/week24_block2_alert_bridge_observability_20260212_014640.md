# Week 24 Block 2 - Alert Bridge Observability

- Date: 2026-02-12T01:46:59.839792+00:00
- Endpoint: `http://127.0.0.1:8795/week24/block2`
- Health URL: `http://127.0.0.1:8795/health`
- Cycles: `4`

## Checks

| Check | Pass |
| --- | --- |
| workflow_exists | True |
| scheduler_spec_exists | True |
| scheduler_health_all_checks | True |
| source_cycle_promote | True |
| source_alerts_promote | True |
| endpoint_present | True |
| cycle_success_ratio_threshold | False |
| latency_p95_threshold | False |
| degradation_high_alerts_none | False |
| retry_path_exercised | True |
| artifacts_written | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Observability Metrics

- cycle_success_ratio: `0.000000`
- attempt_success_ratio: `0.000000`
- dispatch_success_latency_p95_ms: `None`
- dispatch_success_latency_max_ms: `None`
- retries_rate: `1.000000`

## Alerts

- [high] `delivery_success_ratio_below_floor`: Observed cycle success ratio 0.000000 below floor 0.950000.
- [high] `delivery_healthcheck_failed`: Pre or post delivery health-check failed in at least one cycle.

## Decision

- Decision: `iterate`
- Failed checks: ['cycle_success_ratio_threshold', 'latency_p95_threshold', 'degradation_high_alerts_none']
- Rationale: Observability block found unresolved delivery reliability, latency, alerting, or validation issues.

