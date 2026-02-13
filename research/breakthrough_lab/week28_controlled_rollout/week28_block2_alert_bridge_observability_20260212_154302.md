# Week 28 Block 2 - Alert Bridge Observability

- Date: 2026-02-12T15:43:21.840227+00:00
- Endpoint: `http://127.0.0.1:8795/week28/block2`
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
- attempt_success_ratio: `0.800000`
- dispatch_success_latency_p95_ms: `14.213786649315807`
- dispatch_success_latency_max_ms: `14.231511999241775`
- retries_rate: `0.250000`

## Alerts

- none

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Bridge observability metrics stay healthy with degradation alerts under threshold and canonical gates green.

