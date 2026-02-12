# Week 23 Block 2 - Alert Bridge Hardening

- Date: 2026-02-12T00:55:29.705523+00:00
- Endpoint: `http://127.0.0.1:8775/alerts`
- Health URL: `http://127.0.0.1:8775/health`

## Checks

| Check | Pass |
| --- | --- |
| workflow_exists | True |
| scheduler_spec_exists | True |
| scheduler_health_all_checks | True |
| source_cycle_promote | True |
| source_alerts_promote | True |
| endpoint_present_for_live | True |
| delivery_healthcheck_pre_ok | True |
| delivery_healthcheck_post_ok | True |
| retry_configured | True |
| retry_path_exercised | True |
| dispatch_live_success | True |
| bridge_payload_written | True |
| dispatch_record_written | True |
| rollback_record_written | True |
| rollback_triggered_on_failure_or_not_needed | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- dispatch_sent: `True`
- attempts_executed: `2`
- retries_used: `True`
- pre_health_ok: `True`
- post_health_ok: `True`
- rollback_triggered: `False`

## Artifacts

- `bridge_payload_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block2_alert_bridge_hardening_bridge_payload_20260212_005510.json`
- `dispatch_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block2_alert_bridge_hardening_dispatch_20260212_005510.json`
- `scheduler_health_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block2_alert_bridge_hardening_scheduler_health_20260212_005510.json`
- `rollback_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block2_alert_bridge_hardening_rollback_20260212_005510.json`
- `rollback_spool_json`: ``
- `operational_debt_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week23_controlled_rollout/week23_block2_alert_bridge_hardening_operational_debt_20260212_005510.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_005509.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_005529.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Alert bridge hardening validated retry/backoff and delivery health checks with canonical gates green.

