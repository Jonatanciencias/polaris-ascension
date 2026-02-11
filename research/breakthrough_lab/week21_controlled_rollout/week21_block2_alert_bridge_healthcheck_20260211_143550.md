# Week 21 Block 2 - Alert Bridge + Scheduler Health-check

- Date: 2026-02-11T14:36:09.689434+00:00
- Workflow path: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/.github/workflows/week20-monthly-cycle.yml`
- Bridge mode: `dry_run`

## Checks

| Check | Pass |
| --- | --- |
| workflow_exists | True |
| scheduler_spec_exists | True |
| scheduler_health_all_checks | True |
| source_cycle_promote | True |
| source_alerts_promote | True |
| bridge_payload_written | True |
| dispatch_record_written | True |
| dispatch_successful_or_dry_run | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- Source cycle decision: `promote`
- Source alerts decision: `promote`
- Bridged alerts count: `1`
- Heartbeat emitted: `True`
- Dispatch sent: `False`

## Artifacts

- `bridge_payload_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_bridge_payload_20260211_143550.json`
- `dispatch_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_dispatch_20260211_143550.json`
- `scheduler_health_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_scheduler_health_20260211_143550.json`
- `operational_debt_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week21_controlled_rollout/week21_block2_alert_bridge_healthcheck_operational_debt_20260211_143550.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_143550.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_143609.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Alert bridge dry-run and scheduler health-check are stable with canonical gates green.

