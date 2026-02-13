# Week 22 Block 2 - Alert Bridge Live Cutover + Rollback

- Date: 2026-02-11T16:11:24.066048+00:00
- Dispatch mode: `live`
- Rollback channel: `spool-fallback`

## Checks

| Check | Pass |
| --- | --- |
| workflow_exists | True |
| scheduler_spec_exists | True |
| scheduler_health_all_checks | True |
| source_cycle_promote | True |
| source_alerts_promote | True |
| dispatch_mode_live | True |
| endpoint_present_for_live | True |
| bridge_payload_written | True |
| dispatch_record_written | True |
| dispatch_live_success | True |
| rollback_record_written | True |
| rollback_triggered_on_failure_or_not_needed | True |
| no_high_critical_open_debt | True |
| pre_gate_promote | True |
| pre_gate_pytest_tier_green | True |
| post_gate_promote | True |
| post_gate_pytest_tier_green | True |

## Highlights

- source_cycle_decision: `promote`
- source_alerts_decision: `promote`
- bridged_alerts_count: `1`
- dispatch_mode: `live`
- dispatch_sent: `True`
- rollback_triggered: `False`

## Artifacts

- `bridge_payload_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block2_alert_bridge_live_cutover_bridge_payload_20260211_161104.json`
- `dispatch_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block2_alert_bridge_live_cutover_dispatch_20260211_161104.json`
- `scheduler_health_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block2_alert_bridge_live_cutover_scheduler_health_20260211_161104.json`
- `rollback_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block2_alert_bridge_live_cutover_rollback_20260211_161104.json`
- `rollback_spool_json`: ``
- `operational_debt_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week22_controlled_rollout/week22_block2_alert_bridge_live_cutover_operational_debt_20260211_161104.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_161104.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_161124.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Live webhook cutover succeeded with explicit rollback path ready and canonical gates green.

