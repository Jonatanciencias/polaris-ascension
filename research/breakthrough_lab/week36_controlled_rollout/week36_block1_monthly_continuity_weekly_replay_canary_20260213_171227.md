# Week 10 Block 1 - Controlled Low-Scope Rollout

- Date: 2026-02-13T17:12:27.970706+00:00
- Planned snapshots: 8
- Executed snapshots: 6
- Hourly interval (logical): 60.0 minutes
- Auto rollback enabled: True

## Snapshot Decisions

| Snapshot | Decision | Failed checks |
| ---: | --- | --- |
| 1 | continue | - |
| 2 | continue | - |
| 3 | continue | - |
| 4 | continue | - |
| 5 | continue | - |
| 6 | rollback | t5_guardrails_hard, t5_overhead_soft_limit |

## Campaign Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots_completed | True |
| all_snapshot_hard_checks_passed | False |
| soft_overhead_consecutive_below_limit | True |
| rollback_policy_enforced | True |
| drift_abs_percent_bounded | False |

## Auto Rollback Event

- Trigger snapshot: 6
- Trigger checks: ['t5_guardrails_hard']
- Rollback invoked: True
- Rollback success: True

## Decision

- Decision: `iterate`
- Rationale: Auto rollback triggered by guardrail breach; rollout remains in iterate mode.

