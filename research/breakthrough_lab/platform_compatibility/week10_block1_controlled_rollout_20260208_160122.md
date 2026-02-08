# Week 10 Block 1 - Controlled Low-Scope Rollout

- Date: 2026-02-08T16:01:22.914390+00:00
- Planned snapshots: 3
- Executed snapshots: 2
- Hourly interval (logical): 60.0 minutes
- Auto rollback enabled: True

## Snapshot Decisions

| Snapshot | Decision | Failed checks |
| ---: | --- | --- |
| 1 | warn | t5_overhead_soft_limit |
| 2 | rollback | t5_guardrails_hard, t5_overhead_soft_limit |

## Campaign Checks

| Check | Pass |
| --- | --- |
| minimum_snapshots_completed | True |
| all_snapshot_hard_checks_passed | False |
| soft_overhead_consecutive_below_limit | False |
| rollback_policy_enforced | True |
| drift_abs_percent_bounded | True |

## Auto Rollback Event

- Trigger snapshot: 2
- Trigger checks: ['t5_guardrails_hard']
- Rollback invoked: True
- Rollback success: True

## Decision

- Decision: `iterate`
- Rationale: Auto rollback triggered by guardrail breach; rollout remains in iterate mode.

