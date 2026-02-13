# WEEK37 Stable Release Notes (`v0.15.0`)

## Summary

- Release type: stable freeze after sustained continuity and extended RX590 wall-clock validation.
- Baseline chain: Week36 Block1/2/3 `promote` -> Week37 Block1/2/3 `promote`.
- Production canary status: Week37 RX590 extended wall-clock `GO`.

## Evidence Chain

- Week37 Block1 decision: `research/breakthrough_lab/week37_block1_monthly_continuity_decision.json`
- Week37 Block2 decision: `research/breakthrough_lab/week37_block2_alert_bridge_observability_decision.json`
- Week37 Block3 decision: `research/breakthrough_lab/week37_block3_biweekly_comparative_decision.json`
- Week37 RX590 wall-clock decision: `research/breakthrough_lab/week37_rx590_extended_wallclock_decision.json`

## Key Technical Outcomes

- T5 guardrails stable (`disable_events=0`) across Week37 continuity and wall-clock campaign.
- Platform comparative closes with `dual_go_clover_rusticl` under active guardrails.
- Canonical validation gate remains `promote` before freeze (`validation_suite_canonical_20260213_183821.json`).

## Scope of This Freeze

- Controlled production profile for sizes `1400/2048/3072`.
- Mandatory canonical gate before any scope expansion.
- Rollback SLA remains active and required for operations.

## Out of Scope

- Expansion to new hardware classes without dedicated validation.
- Plugin API breaking changes.
