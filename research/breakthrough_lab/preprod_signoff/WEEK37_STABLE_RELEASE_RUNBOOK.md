# WEEK37 Stable Release Runbook (`v0.15.0`)

## Pre-flight

1. Run canonical gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
2. Verify latest continuity chain:
   - Week37 Block1/2/3 decisions in `promote`.
3. Verify wall-clock and formal go/no-go:
   - `research/breakthrough_lab/week37_rx590_extended_wallclock_decision.json` -> `go`.

## Release Freeze

1. Freeze release tag: `v0.15.0`.
2. Publish release notes and final checklist.
3. Keep controlled deployment discipline:
   - gate canónico obligatorio pre/post por bloque mensual.

## Operational Guardrails

- T3 fallback policy must remain enabled and monitored.
- T5 requirements:
  - `disable_events = 0`
  - overhead within active policy thresholds.
- Platform policy: `dual_go_clover_rusticl` with rollback SLA active.

## Rollback SLA

1. Trigger immediate rollback if any of:
   - canonical gate not `promote`
   - `disable_events > 0`
   - guardrail breach in T3/T5 or critical correctness drift.
2. Revert to last known-good policy set and rerun canonical gate.
3. Record incident and pause scope expansion until formal recovery closes in `promote`.
