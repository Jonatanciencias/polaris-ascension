# v0.15.0 Controlled Release Runbook

## Pre-flight

1. Run canonical gate with driver smoke.
2. Verify platform diagnostics (`verify_drivers.py --json`).
3. Verify latest weekly replay report is `promote`.

## Release

1. Tag release candidate lineage as stable (`v0.15.0`).
2. Publish stable manifest and notes.
3. Keep controlled rollout mode for first production window.

## Rollback

1. If `disable_events > 0` or overhead guardrail breach, rollback immediately.
2. Revert to last RC-known-good policy and rerun canonical gate.
3. Open incident note and freeze scope expansion.

