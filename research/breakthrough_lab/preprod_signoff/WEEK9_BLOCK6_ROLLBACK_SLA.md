# Week9 Block6 Rollback SLA (Controlled Production)

## Scope

Operational rollback commitments for controlled production rollout after Week9 Block6 sign-off.

## Trigger Conditions

1. Correctness breach (`max_error > 1e-3`) in canary/production telemetry.
2. T5 disable events unexpectedly > 0 in stable load profile.
3. T3 fallback sustained above policy limit (`> 0.08`) over one decision window.
4. Driver/runtime health degraded (`verify_drivers.py --json` not `overall_status=good`).

## SLA Targets

1. Detection to decision: <= 10 minutes.
2. Decision to rollback apply: <= 5 minutes.
3. Rollback apply to first post-rollback canonical gate start: <= 10 minutes.
4. Rollback apply to service stabilization (guardrails healthy): <= 30 minutes.

## Rollback Procedure

1. Apply rollback runtime profile:
   - `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply`
2. Run canonical gate:
   - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
3. Record incident note + artifact paths in `research/breakthrough_lab/platform_compatibility/`.
4. Freeze promotion path and reopen in `iterate` mode until cause is resolved.

## Verification of Rollback Path

Validated evidence:

- `research/breakthrough_lab/platform_compatibility/week9_block5_rollback_20260208_035258.md`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_035317.json`

