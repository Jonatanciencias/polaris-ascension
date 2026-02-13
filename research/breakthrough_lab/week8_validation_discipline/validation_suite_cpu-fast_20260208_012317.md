# Validation Suite Report

- Date: 2026-02-08T01:23:17.433403+00:00
- Branch: `feat/breakthrough-roadmap-2026q1`
- Tier: `cpu-fast`
- Driver smoke enabled: `True`

## Command Status

- `validate_breakthrough_results.py`: rc=0
- `pytest` tier command: rc=0
- `pytest` counts: passed=17 failed=None skipped=None
- `verify_drivers.py --json`: rc=0
- Driver JSON parse: `True`
- Driver status: `good`

## Evaluation

| Check | Pass |
| --- | --- |
| results_schema_green | True |
| pytest_tier_green | True |
| verify_drivers_json_smoke | True |

## Decision

- Decision: `promote`
- Rationale: Validation runner checks passed for the selected tier.

