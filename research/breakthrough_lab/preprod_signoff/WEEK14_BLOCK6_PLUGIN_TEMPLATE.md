# Plugin Starter Template (Week14 Block6)

## Metadata

- `plugin_id`:
- `owner`:
- `target_sizes`: [1400, 2048]
- `policy_path`:

## Required Commands

1. `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke`
2. `./venv/bin/python scripts/verify_drivers.py --json`
3. Controlled dry-run command for plugin profile

## Required Evidence

- `results.json` (schema-compatible)
- Execution summary `.md`
- Formal decision JSON (`promote|iterate|refine|stop`)

