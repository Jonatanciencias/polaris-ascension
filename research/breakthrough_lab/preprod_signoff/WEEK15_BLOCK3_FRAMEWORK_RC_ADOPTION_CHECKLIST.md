# Week15 Block3 RC Adoption Checklist

## Repository Readiness

- [ ] Repository can run canonical gate locally.
- [ ] Repository can parse `verify_drivers.py --json` output.
- [ ] Repository consumes policy paths through explicit config/env.

## Runtime Readiness

- [ ] Clover baseline validated on RX590 host.
- [ ] rusticl split tested (optional but recommended).
- [ ] Rollback script tested in dry-run mode.

## Integration Readiness

- [ ] Plugin/project emits JSON+MD evidence per run.
- [ ] Project can store formal decision JSON per block.
- [ ] Project enforces `max_error <= 1e-3` and guardrail checks.

