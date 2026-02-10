# Week14 Block4 Plugins / Base Projects Checklist

## Extension Contracts

- [ ] Plugin declares deterministic seed protocol for benchmarks.
- [ ] Plugin exposes explicit fallback behavior (`promote|iterate|stop` compatible).
- [ ] Plugin preserves correctness contract (`max_error <= 1e-3`).
- [ ] Plugin emits machine-readable artifact (JSON + summary MD).
- [ ] Plugin is compatible with canonical gate (`scripts/run_validation_suite.py`).

## Runtime Compatibility

- [ ] Works on RX590 Clover baseline.
- [ ] Handles optional rusticl split mode without hard failure.
- [ ] Honors rollback environment profile (`week9_block5_rusticl_rollback.sh`).

## Base Project Integration

- [ ] Inference project: integrates selector/guardrails without bypassing policy.
- [ ] Benchmark project: logs gflops, p95, fallback, disable_events.
- [ ] Operations project: supports weekly cadence + monthly audit path.
- [ ] CI project: validates schema and canonical tier before merge.

## Documentation Minimum

- [ ] README includes activation/deactivation instructions.
- [ ] Rollback instructions tested in dry-run mode.
- [ ] Known limits and risk notes documented.

