# Week14 Block6 Extension Contracts

## Contract A - Benchmark Interface

- Entry point: `src/benchmarking/production_kernel_benchmark.py`
- Required output fields:
  - throughput (`avg_gflops`, `peak_gflops`), latency (`time_ms`), correctness (`max_error`).
- Deterministic controls:
  - fixed seed path, explicit platform selector, explicit policy paths.

## Contract B - Validation Gate

- Entry point: `scripts/run_validation_suite.py --tier canonical --driver-smoke`
- Required status: `promote` before merge/promotion.
- Required checks:
  - schema validation green, pytest tier green, verify_drivers JSON parse green.

## Contract C - Runtime Health

- Entry point: `scripts/verify_drivers.py --json`
- Required field: `overall_status=good` on target host.
- Runtime split support:
  - Clover baseline required; rusticl split optional but encouraged.

## Contract D - Safety and Rollback

- Rollback entry: `research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh`
- SLA reference: `research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md`
- Any correctness breach or disable-event spike forces `iterate/no-go` state.

