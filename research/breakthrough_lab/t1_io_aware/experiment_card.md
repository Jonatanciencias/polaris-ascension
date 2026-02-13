# Experiment Card - T1 IO Aware

## Metadata

- Experiment ID: `t1-001`
- Track: `t1_io_aware`
- Owner: `track-t1`
- Status: `planned`

## Hypothesis

If kernel variants are optimized for data movement (not only FLOPs), then sustained throughput at target sizes will increase with stable variance.

## Method

- Baseline: run canonical baseline from `../BASELINE_RUNBOOK.md`.
- Implement 2-3 communication-aware kernel variants.
- Benchmark on sizes: `1400`, `2048`, `3072`.
- Sessions/iterations: `10x20`, seed `42`.

## Variables

- independent:
  - tile size
  - local workgroup dimensions
  - vector load/store strategy
- controlled:
  - hardware, driver, seed, benchmark protocol

## Success Metrics

- `delta_vs_baseline_percent >= 10`
- `max_error <= 1e-3`
- `cv_peak <= 0.03`

## Stop Rule

Stop after 3 variants if no variant reaches at least `+5%` at 1400 or 2048.

## Artifacts

- `results.json`
- benchmark JSON/MD reports in `results/benchmark_reports/`
- short decision report (`promote|iterate|drop`)

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
