# T5 Report - Week 5 Block 3 (Production Wiring + Auto-Disable Guardrails)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (controlled production wiring)

## Summary
- Block 3 wired T5 guardrails into production benchmark path via `auto_t5_guarded`.
- Campaign profile:
  - sizes: `1400`, `2048`
  - sessions: `8`
  - iterations/session: `16`
  - sampling mode: `periodic_8`
- Kernel avg GFLOPS mean: `844.417`.
- Effective overhead mean: `1.221%`.
- False positive rate: `0.000`.
- Correctness max error: `0.0005951` (`<=1e-3`).
- Disable events: `0`.

## Guardrail Outcome
- Policy used:
  - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json`
- Runtime guardrail checks (all pass):
  - `false_positive_rate <= 0.05`: pass (`0.000`)
  - `effective_overhead_percent <= 3.0`: pass (`1.221`)
  - `correctness_error <= 1e-3`: pass (`0.0005951`)
  - recall guardrails remain backed by policy evidence (`critical=1.000`, `uniform=0.967`)
- Auto-disable path remained armed and idle (`disable_events=0`).

## Interpretation
- T5 is now wired in the product benchmark path with active guardrails and deterministic auto-disable semantics.
- Controlled rerun confirms that operational safety is preserved while keeping overhead within policy limits.

## Evidence
- research/breakthrough_lab/t5_reliability_abft/week5_t5_production_wiring_20260207_234133.json
- research/breakthrough_lab/t5_reliability_abft/week5_t5_production_wiring_20260207_234133.md
- research/breakthrough_lab/t5_reliability_abft/run_week5_t5_production_wiring.py
- research/breakthrough_lab/t5_reliability_abft/policy_hardening_block3.json
- src/optimization_engines/t5_abft_guardrails.py
- src/benchmarking/production_kernel_benchmark.py
- src/cli.py
- research/breakthrough_lab/t5_reliability_abft/results.json
- research/breakthrough_lab/ACTA_WEEK5_BLOCK3_T5_PRODUCTION_WIRING_2026-02-07.md
