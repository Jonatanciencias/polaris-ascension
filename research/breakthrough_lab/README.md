# Breakthrough Lab

This directory is the experimental lane for high-risk ideas.

Rules:
- no direct production claims from this lane
- every experiment must include hypothesis, metrics, and stop rule
- only promoted techniques move to `src/` production paths
- use canonical baseline from `BASELINE_RUNBOOK.md`
- promotion decisions must pass `PROMOTION_GATE_CHECKLIST.md`

Suggested track layout:
- `t1_io_aware/` - communication/roofline guided kernels
- `t2_auto_scheduler/` - automated schedule search
- `t3_online_control/` - adaptive runtime selector policies
- `t4_approximate_gemm/` - bounded-error fast paths
- `t5_reliability_abft/` - fault-tolerant verification layers
- `t6_quantum_offline/` - offline quantum-inspired search

Minimum artifact per experiment:
- `experiment_card.md` (hypothesis, design, expected gain)
- `results.json` (machine-readable metrics)
- `report.md` (decision: promote, iterate, or drop)

Shared standards:
- `results.schema.json` - canonical results contract
- `results.template.json` - reusable starting payload
- `WEEK1_BACKLOG_PHASE0.md` - executable atomic task list

Contract validation:
- Local: `python scripts/validate_breakthrough_results.py`
- Unified local/CI runner: `python scripts/run_validation_suite.py --tier canonical --driver-smoke`
- CI: enforced in `.github/workflows/test-tiers.yml` (`cpu-fast` job via `scripts/run_validation_suite.py`)
