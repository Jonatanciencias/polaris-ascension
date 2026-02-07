# Breakthrough Lab

This directory is the experimental lane for high-risk ideas.

Rules:
- no direct production claims from this lane
- every experiment must include hypothesis, metrics, and stop rule
- only promoted techniques move to `src/` production paths

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
