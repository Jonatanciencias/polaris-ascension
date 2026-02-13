# Experiment Card - T4 Approximate GEMM

## Metadata

- Experiment ID: `t4-001`
- Track: `t4_approximate_gemm`
- Owner: `track-t4`
- Status: `planned`

## Hypothesis

For compressible inputs, bounded-error approximate GEMM can deliver significant speedup with an explicit error contract and safe fallback.

## Method

- Baseline: exact production GEMM.
- Candidate methods:
  - low-rank randomized approximation
  - fallback to exact kernel when contract risk is high
- Evaluate on representative matrix families (dense random + compressible).

## Variables

- independent:
  - target rank / compression ratio
  - error budget
  - fallback trigger threshold
- controlled:
  - same benchmark protocol and hardware

## Success Metrics

- speedup >= 1.5x on compressible cases
- contract respected in >= 95% runs
- automatic fallback prevents severe error outliers

## Stop Rule

Stop if contract violation exceeds 5% runs on target workload.

## Artifacts

- `results.json`
- contract compliance summary
- fallback trigger statistics

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
