# Experiment Card - T6 Quantum Offline

## Metadata

- Experiment ID: `t6-001`
- Track: `t6_quantum_offline`
- Owner: `track-t6`
- Status: `planned`

## Hypothesis

Quantum-inspired search can improve offline discovery of scheduling strategies, even if runtime execution remains classical.

## Method

- Restrict scope to offline optimization:
  - schedule/parameter search objective
  - no runtime quantum claims
- Compare search outcomes against classical baseline optimizer.

## Variables

- independent:
  - search heuristic family
  - objective weighting (peak, sustained, stability)
  - search budget
- controlled:
  - benchmark protocol and datasets

## Success Metrics

- top-1 schedule quality >= classical baseline
- search efficiency (time-to-top-k) not worse by > 20%
- reproducible candidate ranking in repeated runs

## Stop Rule

Stop if search quality remains below classical baseline after 3 budget cycles.

## Artifacts

- `results.json`
- search trace and ranking report
- decision note (promote/iterate/drop)

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
