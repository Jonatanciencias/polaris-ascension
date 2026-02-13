# Experiment Card - T3 Online Control

## Metadata

- Experiment ID: `t3-001`
- Track: `t3_online_control`
- Owner: `track-t3`
- Status: `planned`

## Hypothesis

A contextual online selector can improve sustained performance and reduce worst-case runs under drift versus a static selector.

## Method

- Baseline selector: current static production selector.
- Prototype policy: contextual bandit with conservative fallback.
- Inputs:
  - matrix size tuple
  - recent latency and throughput windows
  - thermal/load proxy from runtime behavior

## Variables

- independent:
  - exploration rate
  - reward function
  - fallback thresholds
- controlled:
  - benchmark workload mix
  - reproducibility protocol

## Success Metrics

- mixed-workload mean uplift >= +5%
- p95 latency not worse than baseline by > 3%
- no correctness regression

## Stop Rule

Stop immediately if fallback triggers > 20% runs or correctness gate fails once.

## Artifacts

- `results.json`
- policy config snapshot
- comparison report static vs online policy

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
