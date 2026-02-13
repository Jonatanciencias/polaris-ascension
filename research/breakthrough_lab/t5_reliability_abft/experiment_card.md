# Experiment Card - T5 Reliability ABFT

## Metadata

- Experiment ID: `t5-001`
- Track: `t5_reliability_abft`
- Owner: `track-t5`
- Status: `planned`

## Hypothesis

ABFT-lite checks can detect critical GEMM output anomalies with low overhead and improve production trustworthiness.

## Method

- Implement checksum-based verification pass for selected runs.
- Compare:
  - detection quality
  - overhead on latency and throughput
- Test on normal runs and injected fault scenarios.

## Variables

- independent:
  - checksum granularity
  - sampling rate (always vs periodic)
  - correction strategy (detect-only vs recover)
- controlled:
  - benchmark protocol
  - workload sizes

## Success Metrics

- anomaly detection recall >= 95% on injected faults
- overhead <= 5% on mean throughput
- no false-negative critical failures in campaign

## Stop Rule

Stop if overhead exceeds 8% without clear reliability gain.

## Artifacts

- `results.json`
- fault-injection report
- overhead report

## Gate Reference

Before promotion, pass `../PROMOTION_GATE_CHECKLIST.md`.
