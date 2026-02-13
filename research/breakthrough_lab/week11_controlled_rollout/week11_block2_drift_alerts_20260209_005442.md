# Week 11 Block 2 - Drift Alerts T3/T5

- Date: 2026-02-09T00:55:25.974818+00:00
- Source canary: `research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json`
- Snapshots: 6

## Thresholds

- `throughput_drift_abs_percent_max`: `3.0`
- `p95_drift_percent_max`: `8.0`
- `t3_fallback_max`: `0.08`
- `t5_overhead_percent_max`: `3.0`
- `t5_disable_events_total_max`: `0`

## Rows

| Kernel | Size | Throughput drift % | P95 drift % | T3 fallback max | T5 overhead max % | T5 disable total | Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 0.1263 | 1.0117 | 0.0000 | - | - | 0 |
| auto_t3_controlled | 2048 | -0.0937 | -0.0554 | 0.0000 | - | - | 0 |
| auto_t5_guarded | 1400 | 0.1030 | -0.5128 | - | 2.8982 | 0 | 0 |
| auto_t5_guarded | 2048 | -0.1008 | 0.1122 | - | 0.6063 | 0 | 0 |

## Decision

- Alerts total: `0`
- Decision: `promote`

