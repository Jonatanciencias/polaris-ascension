# T3 Report - Week 8 Block 3 (Controlled Drift Campaign)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (rollback-safe drift guardrails)

## Summary
- Drift campaign executed on production benchmark path (`auto` vs `auto_t3_controlled`) with deterministic protocol.
- Scenarios: `cold`, `warm`, `warm_queue_pressure`.
- Scope sizes: `1400`, `1536`, `2048`.
- Protocol: `sessions=2`, `iterations=6`, `seed=42`.

Key outcomes (warm_queue_pressure):
- Auto avg GFLOPS mean: `676.397`
- T3 avg GFLOPS mean: `808.205`
- Delta T3 vs auto: `+19.487%`
- P95 latency delta (T3 vs auto): `-20.313%`
- T3 fallback rate: `0.000`
- Correctness failures: `0`
- Policy disabled: `false`

## Interpretation
- Guardrails stayed intact under controlled drift pressure with no correctness escapes.
- Fallback and disable controls remained inactive (`0`), indicating policy stability.
- Throughput and latency deltas under pressure were favorable versus static auto for this scope.

## Versioned Policy and Rollback Defaults
- Versioned policy introduced for Block 3:
  - `research/breakthrough_lab/t3_online_control/policy_hardening_block3.json`
- Rollback-safe defaults documented in:
  - `research/breakthrough_lab/t3_online_control/rollback_playbook_block3.md`

## Evidence
- `research/breakthrough_lab/t3_online_control/policy_hardening_block3.json`
- `research/breakthrough_lab/t3_online_control/rollback_playbook_block3.md`
- `research/breakthrough_lab/t3_online_control/run_week8_t3_drift_campaign.py`
- `research/breakthrough_lab/t3_online_control/week8_t3_drift_campaign_20260208_020148.json`
- `research/breakthrough_lab/t3_online_control/week8_t3_drift_campaign_20260208_020148.md`
- `research/breakthrough_lab/t3_online_control/results.json`
