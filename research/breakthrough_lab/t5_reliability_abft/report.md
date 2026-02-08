# T5 Report - Week 8 Block 5 (Reliability Maturation)

- Status: completed
- Decision: promote
- Promotion gate: PASSED (fault-injection maturation + recall/overhead tuning)

## Summary
- Campaign mode: baseline policy (`block3`) vs candidate policy (`block5`) under deterministic fault-injection replay.
- Scope:
  - sizes: `1400`, `2048`
  - sessions: `10`
  - iterations/session: `24`
  - sampling mode: `periodic_8`
  - faults modelled per matrix: `2`

Baseline vs candidate:
- Uniform recall: `0.967 -> 0.983` (`+0.017`)
- Critical recall: `1.000 -> 1.000`
- Effective overhead: `1.165% -> 1.230%` (`+0.065%`)
- False positive rate: `0.000 -> 0.000`
- Correctness max error: `0.0005646` (`<=1e-3`)

## Guardrail Outcome
- Candidate policy:
  - `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json`
- Checks (all pass):
  - `candidate_correctness <= 1e-3`
  - `candidate_false_positive_rate <= 0.05`
  - `candidate_effective_overhead_percent <= 2.5`
  - `candidate_critical_recall >= 0.99`
  - `candidate_uniform_recall >= 0.97`
  - `uniform_recall_delta_vs_baseline >= 0.01`
  - `overhead_delta_vs_baseline <= 0.20`

## Interpretation
- Block 5 meets the objective: improved non-critical fault detection (`uniform_random`) with bounded overhead and preserved critical safety.
- Candidate policy is suitable for the next controlled production gate.

## Evidence
- `research/breakthrough_lab/t5_reliability_abft/policy_hardening_block5.json`
- `research/breakthrough_lab/t5_reliability_abft/run_week8_t5_maturation.py`
- `research/breakthrough_lab/t5_reliability_abft/week8_t5_maturation_20260208_022633.json`
- `research/breakthrough_lab/t5_reliability_abft/week8_t5_maturation_20260208_022633.md`
- `research/breakthrough_lab/t5_reliability_abft/results.json`
- `research/breakthrough_lab/ACTA_WEEK8_BLOCK5_T5_RELIABILITY_MATURATION_2026-02-08.md`
