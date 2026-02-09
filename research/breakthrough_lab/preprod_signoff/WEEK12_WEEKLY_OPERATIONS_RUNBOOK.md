# Week 12 Weekly Operations Runbook

## Scope

Operational weekly cadence for controlled production with policy-driven replay.

- Policy: `research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Automation entrypoint: `research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py`
- Split evaluation: `research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py`

## Weekly Checklist

1. Run canonical gate pre-check.
2. Run weekly automation replay (`1400`, `2048`, optionally `3072`).
3. Run weekly split Clover/rusticl evaluation.
4. Apply severity mapping from `WEEK12_WEEKLY_ALERT_SLA.json`.
5. If required, apply rollback and rerun canonical gate.
6. Close week with acta + decision JSON.

## Commands

### A) Weekly automation replay

```bash
./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py \
  --mode local \
  --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json \
  --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json \
  --baseline-path research/breakthrough_lab/week11_controlled_rollout/week11_block2_continuous_canary_20260209_005442.json \
  --sizes 1400 2048 3072 \
  --snapshots 8 \
  --sessions 2 \
  --iterations 8 \
  --pressure-size 896 \
  --pressure-iterations 3 \
  --pressure-pulses-per-snapshot 3 \
  --seed 13011 \
  --output-dir research/breakthrough_lab/week13_controlled_rollout \
  --output-prefix week13_block1_extended_controlled
```

### B) Weekly split Clover/rusticl

```bash
./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py \
  --seeds 412 512 \
  --sizes 1400 2048 3072 \
  --kernels auto_t3_controlled auto_t5_guarded \
  --sessions 1 \
  --iterations 8 \
  --pressure-size 896 \
  --pressure-iterations 3 \
  --pressure-pulses-per-seed 3 \
  --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json \
  --baseline-block3-path research/breakthrough_lab/week12_controlled_rollout/week12_block3_size3072_pilot_canary_20260209_012745.json \
  --output-dir research/breakthrough_lab/week12_controlled_rollout \
  --output-prefix week12_block4_combined_split_3072

./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py \
  --split-artifact research/breakthrough_lab/week12_controlled_rollout/week12_block4_combined_split_3072_*.json \
  --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json \
  --required-sizes 1400 2048 3072 \
  --min-rusticl-ratio 0.85 \
  --output-dir research/breakthrough_lab/week12_controlled_rollout \
  --output-prefix week12_block4_combined_split_3072_eval
```

## Rollback Protocol

1. Trigger criteria from `SEV1` in `WEEK12_WEEKLY_ALERT_SLA.json`.
2. Apply rollback:

```bash
research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh apply
```

3. Validate after rollback:

```bash
./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke
```

4. Mark weekly decision as `iterate` until rerun is `promote`.

## Escalation and Ownership

- See `research/breakthrough_lab/preprod_signoff/WEEK12_WEEKLY_ESCALATION_MATRIX.md`.
- Attach all JSON/MD artifacts into weekly acta before closing decision.
