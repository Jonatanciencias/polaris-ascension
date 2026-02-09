# Acta Week 13 - Block 2 (Replay semanal automatizado + split Clover/rusticl)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar replay semanal automatizado post-Block 1 sobre `1400/2048/3072`,
  - ejecutar split Clover/rusticl con el mismo perfil y guardrails,
  - cerrar bloque con evaluacion formal de policy y gate canonico obligatorio.

## Objetivo

1. Confirmar estabilidad semanal con policy formal y drift acotado.
2. Confirmar portabilidad Clover/rusticl con ratio saludable en tamanos criticos.
3. Cerrar Week 13 Block 2 con decision formal `promote|iterate|refine|stop`.

## Ejecucion Formal

Replay semanal automatizado:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-path research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json --sizes 1400 2048 3072 --snapshots 6 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --seed 14021 --output-dir research/breakthrough_lab/week13_controlled_rollout --output-prefix week13_block2_weekly_replay`
  - Artifact JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_20260209_015702.json`
  - Artifact MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_20260209_015702.md`
  - Decision: `promote`

Canary y evaluacion derivados del replay:

- Canary JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json`
- Canary MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.md`
- Eval JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_eval_20260209_020050.json`
- Eval MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_eval_20260209_020050.md`
- Decision (eval): `promote`

Split Clover/rusticl:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 612 712 --sizes 1400 2048 3072 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-seed 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-block3-path research/breakthrough_lab/week13_controlled_rollout/week13_block1_extended_controlled_canary_20260209_014522.json --output-dir research/breakthrough_lab/week13_controlled_rollout --output-prefix week13_block2_platform_split`
  - Artifact JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_20260209_020302.json`
  - Artifact MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_20260209_020302.md`
  - Decision: `promote`

Evaluacion split contra policy formal:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py --split-artifact research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_20260209_020302.json --policy-path research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json --min-rusticl-ratio 0.85 --required-sizes 1400 2048 3072 --output-dir research/breakthrough_lab/week13_controlled_rollout --output-prefix week13_block2_platform_split_eval`
  - Artifact JSON: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_eval_20260209_020309.json`
  - Artifact MD: `research/breakthrough_lab/week13_controlled_rollout/week13_block2_platform_split_eval_20260209_020309.md`
  - Decision: `promote`

Gate canonico obligatorio de cierre:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_020335.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_020335.md`
  - Decision: `promote`

## Resultados

Replay semanal (6 snapshots):

- `executed_snapshots = 6`
- `rollback_triggered = false`
- `max_error = 0.0008697509765625`
- `t3_fallback_max = 0.0`
- `t5_disable_events_total = 0`
- `t5_overhead_max = 1.3788347389875628%`
- `policy_eval_decision = promote`

Split Clover/rusticl:

- `split_decision = promote`
- `split_eval_decision = promote`
- `split_eval_failed_checks = []`
- `split_max_error = 0.0008697509765625`
- `split_t5_overhead_max = 1.4717036730802733%`
- `split_t5_disable_events_total = 0`
- ratio minimo rusticl/clover observado (avg): `0.9227649050049238` (`>= 0.85` requerido)
- `required_sizes_present_on_split = pass` para `1400/2048/3072`

Gate canonico final:

- `decision = promote`

## Decision Formal

Tracks:

- `week13_block2_weekly_replay_automation`: **promote**
- `week13_block2_platform_split_execution`: **promote**
- `week13_block2_platform_split_policy_eval`: **promote**
- `week13_block2_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El replay semanal post-Block 1 mantiene guardrails y SLO en `promote`.
- El split Clover/rusticl mantiene portabilidad con ratios por encima del umbral formal.
- El gate canonico final permanece `promote`.

## Estado del Bloque

`Week 13 - Block 2` cerrado en `promote`.
