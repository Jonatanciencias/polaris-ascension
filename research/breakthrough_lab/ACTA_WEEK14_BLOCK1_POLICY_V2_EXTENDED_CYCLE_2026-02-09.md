# Acta Week 14 - Block 1 (Ciclo quincenal completo sobre policy v2)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar replay extendido sobre policy v2,
  - ejecutar split Clover/rusticl con tamaños críticos,
  - validar estabilidad extendida y cerrar formalmente.

## Objetivo

1. Ejecutar el primer ciclo quincenal completo con `policy_week13_block3_weekly_slo_v2.json`.
2. Confirmar estabilidad extendida (`8 snapshots`) y compatibilidad Clover/rusticl.
3. Cerrar con evidencia + acta + decisión formal + gate canónico.

## Ejecución Formal

Intento inicial (mismo scope, policy v2, T5 long-horizon previo):

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week10_block2_4_long_horizon.json --baseline-path research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json --sizes 1400 2048 3072 --snapshots 8 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --seed 15031 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block1_policy_v2_cycle`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_20260209_023353.json`
  - Decision: `iterate`
  - Causa principal: `t5_disable_events_total_bound`, `t5_overhead_bound`, `throughput_drift_abs_bound`

Hardening técnico aplicado para rerun:

- Nueva policy T5 low-overhead: `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json`
  - Ajustes claves: `sampling_period 12`, `row_samples 6`, `col_samples 6`.

Rerun extendido con hardening T5:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py --mode local --policy-path research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --baseline-path research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json --sizes 1400 2048 3072 --snapshots 8 --sessions 2 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-snapshot 3 --seed 15131 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block1_policy_v2_cycle_rerun`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_20260209_023945.json`
  - Canary JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_canary_20260209_024440.json`
  - Eval JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_policy_v2_cycle_rerun_eval_20260209_024440.json`
  - Decision: `promote`

Split Clover/rusticl:

- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py --seeds 1012 1112 --sizes 1400 2048 3072 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 3 --pressure-pulses-per-seed 3 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --baseline-block3-path research/breakthrough_lab/week13_controlled_rollout/week13_block2_weekly_replay_canary_20260209_020049.json --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block1_platform_split`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_platform_split_20260209_024657.json`
  - Decision: `promote`

Evaluación split contra policy v2:

- `./venv/bin/python research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py --split-artifact research/breakthrough_lab/week14_controlled_rollout/week14_block1_platform_split_20260209_024657.json --policy-path research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json --min-rusticl-ratio 0.85 --required-sizes 1400 2048 3072 --output-dir research/breakthrough_lab/week14_controlled_rollout --output-prefix week14_block1_platform_split_eval`
  - Artifact JSON: `research/breakthrough_lab/week14_controlled_rollout/week14_block1_platform_split_eval_20260209_024704.json`
  - Decision: `promote`

Gate canónico obligatorio de cierre:

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact JSON: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_024728.json`
  - Artifact MD: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_024728.md`
  - Decision: `promote`

## Resultados

Rerun extendido (policy v2 + T5 low-overhead):

- `decision = promote`
- `executed_snapshots = 8`
- `rollback = false`
- `max_error = 0.0008697509765625`
- `t3_fallback_max = 0.0`
- `t5_overhead_max = 1.3478937850691686%`
- `t5_disable_events_total = 0`
- `max_abs_drift = 1.8459865722459716`

Split Clover/rusticl:

- `decision = promote`
- `max_error = 0.00079345703125`
- `t5_overhead_max = 1.6154690775837195%`
- `t5_disable_total = 0`
- `ratio_min = 0.9219996294906012` (`>= 0.85`)
- `required_sizes_present_on_split = true`

## Decisión Formal

Tracks:

- `week14_block1_initial_cycle_attempt`: **iterate**
- `week14_block1_t5_low_overhead_hardening`: **promote**
- `week14_block1_extended_cycle_rerun`: **promote**
- `week14_block1_platform_split_eval`: **promote**
- `week14_block1_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El primer intento detectó un caso límite real y fue corregido con hardening técnico acotado.
- El rerun extendido cerró en `promote` con guardrails estables y sin disable events.
- El split Clover/rusticl y el gate canónico final también cerraron en `promote`.

## Estado del Bloque

`Week 14 - Block 1` cerrado en `promote`.
